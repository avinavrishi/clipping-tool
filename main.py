import os
import subprocess
import textwrap
import json
import torch

import whisper
import yt_dlp
from transformers import pipeline
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# ============================================================
# LOAD MODELS (GPU if available in Colab)
# ============================================================

print("Loading models...")

device_id = 0 if torch.cuda.is_available() else -1
whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

sentiment_model = pipeline("sentiment-analysis",
                           model="distilbert-base-uncased-finetuned-sst-2-english",
                           device=device_id)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device_id
)

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device=device_id)

KEYWORDS = ["AI", "security", "funny", "important", "education", "technology"]

# ============================================================
# DOWNLOAD VIDEO
# ============================================================

def download_video(video_url, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": (
            "bestvideo[codec^=avc]+bestaudio[ext=m4a]/"
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
            "best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(output_folder, "%(title)s-%(id)s.%(ext)s")
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4"

    fixed_filename = os.path.splitext(filename)[0] + "_fixed.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", filename,
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
        fixed_filename
    ], check=True)

    return fixed_filename, info.get("duration", 0)


# ============================================================
# SPLIT VIDEO BY TIME
# ============================================================

def split_video(video_file, output_folder, segment_time):
    os.makedirs(output_folder, exist_ok=True)
    output_pattern = os.path.join(output_folder, "clip_%03d.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_file,
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-f", "segment", "-segment_time", str(segment_time),
        "-reset_timestamps", "1", output_pattern
    ]
    subprocess.run(cmd, check=True)
    print(f"\n✅ Clips saved to: {output_folder}")

# ============================================================
# SCENE DETECTION
# ============================================================

def detect_scenes(video_file, threshold=30.0):
    video = open_video(video_file)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()

def split_by_scenes(video_file, output_folder, scene_list, min_length=10, delete_original=True):
    os.makedirs(output_folder, exist_ok=True)
    for idx, (start, end) in enumerate(scene_list):
        duration = end.get_seconds() - start.get_seconds()
        if duration < min_length:
            print(f"⏩ Skipped scene {idx:03d} (too short: {duration:.1f}s)")
            continue

        output_path = os.path.join(output_folder, f"scene_{idx:03d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start.get_seconds()), "-to", str(end.get_seconds()),
            "-i", video_file,
            "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", output_path
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Saved {output_path}")

    if delete_original:
        try:
            os.remove(video_file)
            print(f"🗑️ Deleted original video: {video_file}")
        except Exception as e:
            print(f"⚠️ Could not delete {video_file}: {e}")


# ============================================================
# AUDIO INTENSITY
# ============================================================

def calculate_audio_intensity(video_path):
    cmd = ["ffmpeg", "-i", video_path, "-af", "volumedetect", "-f", "null", "-"]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if "mean_volume" in line:
            try:
                return float(line.split(":")[1].replace("dB", "").strip())
            except:
                pass
    return -30.0

# ============================================================
# SUBTITLES
# ============================================================

def add_subtitles(clip_path, transcript, output_path, line_width=40):
    wrapped = "\n".join(textwrap.wrap(transcript, width=line_width))
    srt_file = output_path.replace(".mp4", ".srt")
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:10:00,000\n")
        f.write(wrapped)
    cmd = ["ffmpeg", "-y", "-i", clip_path, "-vf", f"subtitles={srt_file}", "-c:a", "copy", output_path]
    subprocess.run(cmd, check=True)
    os.remove(srt_file)

# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe_clip(clip_path):
    result = whisper_model.transcribe(clip_path, fp16=False)
    return result["text"].strip()

# ============================================================
# FILTER & SAVE METADATA
# ============================================================

def filter_emotional_clips(clips_folder, final_folder, metadata_file="metadata.json"):
    os.makedirs(final_folder, exist_ok=True)
    emotional_clips = []
    metadata = []
    clips = sorted(f for f in os.listdir(clips_folder) if f.endswith(".mp4"))

    for file in clips:
        clip_path = os.path.join(clips_folder, file)
        print(f"\n🎤 Transcribing {file}...")
        transcript = transcribe_clip(clip_path)
        if len(transcript.split()) < 5:
            print("❌ Not enough speech")
            continue

        sentiment = sentiment_model(transcript)[0]
        label, score = sentiment["label"], sentiment["score"]
        intensity = calculate_audio_intensity(clip_path)
        highlight_score = score + len(transcript.split()) / 50 + intensity / 100

        summary = summarizer(transcript, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
        keyword_hits = [kw for kw in KEYWORDS if kw.lower() in transcript.lower()]
        topics = classifier(transcript, candidate_labels=["technology", "comedy", "politics", "personal story", "education"])
        top_topic = topics["labels"][0]

        if score > 0.7 or intensity > -20 or keyword_hits:
            output_path = os.path.join(final_folder, file)
            add_subtitles(clip_path, transcript, output_path)
            clip_data = {
                "file": file,
                "transcript": transcript,
                "summary": summary,
                "sentiment": label,
                "sentiment_score": score,
                "intensity_db": intensity,
                "highlight_score": highlight_score,
                "keywords": keyword_hits,
                "topic": top_topic
            }
            metadata.append(clip_data)
            emotional_clips.append((file, transcript, label, score, intensity, highlight_score))
            print(f"✅ KEPT | {label} {score:.2f} | Topic: {top_topic} | Keywords: {keyword_hits}")
        else:
            print("❌ Skipped")

    with open(os.path.join(final_folder, metadata_file), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    emotional_clips.sort(key=lambda x: x[5], reverse=True)
    print(f"\n🎯 Metadata saved to {os.path.join(final_folder, metadata_file)}")
    return emotional_clips

# ============================================================
# MAIN
# ============================================================

def main():
    video_url = input("Enter YouTube URL: ").strip()
    folder_name = input("Folder Name: ").strip()
    mode = input("Mode (time/scene): ").strip().lower()

    base_folder = os.path.join("/content", folder_name)  # Colab path
    os.makedirs(base_folder, exist_ok=True)

    video_file, duration = download_video(video_url, base_folder)
    clips_folder = os.path.join(base_folder, "clips")

    if mode == "time":
        segment_time = int(input("Segment Length (sec): "))
        print(f"\nVideo Duration: {duration} sec")
        print(f"Approx Clips: {duration // segment_time}")
        split_video(video_file, clips_folder, segment_time)
    elif mode == "scene":
        print("\n🔍 Detecting scenes...")
        scene_list = detect_scenes(video_file)
        print(f"Found {len(scene_list)} scenes")
        split_by_scenes(video_file, clips_folder, scene_list)
    else:
        print("Invalid mode")
        return

    final_folder = os.path.join(base_folder, "final")
    emotional = filter_emotional_clips(clips_folder, final_folder)

    print("\n🏆 TOP HIGHLIGHTS\n")
    for item in emotional:
        print(f"{item[0]} | {item[2]} | {item[3]:.2f} | {item[4]:.1f} dB | Score {item[5]:.2f}")

if __name__ == "__main__":
    main()
