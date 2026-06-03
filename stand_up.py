import yt_dlp
import os
import subprocess
import whisper
import textwrap
from transformers import pipeline

# Load HuggingFace sentiment + emotion models
sentiment_model = pipeline("sentiment-analysis")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def download_comedy_video(video_url, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_folder, '%(title)s-%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info_dict)
    return filename, info_dict.get("duration", None)

def split_into_segments(input_file, output_folder, segment_time=30):
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        "-c:a", "copy",
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-reset_timestamps", "1",
        os.path.join(output_folder, "clip_%03d.mp4")
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Comedy video split into {segment_time}-second clips!")

def add_subtitles(clip_path, transcript, output_path, line_width=40):
    wrapped_text = "\n".join(textwrap.wrap(transcript, width=line_width))
    srt_file = output_path.replace(".mp4", ".srt")
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:10,000\n")
        f.write(wrapped_text + "\n")
    cmd = [
        "ffmpeg",
        "-i", clip_path,
        "-vf", f"subtitles={srt_file}:force_style='Fontsize=28,PrimaryColour=&HFFFFFF&'",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(srt_file)

def filter_funny_clips(clips_folder, final_folder):
    os.makedirs(final_folder, exist_ok=True)
    model = whisper.load_model("base")
    funny_clips = []
    for file in sorted(os.listdir(clips_folder)):
        if file.endswith(".mp4"):
            clip_path = os.path.join(clips_folder, file)
            result = model.transcribe(clip_path)
            text = result["text"].strip()

            if len(text.split()) >= 5:
                sentiment = sentiment_model(text)[0]
                emotion = emotion_model(text)[0]

                label, score = sentiment["label"], sentiment["score"]
                emo_label, emo_score = emotion["label"], emotion["score"]

                # Keep clips with humor/emotion
                if emo_label in ["joy", "amusement", "surprise"] and emo_score > 0.6:
                    output_path = os.path.join(final_folder, file)
                    add_subtitles(clip_path, text, output_path)
                    funny_clips.append((file, text, emo_label, emo_score))
                    print(f"😂 Kept {file} ({emo_label}, {emo_score:.2f}) → {text[:60]}...")
                else:
                    print(f"❌ Skipped {file} (not funny/emotional enough)")
            else:
                print(f"❌ Skipped {file} (too short)")
    print(f"🎯 Final stand-up comedy highlights saved in {final_folder}")
    return funny_clips

def main():
    video_url = input("Enter the stand-up comedy video URL: ").strip()
    folder_name = input("Enter the folder name to save this video: ").strip()
    segment_time = int(input("Enter segment length in seconds (default 30): ").strip() or 30)

    base_folder = os.path.join(r"D:\comedy_downloads", folder_name)
    os.makedirs(base_folder, exist_ok=True)

    video_file, duration = download_comedy_video(video_url, base_folder)
    clips_folder = os.path.join(base_folder, "clips")

    split_into_segments(video_file, clips_folder, segment_time)

    final_folder = os.path.join(base_folder, "final")
    funny = filter_funny_clips(clips_folder, final_folder)

    print("✅ Process complete! Funny highlights:")
    for fname, transcript, emo_label, emo_score in funny:
        print(f"- {fname}: ({emo_label}, {emo_score:.2f}) {transcript[:60]}...")

if __name__ == "__main__":
    main()
