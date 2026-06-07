import os
import json
import textwrap
import subprocess

import whisper
import yt_dlp

from transformers import pipeline

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

whisper_model = whisper.load_model(
    "base",
    device="cpu"
)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

KEYWORDS = [
    "AI",
    "security",
    "funny",
    "important",
    "education",
    "technology"
]

# ============================================================
# DOWNLOAD VIDEO
# ============================================================

def download_video(video_url, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(
            output_folder,
            "%(title)s-%(id)s.%(ext)s"
        )
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            video_url,
            download=True
        )

        filename = ydl.prepare_filename(info)

        if not filename.endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4"

    return filename, info.get("duration", 0)

# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe_clip(video_path):

    result = whisper_model.transcribe(
        video_path,
        fp16=False
    )

    return result["text"].strip()

# ============================================================
# AUDIO LEVEL
# ============================================================

def calculate_audio_intensity(video_path):

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        text=True
    )

    for line in result.stderr.splitlines():

        if "mean_volume" in line:

            try:
                return float(
                    line.split(":")[1]
                    .replace("dB", "")
                    .strip()
                )
            except:
                pass

    return -30.0

# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(video_path):

    print("\nTranscribing...")

    transcript = transcribe_clip(video_path)

    print("\nTranscript:")
    print(transcript[:500])

    sentiment = sentiment_model(
        transcript
    )[0]

    topic = classifier(
        transcript,
        candidate_labels=[
            "technology",
            "education",
            "comedy",
            "politics",
            "personal story"
        ]
    )

    intensity = calculate_audio_intensity(
        video_path
    )

    words = transcript.split()

    if len(words) > 30:
        summary = " ".join(words[:30]) + "..."
    else:
        summary = transcript

    result = {
        "summary": summary,
        "sentiment": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "topic": topic["labels"][0],
        "audio_intensity": intensity
    }

    return result

# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===== VIDEO CLIPPING TOOL =====\n")

    video_url = input(
        "Enter YouTube URL: "
    ).strip()

    project_name = input(
        "Project Name: "
    ).strip()

    # Fixed download location
    base_folder = os.path.join(
        "D:\\Downloads\\ClipData",
        project_name
    )

    os.makedirs(
        base_folder,
        exist_ok=True
    )

    print(f"\nOutput Folder: {base_folder}")

    print("\nDownloading video...")

    video_file, duration = download_video(
        video_url,
        base_folder
    )

    print(
        f"\nDownloaded. Duration: {duration} seconds"
    )

    result = analyze_video(
        video_file
    )

    print("\n===== RESULTS =====\n")

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    metadata_file = os.path.join(
        base_folder,
        "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nMetadata saved:\n{metadata_file}"
    )

    print(
        f"\nVideo saved in:\n{base_folder}"
    )

if __name__ == "__main__":
    main()