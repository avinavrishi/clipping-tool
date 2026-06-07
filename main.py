import os
import subprocess

import yt_dlp

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

# ============================================================
# DOWNLOAD VIDEO
# ============================================================

def download_video(video_url, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best"
        ),
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

    return filename


# ============================================================
# SCENE DETECTION
# ============================================================

def detect_scenes(video_file, threshold=30):

    print("\nDetecting scenes...")

    video = open_video(video_file)

    scene_manager = SceneManager()

    scene_manager.add_detector(
        ContentDetector(threshold=threshold)
    )

    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()

    print(f"Found {len(scenes)} scenes")

    return scenes


# ============================================================
# SPLIT SCENES
# ============================================================

def split_by_scenes(
        video_file,
        output_folder,
        scenes,
        min_duration=5):

    clips_folder = os.path.join(
        output_folder,
        "clips"
    )

    os.makedirs(
        clips_folder,
        exist_ok=True
    )

    clip_count = 0

    for idx, (start, end) in enumerate(scenes):

        start_sec = start.get_seconds()
        end_sec = end.get_seconds()

        duration = end_sec - start_sec

        if duration < min_duration:
            continue

        output_path = os.path.join(
            clips_folder,
            f"scene_{clip_count:03d}.mp4"
        )

        cmd = [
            "ffmpeg",
            "-y",

            "-ss", str(start_sec),
            "-to", str(end_sec),

            "-i", video_file,

            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",

            "-c:a", "aac",
            "-b:a", "128k",

            output_path
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        clip_count += 1

        print(
            f"Saved: scene_{clip_count:03d}.mp4"
        )

    print(
        f"\nFinished. Created {clip_count} clips."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===== VIDEO SCENE SPLITTER =====\n")

    video_url = input(
        "Enter YouTube URL: "
    ).strip()

    project_name = input(
        "Project Name: "
    ).strip()

    base_folder = os.path.join(
        r"D:\Downloads\ClipData",
        project_name
    )

    os.makedirs(
        base_folder,
        exist_ok=True
    )

    print(
        f"\nSaving to:\n{base_folder}"
    )

    print("\nDownloading video...")

    video_file = download_video(
        video_url,
        base_folder
    )

    print("\nDownload complete.")

    scenes = detect_scenes(
        video_file
    )

    split_by_scenes(
        video_file,
        base_folder,
        scenes
    )

    print("\nDone!")


if __name__ == "__main__":
    main()