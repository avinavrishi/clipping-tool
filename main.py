import os
import subprocess
import yt_dlp
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from google.colab import drive

# ============================================================
# MOUNT GOOGLE DRIVE
# ============================================================

drive.mount('/content/drive')

# ============================================================
# DOWNLOAD VIDEO (robust settings)
# ============================================================

def download_video(video_url, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(output_folder, "%(title)s-%(id)s.%(ext)s"),
        "socket_timeout": 60,
        "retries": 10,
        "continuedl": True,
        "fragment_retries": 10,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "cookies": "cookies.txt"  # upload cookies.txt to Colab
    }

    info = None
    filename = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    if not filename:
        raise RuntimeError("No filename returned by yt-dlp")

    filename = os.path.normpath(filename)
    if not filename.endswith(".mp4"):
        filename = os.path.splitext(filename)[0] + ".mp4"

    fixed_filename = os.path.normpath(os.path.splitext(filename)[0] + "_fixed.mp4")

    # ✅ GPU encoding with h264_nvenc
    subprocess.run([
        "ffmpeg", "-y", "-i", filename,
        "-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "2M",
        "-c:a", "aac", "-b:a", "128k",
        fixed_filename
    ], check=True)

    return fixed_filename, info.get("duration", 0) if info else 0

# ============================================================
# SPLIT VIDEO BY TIME    
# ============================================================

def split_video(video_file, output_folder, segment_time):
    os.makedirs(output_folder, exist_ok=True)
    output_pattern = os.path.normpath(os.path.join(output_folder, "clip_%03d.mp4"))
    cmd = [
        "ffmpeg", "-y", "-i", video_file,
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        "-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "2M",
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

        output_path = os.path.normpath(os.path.join(output_folder, f"scene_{idx:03d}.mp4"))
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start.get_seconds()), "-to", str(end.get_seconds()),
            "-i", video_file,
            "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
            "-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "2M",
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
# MAIN
# ============================================================

def main():
    video_url = input("Enter YouTube URL: ").strip()
    folder_name = input("Folder Name: ").strip()

    print("\nChoose splitting mode:")
    print("1. Split by fixed time segments")
    print("2. Split by scene detection")
    choice = input("Enter 1 or 2: ").strip()

    # ✅ Store clips in Google Drive → Youtube clip folder
    base_folder = os.path.normpath(os.path.join("/content/drive/MyDrive/Youtube clip", folder_name))
    os.makedirs(base_folder, exist_ok=True)

    video_file, duration = download_video(video_url, base_folder)
    clips_folder = os.path.join(base_folder, "clips")

    if choice == "1":
        segment_time = int(input("Segment Length (sec): "))
        print(f"\nVideo Duration: {duration} sec")
        print(f"Approx Clips: {duration // segment_time}")
        split_video(video_file, clips_folder, segment_time)

    elif choice == "2":
        print("\n🔍 Detecting scenes...")
        scene_list = detect_scenes(video_file)
        print(f"Found {len(scene_list)} raw scenes")
        split_by_scenes(video_file, clips_folder, scene_list)

    else:
        print("Invalid choice")
        return

    print(f"\n✅ All clips saved in Google Drive: {clips_folder}")

if __name__ == "__main__":
    main()
