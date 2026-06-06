import os
import re
import subprocess
import yt_dlp

try:
    import whisper
except ImportError:
    whisper = None

# ============================================================
# FFmpeg Location
# ============================================================

FFMPEG_LOCATION = (
    r"C:\Users\Rishi\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.1-essentials_build\bin"
)

FFMPEG_EXE = os.path.join(FFMPEG_LOCATION, "ffmpeg.exe")
FFPROBE_EXE = os.path.join(FFMPEG_LOCATION, "ffprobe.exe")

# ============================================================
# Helpers
# ============================================================

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "_", name)

def get_video_info(url):
    with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
        return ydl.extract_info(url, download=False)

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_video(video_path, title, model_name="base"):
    if whisper is None:
        print("\nWhisper is not installed. Install openai-whisper to enable transcription.")
        return None

    print("\nTranscribing audio to text...")
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, word_timestamps=True, verbose=False)
    text = result.get("text", "").strip()

    output_folder = os.path.dirname(video_path)
    transcript_path = os.path.join(output_folder, f"{title}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        if result.get("segments"):
            for segment in result["segments"]:
                words = segment.get("words") or []
                if words:
                    for word_info in words:
                        word_text = word_info.get("word", "").strip()
                        if not word_text:
                            continue
                        f.write(
                            f"{format_timestamp(word_info['start'])} --> {format_timestamp(word_info['end'])}\t{word_text}\n"
                        )
                else:
                    f.write(
                        f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\t{segment.get('text', '').strip()}\n"
                    )
        else:
            f.write(text)

    if result.get("segments"):
        srt_path = os.path.join(output_folder, f"{title}_transcript.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for index, segment in enumerate(result["segments"], start=1):
                f.write(f"{index}\n")
                f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
                f.write(segment.get("text", "").strip() + "\n\n")
        print("SRT transcript saved:")
        print(srt_path)

    print("Text transcript saved:")
    print(transcript_path)
    return transcript_path


def list_qualities(info):
    resolutions = set()
    for fmt in info.get("formats", []):
        if fmt.get("height") and fmt.get("vcodec") != "none":
            resolutions.add(fmt["height"])
    return sorted(resolutions)

def print_audio_formats(info):
    print("\n========== AUDIO STREAMS ==========\n")
    found = False
    for fmt in info.get("formats", []):
        if fmt.get("acodec") and fmt.get("acodec") != "none":
            found = True
            print(
                f"ID={fmt.get('format_id')} | "
                f"ACODEC={fmt.get('acodec')} | "
                f"ABR={fmt.get('abr')} | "
                f"EXT={fmt.get('ext')}"
            )
    if not found:
        print("No audio streams found")
    print("\n===================================\n")

def inspect_file(video_path):
    print("\n========== FINAL FILE STREAMS ==========\n")
    try:
        result = subprocess.run(
            [FFPROBE_EXE, "-v", "error", "-show_streams", video_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        print(result.stdout)
    except Exception as e:
        print("Inspection failed:", e)
    print("\n========================================\n")

def find_downloaded_file(download_info, video_folder):
    requested_downloads = download_info.get("requested_downloads") or []
    for item in requested_downloads:
        filepath = item.get("filepath")
        if filepath and os.path.exists(filepath):
            return filepath

    candidates = []
    for file in os.listdir(video_folder):
        full_path = os.path.join(video_folder, file)
        if os.path.isfile(full_path) and file.startswith("source."):
            candidates.append(full_path)

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)

# ============================================================
# Download Video
# ============================================================

def download_video(url, selected_height):
    info = get_video_info(url)
    title = sanitize_filename(info["title"])

    project_root = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(project_root, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    video_folder = os.path.join(downloads_dir, title)
    os.makedirs(video_folder, exist_ok=True)

    print_audio_formats(info)

    temp_output = os.path.join(video_folder, "source.%(ext)s")

    # Download one merged video+audio file. yt-dlp may fetch video/audio
    # separately internally, but the filepath below points to the merged result.
    format_selector = (
        f"bestvideo[height<={selected_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={selected_height}]+bestaudio/"
        f"best[height<={selected_height}]/best"
    )

    print("\nUsing format selector:")
    print(format_selector)

    ydl_opts = {
        "format": format_selector,
        "ffmpeg_location": FFMPEG_LOCATION,
        "outtmpl": temp_output,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "keepvideo": False,
        "quiet": False,
        "verbose": True,
    }

    print("\nDownloading...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        downloaded_info = ydl.extract_info(url, download=True)

    source_file = find_downloaded_file(downloaded_info, video_folder)
    if not source_file:
        raise FileNotFoundError("Downloaded file not found.")

    final_output = os.path.join(video_folder, f"{title}_final.mp4")

    print("\nRe-encoding to H.264 + AAC...")

    command = [
        FFMPEG_EXE, "-y",
        "-i", source_file,
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        final_output
    ]

    subprocess.run(command, check=True)

    print("\nFinal file created:")
    print(final_output)

    inspect_file(final_output)

    transcript_path = transcribe_video(final_output, title)
    if transcript_path:
        print("\nTranscript generation complete:")
        print(transcript_path)
    else:
        print("\nTranscript generation was not created.")

    return final_output

# ============================================================
# Main
# ============================================================

def main():
    print("\n=== YouTube Downloader ===\n")
    url = input("Enter YouTube URL: ").strip()
    if not url:
        print("No URL entered")
        return

    info = get_video_info(url)
    print("\nTitle:", info.get("title"))
    print("Channel:", info.get("uploader"))

    qualities = list_qualities(info)
    print("\nAvailable qualities:\n")
    for index, quality in enumerate(qualities, start=1):
        print(f"{index}. {quality}p")

    try:
        choice = int(input("\nSelect quality: "))
        selected_height = qualities[choice - 1]
    except Exception:
        print("Invalid selection")
        return

    output = download_video(url, selected_height)
    print("\nDownload complete:")
    print(output)

if __name__ == "__main__":
    main()
