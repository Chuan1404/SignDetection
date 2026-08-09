import os
import json
import time
from urllib.parse import urlparse

import requests

from config import ROOT

WLASL_FULL_JSON_PATH = os.path.join(
    ROOT, "datasets", "raw", "WLASL", "WLASL_v0_3.json"
)

# !!! CHỈNH LẠI cho đúng thư mục đang chứa video hiện có của bạn !!!
VIDEO_DIR = os.path.join(
    ROOT, "datasets", "raw", "WLASL", "videos"
)

MISSING_PATH = os.path.join(
    ROOT, "datasets", "annotations", "missing.txt"
)

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MISSING_PATH), exist_ok=True)

REQUEST_TIMEOUT = 15
NUM_RETRY = 2
RETRY_SLEEP_SEC = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def is_youtube(url: str) -> bool:
    host = urlparse(url).netloc
    return "youtube.com" in host or "youtu.be" in host


def download_direct(url: str, out_path: str) -> bool:
    """Tải trực tiếp bằng requests — dùng cho các nguồn serve file mp4 thẳng
    (signingsavvy, handspeak, aslpro, aslbricks, spreadthesign, ...)."""
    try:
        resp = requests.get(
            url, stream=True, timeout=REQUEST_TIMEOUT, headers=HEADERS
        )
        resp.raise_for_status()

        tmp_path = out_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)

        if os.path.getsize(tmp_path) == 0:
            os.remove(tmp_path)
            return False

        os.replace(tmp_path, out_path)
        return True

    except Exception as e:
        print(f"    [direct] lỗi: {e}")
        return False


_YOUTUBE_FALLBACK_CLIENTS = ["android_vr", "tv", "mweb", "web_safari"]


def download_youtube(url: str, out_path: str) -> bool:
    """Tải từ YouTube bằng yt-dlp (pip install -U yt-dlp).

    YouTube liên tục đổi cơ chế xác thực player (JS challenge), khiến client
    mặc định ("web") của yt-dlp hay bị lỗi "The page needs to be reloaded"
    theo từng đợt. Thử lần lượt vài client khác trước khi bỏ cuộc.
    """
    try:
        import yt_dlp
    except ImportError:
        print("    [youtube] chưa cài yt-dlp -> pip install yt-dlp")
        return False

    base_opts = {
        "outtmpl": out_path,
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }

    last_err = None
    for client in [None, *_YOUTUBE_FALLBACK_CLIENTS]:
        opts = dict(base_opts)
        if client is not None:
            opts["extractor_args"] = {"youtube": {"player_client": [client]}}

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return True
        except Exception as e:
            last_err = e
            continue

    print(f"    [youtube] lỗi (đã thử {1 + len(_YOUTUBE_FALLBACK_CLIENTS)} client): {last_err}")
    return False


def download_one(url: str, out_path: str) -> bool:
    for attempt in range(1, NUM_RETRY + 1):
        ok = download_youtube(url, out_path) if is_youtube(url) else download_direct(url, out_path)
        if ok:
            return True
        if attempt < NUM_RETRY:
            time.sleep(RETRY_SLEEP_SEC)
    return False


def download_missing_videos():

    with open(WLASL_FULL_JSON_PATH, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    existing = set(os.listdir(VIDEO_DIR))
    print(f"Thư mục video: {VIDEO_DIR}")
    print(f"Đã có sẵn {len(existing)} file trong thư mục này.\n")

    total = 0
    already_have = 0
    downloaded = 0
    failed_ids = []

    for gloss_entry in wlasl_data:
        gloss = gloss_entry["gloss"]

        for inst in gloss_entry["instances"]:
            total += 1
            video_id = inst["video_id"]
            filename = f"{video_id}.mp4"

            if filename in existing:
                already_have += 1
                continue

            out_path = os.path.join(VIDEO_DIR, filename)
            url = inst["url"]

            print(f"[{total}/?] {gloss:<20} id={video_id}  <- {url}")
            ok = download_one(url, out_path)

            if ok:
                downloaded += 1
                existing.add(filename)
            else:
                failed_ids.append(video_id)
                print(f"    -> THẤT BẠI: {video_id}")

    # Ghi các video tải lỗi vào missing.txt — flat.py sẽ tự động bỏ qua
    # các video_id này khi build train/val/test annotation.
    if failed_ids:
        with open(MISSING_PATH, "a", encoding="utf-8") as f:
            for vid in failed_ids:
                f.write(vid + "\n")

    print("\n" + "=" * 60)
    print(f"Tổng số video trong WLASL_v0_3.json : {total}")
    print(f"Đã có sẵn từ trước                  : {already_have}")
    print(f"Tải mới thành công                  : {downloaded}")
    print(f"Thất bại (đã ghi vào missing.txt)    : {len(failed_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    download_missing_videos()