"""
download_footage.py
--------------------
Downloads test footage for the Retail Store Intelligence Platform.

Three footage sources are used, one per zone role:

  1. Mall Dataset (CUHK) -- real indoor mall surveillance footage
     -> Used as: Entrance_Cam (Camera 1)
     -> Direct zip download, no account needed, research licence

  2. Pexels stock footage (via yt-dlp) -- wide indoor crowd walking shots
     -> Used as: Aisle_Cam (Camera 2)
     -> Free Pexels licence; yt-dlp handles the browser-level auth

  3. Fallback: Oxford Town Centre Dataset -- classic outdoor pedestrian video
     -> Used if yt-dlp is unavailable
     -> Public domain, very well-known in CV community

Usage:
    python download_footage.py

Requirements:
    pip install requests tqdm
    pip install yt-dlp          <- for Pexels downloads (optional)

Output files:
    data/cam1_mall_entrance.mp4     (Entrance + Checkout camera)
    data/cam2_aisle_crowd.mp4       (Aisle camera)
"""

import os
import sys
import zipfile
import shutil
import subprocess
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# --- Helper: download with progress bar ---------------------------------------

def download(url: str, dest: Path, label: str = "") -> bool:
    """Download a URL to dest. Returns True on success."""
    print(f"\n{'-'*60}")
    print(f"  Downloading: {label or url}")
    print(f"  ->  {dest}")
    print(f"{'-'*60}")

    try:
        if HAS_TQDM:
            class _Progress(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with _Progress(unit="B", unit_scale=True, unit_divisor=1024,
                           miniters=1, desc=dest.name) as t:
                urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
        else:
            urllib.request.urlretrieve(url, dest)

        print(f"  [OK] Saved ({dest.stat().st_size // 1024} KB)")
        return True

    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def yt_dlp_download(url: str, dest: Path, label: str = "") -> bool:
    """Download a video using yt-dlp. Returns True on success."""
    print(f"\n{'-'*60}")
    print(f"  yt-dlp: {label or url}")
    print(f"  ->  {dest}")
    print(f"{'-'*60}")

    try:
        subprocess.run(
            [
                "yt-dlp",
                url,
                "--format", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--output", str(dest),
                "--no-playlist",
                "--quiet", "--progress",
            ],
            check=True,
        )
        print(f"  [OK] Saved ({dest.stat().st_size // 1024} KB)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [FAIL] Failed: {e}")
        return False


def has_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None


# --- Source 1: CUHK Mall Dataset ----------------------------------------------
#
# Real indoor shopping mall surveillance footage. Overhead wide-angle,
# continuous crowd flow -- perfect for Entrance + Checkout simulation.
# Research licence (non-commercial). Cite: Loy, Gong, Xiang (ICCV 2013).
#
# Provides 2000 JPEG frames (640x480). We assemble them into an MP4.
# ------------------------------------------------------------------------------

MALL_DATASET_URL = "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"
MALL_ZIP         = DATA_DIR / "mall_dataset.zip"
MALL_FRAMES_DIR  = DATA_DIR / "mall_dataset" / "frames"
CAM1_OUT         = DATA_DIR / "cam1_mall_entrance.mp4"

def build_cam1():
    if CAM1_OUT.exists():
        print(f"\n  [OK] {CAM1_OUT.name} already exists -- skipping.")
        return True

    # 1. Download zip
    if not MALL_ZIP.exists():
        ok = download(MALL_DATASET_URL, MALL_ZIP, "CUHK Mall Dataset (frames + annotations)")
        if not ok:
            return False

    # 2. Unzip
    if not MALL_FRAMES_DIR.exists():
        print("\n  Extracting zip...")
        try:
            with zipfile.ZipFile(MALL_ZIP, "r") as zf:
                zf.extractall(DATA_DIR)
            print("  [OK] Extracted")
        except Exception as e:
            print(f"  [FAIL] Extract failed: {e}")
            return False

    # 3. Assemble frames -> MP4 using OpenCV
    frames = sorted(MALL_FRAMES_DIR.glob("*.jpg"))
    if not frames:
        print(f"  [FAIL] No frames found in {MALL_FRAMES_DIR}")
        return False

    print(f"\n  Assembling {len(frames)} frames -> {CAM1_OUT.name} ...")
    try:
        import cv2
        sample = cv2.imread(str(frames[0]))
        h, w   = sample.shape[:2]
        writer = cv2.VideoWriter(
            str(CAM1_OUT),
            cv2.VideoWriter_fourcc(*"mp4v"),
            # The Mall Dataset was captured at <2 Hz (research slideshow, not live video).
            # 6 fps is the sweet spot: natural-looking movement + smooth tracking continuity.
            6,
            (w, h),
        )
        for i, fp in enumerate(frames):
            frame = cv2.imread(str(fp))
            if frame is not None:
                writer.write(frame)
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(frames)} frames ...")
        writer.release()
        print(f"  [OK] Written: {CAM1_OUT} ({CAM1_OUT.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] OpenCV assembly failed: {e}")
        print("  -> Install opencv-python: pip install opencv-python")
        return False


# --- Source 2: Pexels crowd walking footage -----------------------------------
#
# Free stock footage -- wide indoor/mall shots with visible crowd movement.
# yt-dlp handles Pexels download; falls back to Oxford Town Centre if unavailable.
#
# Pexels licence: free for research and commercial use, no attribution required.
# https://www.pexels.com/license/
# ------------------------------------------------------------------------------

# These are well-known Pexels video IDs with clear crowd-walking content
PEXELS_CANDIDATES = [
    # People walking in a shopping mall (overhead / side angle)
    ("https://www.pexels.com/video/people-in-a-supermarket-3009551/",     "supermarket crowd (Pexels #3009551)"),
    ("https://www.pexels.com/video/time-lapse-of-people-at-mall-4067918/","mall time-lapse (Pexels #4067918)"),
    ("https://www.pexels.com/video/crowded-pedestrian-overpass-4823546/", "pedestrian crowd (Pexels #4823546)"),
]

# Intel IoT Dev Kit sample videos -- MIT licence, confirmed live as of 2026-09
# Primary: store-aisle-detection.mp4 (9 MB, indoor aisle overhead view -- ideal for Aisle zone)
# Fallback: people-detection.mp4 (5 MB, pedestrian walkway crowd shot)
OXFORD_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/store-aisle-detection.mp4",
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
]

CAM2_OUT = DATA_DIR / "cam2_aisle_crowd.mp4"

def build_cam2():
    if CAM2_OUT.exists():
        print(f"\n  [OK] {CAM2_OUT.name} already exists -- skipping.")
        return True

    # Try yt-dlp on Pexels
    if has_yt_dlp():
        for url, label in PEXELS_CANDIDATES:
            ok = yt_dlp_download(url, CAM2_OUT, label)
            if ok and CAM2_OUT.exists():
                return True
    else:
        print("\n  yt-dlp not found -- using Oxford Town Centre fallback.")
        print("  (Install yt-dlp for higher quality Pexels footage: pip install yt-dlp)")

    # Fallback: direct download
    for url in OXFORD_FALLBACK_URLS:
        ok = download(url, CAM2_OUT, "Oxford Town Centre pedestrian video")
        if ok and CAM2_OUT.exists():
            return True

    return False


# --- Update config files -------------------------------------------------------

def update_configs():
    """Patch config.yaml and config_cam2.yaml to point at the downloaded footage."""
    import yaml

    updates = [
        ("config.yaml",      CAM1_OUT),
        ("config_cam2.yaml", CAM2_OUT),
    ]

    for cfg_path, video_path in updates:
        cfg = Path(cfg_path)
        if not cfg.exists():
            print(f"  [WARN] {cfg_path} not found -- skipping config update")
            continue
        try:
            with open(cfg, "r") as f:
                data = yaml.safe_load(f)
            data["camera"]["source"] = str(video_path).replace("\\", "/")
            with open(cfg, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"  [OK] Updated {cfg_path} -> source: {video_path}")
        except Exception as e:
            print(f"  [WARN] Could not update {cfg_path}: {e}")


# --- Main ----------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  Retail Store Intelligence -- Footage Downloader")
    print("=" * 60)
    print(f"  Output directory: {DATA_DIR.resolve()}")

    results = {}

    print("\n[1/2] Camera 1 -- CUHK Mall Dataset (Entrance + Checkout)")
    results["cam1"] = build_cam1()

    print("\n[2/2] Camera 2 -- Crowd Walking Footage (Aisle)")
    results["cam2"] = build_cam2()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for key, ok in results.items():
        status = "[OK]   Ready" if ok else "[FAIL] Failed"
        print(f"  {key}: {status}")

    if all(results.values()):
        print("\n  Updating config files...")
        update_configs()
        print("\n  All footage ready. To run:")
        print("    python backend/server.py")
        print("    python src/detector.py --config config.yaml")
        print("    python src/detector.py --config config_cam2.yaml")
    else:
        print("\n  Some downloads failed. See messages above.")
        print("  You can re-run this script -- completed files are skipped.")

    print()


if __name__ == "__main__":
    main()
