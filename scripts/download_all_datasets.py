"""
Dataset Downloader for Visual Odometry Benchmarks.
Downloads and extracts the datasets specified in the 1st review presentation:
1. TUM RGB-D (freiburg1_xyz)
2. EuRoC MAV (V102 sequence with ground truth)
3. TartanAir (abandonedfactory trajectory, poses, segmentation, and flow)
4. KITTI (Sequence 10 odometry test trajectory)
"""

import os
import sys
import tarfile
import zipfile
import urllib.request
from typing import Optional


DATASETS_CONFIG = [
    {
        "name": "TUM RGB-D (freiburg1_xyz)",
        "url": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
        "output_dir": "datasets/tum_rgbd",
        "filename": "rgbd_dataset_freiburg1_xyz.tgz",
        "extract": True,
        "format": "tar"
    },
    {
        "name": "EuRoC MAV (V102)",
        "url": "https://cmu.box.com/shared/static/1ctocidptdv1xev6pjxdj0b782mrstps.zip",
        "output_dir": "datasets/euroc",
        "filename": "EuRoC_V102.zip",
        "extract": True,
        "format": "zip"
    },
    {
        "name": "TartanAir Camera Poses & Metadata",
        "url": "https://huggingface.co/datasets/KangLiao/TartanAir-Absolute-Camera/resolve/main/abandonedfactory.zip",
        "output_dir": "datasets/tartanair",
        "filename": "abandonedfactory_poses.zip",
        "extract": True,
        "format": "zip"
    },
    {
        "name": "TartanAir Segmentation Labels",
        "url": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/abandonedfactory/Easy/seg_left.zip",
        "output_dir": "datasets/tartanair",
        "filename": "abandonedfactory_seg_left.zip",
        "extract": True,
        "format": "zip"
    },
    {
        "name": "TartanAir Flow Masks",
        "url": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/abandonedfactory/Easy/flow_mask.zip",
        "output_dir": "datasets/tartanair",
        "filename": "abandonedfactory_flow_mask.zip",
        "extract": True,
        "format": "zip"
    },
    {
        "name": "KITTI Visual Odometry (Seq 10)",
        "url": "https://cmu.box.com/shared/static/nw3bi7x5vng2xy296ndxt19uozpk64jq.zip",
        "output_dir": "datasets/kitti",
        "filename": "KITTI_10.zip",
        "extract": True,
        "format": "zip"
    }
]


def download_with_progress(url: str, output_path: str, desc: str):
    print(f"\n[DOWNLOADING] {desc}")
    print(f"  URL: {url}")
    print(f"  Destination: {output_path}")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"  File already exists ({os.path.getsize(output_path) / (1024*1024):.2f} MB). Skipping download.")
        return

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded * 100.0 / total_size
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Downloaded: {mb_down:.1f} MB")
            sys.stdout.flush()

    print("\n  Download complete!")


def extract_archive(file_path: str, extract_to: str, fmt: str):
    print(f"[EXTRACTING] {file_path} -> {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    try:
        if fmt == "tar" or file_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=extract_to)
        elif fmt == "zip" or file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(path=extract_to)
        print("  Extraction complete!")
    except Exception as e:
        print(f"  Warning: extraction encountered an issue: {e}")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    for item in DATASETS_CONFIG:
        target_dir = os.path.join(root_dir, item["output_dir"])
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, item["filename"])

        download_with_progress(item["url"], file_path, item["name"])
        if item.get("extract", False):
            extract_archive(file_path, target_dir, item.get("format", "zip"))

    print("\n" + "=" * 60)
    print("All datasets downloaded and organized in 'datasets/' successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
