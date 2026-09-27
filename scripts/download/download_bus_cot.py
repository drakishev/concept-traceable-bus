"""Download BUS-CoT dataset from Figshare.

Paper: "A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering
        All Histopathology Categories"
Figshare: https://figshare.com/articles/dataset/30838715
File: BUSCoT.zip (6.9 GB), file_id=60240116
MD5: 832db89dd4095f49a1a4182a72c75be8

After extraction the zip contains:
    vlm/vlm_dataset/
        vlm_reasoning_trainval.json   <- chain-of-thought train+val
        vlm_reasoning_test.json       <- chain-of-thought test
        vlm_baseline_trainval.json    <- baseline (no CoT) train+val
        vlm_baseline_test.json        <- baseline test
    vlm/images/                       <- ultrasound images

Usage:
    python scripts/download/download_bus_cot.py [--output_dir data/raw/bus_cot]
    python scripts/download/download_bus_cot.py --zip_path /path/to/BUSCoT.zip
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGSHARE_DIRECT_URL = "https://ndownloader.figshare.com/files/60240116"
FIGSHARE_MD5 = "832db89dd4095f49a1a4182a72c75be8"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar. Resumes if partial download exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check for partial download
    headers = {}
    existing_size = dest.stat().st_size if dest.exists() else 0
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logger.info("Resuming download from byte %d", existing_size)

    response = requests.get(url, stream=True, headers=headers, timeout=60)

    if response.status_code == 416:
        logger.info("File already fully downloaded: %s", dest)
        return
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0)) + existing_size
    mode = "ab" if existing_size > 0 else "wb"

    with open(dest, mode) as f, tqdm(
        total=total, initial=existing_size, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file, skipping already-extracted files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        for member in tqdm(members, desc="Extracting"):
            target = dest_dir / member
            if not target.exists():
                z.extract(member, dest_dir)
    logger.info("Extracted to %s", dest_dir)


def verify_md5(path: Path, expected: str) -> bool:
    """Return True if file MD5 matches expected."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        logger.warning("MD5 mismatch: got %s, expected %s", actual, expected)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BUS-CoT dataset")
    parser.add_argument("--output_dir", default="data/raw/bus_cot", help="Output directory")
    parser.add_argument("--zip_path", default=None, help="Path to already-downloaded zip file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    json_dir = output_dir / "BUSCoT" / "DatasetFiles"
    if json_dir.exists() and any(json_dir.glob("*.json")):
        logger.info("BUS-CoT already extracted at %s. Skipping.", output_dir)
        return

    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
    else:
        zip_path = output_dir / "BUSCoT.zip"
        if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
            logger.info("Zip already downloaded: %s", zip_path)
        else:
            logger.info("Downloading BUS-CoT (6.9 GB) from Figshare...")
            download_file(FIGSHARE_DIRECT_URL, zip_path)
            logger.info("Verifying MD5...")
            if not verify_md5(zip_path, FIGSHARE_MD5):
                logger.error("MD5 check failed - download may be corrupt. Delete and retry.")
                return
            logger.info("MD5 OK.")

    logger.info("Extracting %s -> %s", zip_path, output_dir)
    extract_zip(zip_path, output_dir)

    # Report what we got
    for json_file in sorted((output_dir / "BUSCoT" / "DatasetFiles").glob("vlm_*.json")):
        import json
        with open(json_file) as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else "?"
        logger.info("  %s: %s samples", json_file.name, count)

    logger.info("BUS-CoT download complete. Files in: %s", output_dir)


if __name__ == "__main__":
    main()
