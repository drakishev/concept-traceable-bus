"""Prepare BrEaST (Breast-Lesions-USG) for external evaluation.

TCIA distributes BrEaST as a PNG zip and a clinical XLSX behind a click-through
Data Usage Agreement, so the two files are downloaded by hand:

    1. Open https://doi.org/10.7937/9WKK-Q141 (TCIA collection BrEaST-Lesions-USG, CC BY 4.0)
       and accept the Data Usage Agreement.
    2. Download BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip and
       BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx into data/raw/breast/.
    3. Unzip the images into the same folder:
           unzip BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip -d data/raw/breast/

This script then checks the layout and writes the evaluation file
data/raw/breast/breast_eval.jsonl (252 lesion cases, mask-bounding-box crops in
data/raw/breast/crops/, see src/data/datasets/breast.py).

Usage:
    python scripts/download/download_breast.py [--raw_dir data/raw/breast]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.datasets.breast import IMAGE_DIR, XLSX, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default="data/raw/breast")
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    missing = [p for p in (raw / XLSX, raw / IMAGE_DIR) if not p.exists()]
    if missing:
        sys.exit(f"Missing {', '.join(map(str, missing))}.\n\n{__doc__}")
    records = load(raw)
    out = raw / "breast_eval.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"{len(records)} BrEaST lesion records -> {out}")


if __name__ == "__main__":
    main()
