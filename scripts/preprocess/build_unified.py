"""Build unified dataset from all raw sources.

Merges BUS-CoT, BrEaST, BUS-BRA, BUSI into train/val/test JSONL files.
Idempotent: skips sources whose raw directories don't exist.

Usage:
    python scripts/preprocess/build_unified.py [--raw_root data/raw] [--out_dir data/unified]
    python scripts/preprocess/build_unified.py --sources bus_cot busi   # only these sources
    python scripts/preprocess/build_unified.py --copy_images            # copy images to out_dir
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified import build

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified breast ultrasound dataset")
    parser.add_argument("--raw_root", default="data/raw", help="Root dir of raw datasets")
    parser.add_argument("--out_dir", default="data/unified", help="Output directory for JSONL")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=["bus_cot", "breast", "bus_bra", "busi"],
        help="Sources to include (default: all available)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images into out_dir/images/ (uses more disk space)",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Leakage-free protocol: BUS-CoT official test split + group-disjoint "
             "train/val (patient for BUS-CoT and BUS-BRA)",
    )
    parser.add_argument(
        "--drop_histopathology",
        action="store_true",
        help="Remove the histopathology sentence from every report target",
    )
    args = parser.parse_args()

    logger.info("Building unified dataset...")
    logger.info("  Raw root: %s", args.raw_root)
    logger.info("  Output:   %s", args.out_dir)
    logger.info("  Sources:  %s", args.sources or "all")

    counts = build(
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        sources=args.sources,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_images=args.copy_images,
        grouped=args.grouped,
        drop_histopathology=args.drop_histopathology,
    )

    total = sum(counts.values())
    logger.info("Done. Total records: %d", total)
    for split, n in counts.items():
        logger.info("  %s: %d", split, n)

    if total == 0:
        logger.warning(
            "No records were written. Make sure raw data directories exist under %s",
            args.raw_root,
        )
        logger.warning("Download datasets first:")
        logger.warning("  python scripts/download/download_bus_cot.py")
        logger.warning("  python scripts/download/download_bus_bra.py")
        logger.warning("  python scripts/download/download_busi.py")
        logger.warning("  python scripts/download/download_breast.py --auto")


if __name__ == "__main__":
    main()
