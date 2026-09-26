"""Download U2-BENCH from HuggingFace (evaluation split only).

U2-BENCH: A Multi-task Benchmark for Ultrasound Understanding
HuggingFace: https://huggingface.co/datasets/U2-BENCH/U2-BENCH

Usage:
    python scripts/download/download_u2bench.py [--cache_dir data/raw/u2bench]
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET_ID = "DolphinAI/u2-bench"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download U2-BENCH from HuggingFace")
    parser.add_argument("--cache_dir", default="data/raw/u2bench", help="HF cache directory")
    parser.add_argument(
        "--split", default="test", help="Dataset split to download (default: test)"
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install 'datasets': pip install datasets")
        return

    logger.info("Downloading U2-BENCH split='%s' to %s...", args.split, args.cache_dir)
    logger.info("If this fails with an auth error, run: huggingface-cli login")

    try:
        ds = load_dataset(
            HF_DATASET_ID,
            split=args.split,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        logger.info("U2-BENCH downloaded: %d samples", len(ds))

        # Print task distribution
        if "task" in ds.column_names:
            from collections import Counter
            task_counts = Counter(ds["task"])
            logger.info("Task distribution:")
            for task, count in sorted(task_counts.items()):
                logger.info("  %s: %d", task, count)

    except Exception as e:
        logger.error("Download failed: %s", e)
        logger.info(
            "Manual alternative:\n"
            "  huggingface-cli login\n"
            "  huggingface-cli download %s --repo-type dataset --local-dir %s",
            HF_DATASET_ID, args.cache_dir,
        )
        raise


if __name__ == "__main__":
    main()
