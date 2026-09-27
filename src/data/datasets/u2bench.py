"""U2-BENCH dataset loader (evaluation only).

Source: HuggingFace - "U2-BENCH: A Multi-task Benchmark for Ultrasound VLMs"
Used exclusively for evaluation, not training.

Usage:
    from src.data.datasets.u2bench import load_eval
    samples = load_eval(split="test")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_eval(
    split: str = "test",
    cache_dir: str | None = None,
    task: str | None = None,
) -> list[dict[str, Any]]:
    """Load U2-BENCH from HuggingFace datasets.

    Args:
        split: Dataset split ("test" recommended for eval).
        cache_dir: HuggingFace cache directory.
        task: Filter to a specific task (e.g. "report_generation", "classification").
              None = return all tasks.

    Returns:
        List of records with keys: id, image, question, answer, task, metadata.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    logger.info("Loading U2-BENCH split='%s' from HuggingFace...", split)

    # Dataset ID may need updating if the repo name changes
    HF_DATASET_ID = "U2-BENCH/U2-BENCH"

    try:
        ds = load_dataset(HF_DATASET_ID, split=split, cache_dir=cache_dir, trust_remote_code=True)
    except Exception as e:
        logger.error(
            "Failed to load U2-BENCH from HuggingFace (%s). "
            "You may need to run: huggingface-cli login\n"
            "Error: %s",
            HF_DATASET_ID,
            e,
        )
        raise

    records: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        item_task = item.get("task", "unknown")
        if task is not None and item_task != task:
            continue

        records.append(
            {
                "id": item.get("id", f"u2bench_{i}"),
                "image": item.get("image"),       # PIL Image
                "question": item.get("question", item.get("prompt", "")),
                "answer": item.get("answer", item.get("label", "")),
                "task": item_task,
                "metadata": {k: v for k, v in item.items()
                             if k not in {"image", "question", "answer", "id", "task"}},
            }
        )

    logger.info("U2-BENCH: loaded %d eval samples (task=%s)", len(records), task or "all")
    return records


def available_tasks(cache_dir: str | None = None) -> list[str]:
    """Return list of task names in U2-BENCH."""
    samples = load_eval(split="test", cache_dir=cache_dir)
    return sorted({s["task"] for s in samples})
