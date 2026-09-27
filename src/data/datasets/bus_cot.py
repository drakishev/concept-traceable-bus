"""BUS-CoT dataset loader.

Paper: "A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering
        All Histopathology Categories"
Figshare: https://figshare.com/articles/dataset/30838715

Extracted layout (zip root = BUSCoT/):
    data/raw/bus_cot/BUSCoT/
        BUS-Lesion/
            trainval/   *.png     (4,338 images)
            test/       *.png     (873 images)
        DatasetFiles/
            vlm_reasoning_trainval.json   <- CoT train+val (preferred)
            vlm_reasoning_test.json       <- CoT test
            vlm_baseline_trainval.json    <- baseline train+val
            vlm_baseline_test.json        <- baseline test

JSON record format:
    {
        "query":    "Is the lesion benign or malignant?...<image>",
        "images":   ["BUS-Lesion/trainval/000000@0.png"],   # relative to BUSCoT/
        "response": "<reasoning>...</reasoning> <answer> 0 </answer>"
    }
Answer: 0=Benign, 1=Malignant
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE)
# ".../BUS-Lesion/trainval/001755@0.png" -> source image 001755 (the @N suffix is the
# lesion-crop index).
_SOURCE_IMAGE_RE = re.compile(r"/(\d+)@\d+\.png$")


def source_image_id(image_path: str) -> str | None:
    m = _SOURCE_IMAGE_RE.search(image_path)
    return m.group(1) if m else None


def source_image_index(bus_cot_root: Path) -> dict[str, dict[str, str]]:
    """Source image id -> {patient_id, dataset} from BUS-Expert_dataset.json.

    BUS-CoT aggregates 11 source collections (BUS-BRA and BUSI among them) and
    records a patient identifier per image. Patients carry up to 92 images, so
    the patient, not the source image, is the unit a leakage-free split groups by.
    """
    path = bus_cot_root / "DatasetFiles" / "BUS-Expert_dataset.json"
    if not path.exists():
        return {}
    with open(path) as f:
        expert = json.load(f)
    return {
        v["image_file"]["raw_image"].split("/")[0]: {
            "patient_id": v["patient_id"], "dataset": v["dataset"]}
        for v in expert.values()
    }


def _normalize_query(query: str) -> str:
    """Move the <image> token to the start of the prompt (standard convention)."""
    query = query.replace("<image>", "").strip()
    return f"<image>\n{query}"


def _parse_pathology(response: str) -> str | None:
    """Extract benign/malignant label from the answer tag."""
    m = _ANSWER_RE.search(response)
    if m:
        return "malignant" if m.group(1) == "1" else "benign"
    return None


def load(
    raw_dir: str | Path,
    variant: str = "reasoning",   # "reasoning" (CoT) or "baseline"
    splits: list[str] | None = None,  # ["trainval"] | ["test"] | both
) -> list[dict[str, Any]]:
    """Load BUS-CoT and return list of unified records.

    Args:
        raw_dir:  Path to data/raw/bus_cot  (contains the BUSCoT/ subdir).
        variant:  "reasoning" (chain-of-thought, preferred) or "baseline".
        splits:   Which splits to load. Defaults to ["trainval"] for training.

    Each record: {id, source, image_path, conversations, metadata}
    """
    raw_dir = Path(raw_dir)
    bus_cot_root = raw_dir / "BUSCoT"

    if not bus_cot_root.exists():
        logger.warning(
            "BUS-CoT: BUSCoT/ not found in %s. "
            "Run: python scripts/download/download_bus_cot.py",
            raw_dir,
        )
        return []

    json_dir = bus_cot_root / "DatasetFiles"
    if not json_dir.exists():
        logger.warning("BUS-CoT: DatasetFiles/ not found under %s", bus_cot_root)
        return []

    if splits is None:
        splits = ["trainval"]

    index = source_image_index(bus_cot_root)
    records: list[dict[str, Any]] = []

    for split in splits:
        json_path = json_dir / f"vlm_{variant}_{split}.json"
        if not json_path.exists():
            logger.warning("BUS-CoT: JSON not found: %s", json_path)
            continue

        with open(json_path) as f:
            data = json.load(f)

        logger.info("BUS-CoT: loading %d samples from %s", len(data), json_path.name)

        for i, item in enumerate(data):
            # Resolve image path (relative to BUSCoT/)
            rel_images = item.get("images", [])
            if not rel_images:
                logger.debug("item %d has no images, skipping", i)
                continue

            image_path = bus_cot_root / rel_images[0]
            if not image_path.exists():
                logger.debug("image not found: %s", image_path)
                continue

            query = item.get("query", "")
            response = item.get("response", "")
            if not query or not response:
                continue

            src_image = source_image_id(str(image_path))
            origin = index.get(src_image or "", {})
            records.append(
                {
                    "id": f"bus_cot_{variant}_{split}_{i:05d}",
                    "source": "bus_cot",
                    "image_path": str(image_path),
                    "conversations": [
                        {"role": "user",      "content": _normalize_query(query)},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {
                        "birads": None,
                        "pathology": _parse_pathology(response),
                        "variant": variant,
                        "split": split,
                        "official_split": split,
                        "source_image": src_image,
                        "origin_dataset": origin.get("dataset"),
                        "group": f"bus_cot_{origin.get('patient_id') or src_image}",
                    },
                }
            )

    logger.info("BUS-CoT: loaded %d total records from %s", len(records), raw_dir)
    return records
