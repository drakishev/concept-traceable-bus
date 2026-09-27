"""BUSI (Breast Ultrasound Images) dataset loader.

Source: Kaggle - "Breast Ultrasound Images Dataset" (aryashah2k mirror)
~780 images split into benign/, malignant/, normal/ subdirectories.
No free-text reports - we synthesize a BUS-CoT-style structured report
from the label so the same slot extractor and augmentation pipeline work.

Expected raw layout after download + extract:
    data/raw/busi/
        Dataset_BUSI_with_GT/
            benign/      # image files + _mask.png files
            malignant/
            normal/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPT = "<image>\nIs the lesion benign or malignant?1=Malignant, 0=Benign"

# BUSI is label-only, so we generate the BUS-CoT-style structured report
# using consensus typical findings for each class. The paraphrase augmentation
# pipeline will produce additional surface variants.
CAPTION_TEMPLATES = {
    "benign": (
        "<reasoning>This lesion is parallel, has circumscribed margins and oval shape, "
        "This lesion has hypoechoicwith absence of calcifications. Based on these features, "
        "the lesion is classified as BIRADS 3. The features are consistent with "
        "histopathology category: None. Thus this is a benign lesion.</reasoning> "
        "<answer> 0 </answer>"
    ),
    "malignant": (
        "<reasoning>This lesion is not parallel, has indistinct margins and irregular shape, "
        "This lesion has hypoechoicwith absence of calcifications. Based on these features, "
        "the lesion is classified as BIRADS 4C. The features are consistent with "
        "histopathology category: None. Thus this is a malignant lesion.</reasoning> "
        "<answer> 1 </answer>"
    ),
    # 'normal' images have no lesion → keep a distinct format but valid for both encoders
    "normal": (
        "<reasoning>No discrete lesion identified. Normal breast tissue with no mass, "
        "no architectural distortion, no suspicious calcifications. "
        "Based on these features, the imaging is classified as BIRADS 1. "
        "Thus this is a benign lesion.</reasoning> <answer> 0 </answer>"
    ),
}

BIRADS_MAP = {"benign": 3, "malignant": "4C", "normal": 1}
PATH_MAP = {"benign": "benign", "malignant": "malignant", "normal": "benign"}


def load(raw_dir: str | Path) -> list[dict[str, Any]]:
    """Load BUSI dataset and return list of unified records."""
    raw_dir = Path(raw_dir)

    # the Kaggle zip extracts to Dataset_BUSI_with_GT/
    nested = raw_dir / "Dataset_BUSI_with_GT"
    if nested.exists():
        raw_dir = nested

    records: list[dict[str, Any]] = []

    for label in ("benign", "malignant", "normal"):
        label_dir = raw_dir / label
        if not label_dir.exists():
            logger.warning("BUSI: directory not found: %s", label_dir)
            continue

        image_files = sorted(
            p for p in label_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            and "_mask" not in p.stem
        )

        caption = CAPTION_TEMPLATES[label]
        birads = BIRADS_MAP[label]
        pathology = PATH_MAP[label]

        for img_path in image_files:
            records.append({
                "id": f"busi_{label}_{img_path.stem}",
                "source": "busi",
                "image_path": str(img_path),
                "conversations": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": caption},
                ],
                "metadata": {
                    "birads": birads,
                    "pathology": pathology,
                    "is_normal": label == "normal",
                },
            })

    logger.info("BUSI: loaded %d records from %s", len(records), raw_dir)
    return records
