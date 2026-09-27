"""BrEaST (Breast-Lesions-USG) loader - external, evaluation-only.

Source: TCIA, CC BY 4.0, DOI 10.7937/9WKK-Q141 (Pawłowska et al., Sci Data 2024).
256 B-mode images from 256 patients, 252 with a radiologist-annotated lesion and
the full BI-RADS descriptor lexicon (shape, margin, echogenicity, posterior
features, halo, calcifications, BI-RADS, biopsy/follow-up verification). It is
the only public breast-ultrasound set with finding-level labels, so it is the
external test of descriptor-level report content (reviewer 1 items
7 and 12). Never used for training.

Expected raw layout (direct zip + xlsx from the TCIA collection page):
    data/raw/breast/
        BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx
        BrEaST-Lesions_USG-images_and_masks/
            caseNNN.png            # full B-mode frame
            caseNNN_tumor.png      # lesion mask

Images are full frames while BUS-CoT trains on lesion crops, so `load` also
writes a mask-bounding-box crop per case to `<raw_dir>/crops/` and points
`image_path` at it; `metadata.full_image_path` keeps the original. The margin
matches the BUS-CoT crop protocol: matching BUS-Lesion crops back into their
BUS-Expert source frames gives a median context of 0.6x the lesion bounding box
on each side (IQR 0.4-1.0), so the same 0.6 is used here.

The reference report is rendered in the BUS-CoT style-A template from the
structured fields, so `scripts/evaluate_slots.py` and `src/evaluation/slots.py`
work unchanged. BrEaST has no orientation field, so that clause is omitted.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

PROMPT = "<image>\nIs the lesion benign or malignant?1=Malignant, 0=Benign"

XLSX = "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
IMAGE_DIR = "BrEaST-Lesions_USG-images_and_masks"
CROP_MARGIN = 0.60

#: BrEaST echogenicity -> the BUS-CoT style-A surface form (the exact words the
#: source reasoning texts use, so the extractor regex `This lesion has (...)with`
#: returns the same vocabulary for references and predictions).
ECHO_TO_TEXT = {
    "hypoechoic": "hypoechoic",
    "hyperechoic": "hyperechoic",
    "isoechoic": "isoechoic",
    "anechoic": "anechoic",
    "heterogeneous": "heterogeneous",
    "complex cystic/solid": "complex cystic and solid",
}


def margin_is_circumscribed(margin: str) -> bool | None:
    m = str(margin).strip().lower()
    if m.startswith("circumscribed"):
        return True
    if m.startswith("not circumscribed"):
        return False
    return None


def margin_text(margin: str) -> str | None:
    """Most severe BI-RADS margin descriptor in the BrEaST string. BUS-CoT's own
    vocabulary is circumscribed / indistinct / microlobulated / angular (it never
    says "spiculated"), so descriptor accuracy is scored on the binary
    circumscribed-vs-not axis; the word is kept for the reference text."""
    circ = margin_is_circumscribed(margin)
    if circ is None:
        return None
    if circ:
        return "circumscribed"
    m = str(margin).lower()
    for word in ("spiculated", "angular", "microlobulated", "indistinct"):
        if word in m:
            return word
    return "indistinct"


def _crop_to_mask(image: Image.Image, mask: Image.Image, margin: float) -> Image.Image:
    arr = np.array(mask.convert("L")) > 0
    if not arr.any():
        return image
    rows, cols = np.where(arr)
    h, w = arr.shape
    dy = int(margin * (rows.max() - rows.min() + 1))
    dx = int(margin * (cols.max() - cols.min() + 1))
    box = (max(0, cols.min() - dx), max(0, rows.min() - dy),
           min(w, cols.max() + 1 + dx), min(h, rows.max() + 1 + dy))
    return image.crop(box)


def _reference_report(row: pd.Series) -> str:
    """BUS-CoT style-A rendering of the structured fields (no orientation)."""
    shape = str(row["Shape"]).strip().lower()
    margin = margin_text(row["Margin"])
    echo = ECHO_TO_TEXT.get(str(row["Echogenicity"]).strip().lower())
    calc = "absence of calcific deposits" if str(row["Calcifications"]).strip().lower() == "no" \
        else "calcific deposits"
    birads = str(row["BIRADS"]).strip().upper()
    pathology = str(row["Classification"]).strip().lower()
    answer = "1" if pathology == "malignant" else "0"
    finding = f"This lesion has {margin} margins and {shape} shape, " if margin else ""
    echo_clause = f"This lesion has {echo}with {calc}. " if echo else ""
    return (
        f"<reasoning>{finding}{echo_clause}"
        f"Based on these features, the lesion is classified as BIRADS {birads}. "
        f"Thus this is a {pathology} lesion.</reasoning> <answer> {answer} </answer>"
    )


def load(raw_dir: str | Path, crop: bool = True) -> list[dict[str, Any]]:
    """Return unified records for the 252 lesion cases (the 4 normal cases have
    no lesion, no descriptors and BI-RADS 1, a category absent from training)."""
    raw_dir = Path(raw_dir)
    xlsx = raw_dir / XLSX
    image_dir = raw_dir / IMAGE_DIR
    if not xlsx.exists() or not image_dir.exists():
        logger.warning("BrEaST: expected %s and %s under %s. Skipping.", XLSX, IMAGE_DIR, raw_dir)
        return []

    df = pd.read_excel(xlsx)
    crop_dir = raw_dir / "crops"
    crop_dir.mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        if str(row["Classification"]).strip().lower() not in ("benign", "malignant"):
            continue
        full_path = image_dir / str(row["Image_filename"])
        mask_path = image_dir / str(row["Mask_tumor_filename"])
        if not full_path.exists():
            continue
        image_path = full_path
        if crop and mask_path.exists():
            image_path = crop_dir / full_path.name
            if not image_path.exists():
                img = Image.open(full_path).convert("RGB")
                _crop_to_mask(img, Image.open(mask_path), CROP_MARGIN).save(image_path)

        pathology = str(row["Classification"]).strip().lower()
        records.append({
            "id": f"breast_{int(row['CaseID']):03d}",
            "source": "breast",
            "image_path": str(image_path),
            "conversations": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": _reference_report(row)},
            ],
            "metadata": {
                "birads": str(row["BIRADS"]).strip().upper(),
                "pathology": pathology,
                "shape": str(row["Shape"]).strip().lower(),
                "margin": str(row["Margin"]).strip().lower(),
                "margin_circumscribed": margin_is_circumscribed(row["Margin"]),
                "echogenicity": str(row["Echogenicity"]).strip().lower(),
                "posterior_features": str(row["Posterior_features"]).strip().lower(),
                "calcifications": str(row["Calcifications"]).strip().lower(),
                "verification": str(row["Verification"]).strip().lower(),
                "full_image_path": str(full_path),
                "group": f"breast_{int(row['CaseID']):03d}",
            },
        })

    logger.info("BrEaST: loaded %d lesion records from %s (crop=%s)", len(records), raw_dir, crop)
    return records
