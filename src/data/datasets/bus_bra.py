"""BUS-BRA dataset loader.

Source: Zenodo — "BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided
Detection and Diagnosis Systems" (doi:10.5281/zenodo.8231412)
~1,875 images, biopsy-confirmed, with BI-RADS labels and clinical metadata.

Expected raw layout after download:
    data/raw/bus_bra/
        BUSBRA/
            Images/             # bus_NNNN-{l,r}.png
            Masks/
            bus_data.csv        # ID, Case, Histology, Pathology, BIRADS, Device, ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROMPT = "<image>\nIs the lesion benign or malignant?1=Malignant, 0=Benign"


def _build_busi_cot_style_caption(row: pd.Series) -> str:
    """Build a BUS-CoT-style reasoning report from BUS-BRA structured fields.

    Format matches the BUS-CoT template so the same paraphrase augmentation pipeline
    and slot extractor work uniformly across both datasets.
    """
    birads = row.get("birads") or row.get("birads_category")
    pathology = str(row.get("pathology", "")).lower()
    histology = str(row.get("histology", ""))

    # Map pathology to answer label
    answer = "1" if pathology == "malignant" else "0"
    p_text = "malignant" if pathology == "malignant" else "benign"

    birads_str = str(birads) if birads is not None else "?"

    # BUS-BRA doesn't include the granular descriptors (margins, shape, orientation)
    # so we build a slightly different but compatible structured report.
    reasoning = (
        f"<reasoning>BI-RADS classification is {birads_str}. "
        f"Histopathology category: {histology if histology else 'None'}. "
        f"Thus this is a {p_text} lesion.</reasoning> <answer> {answer} </answer>"
    )
    return reasoning


def load(raw_dir: str | Path) -> list[dict[str, Any]]:
    """Load BUS-BRA dataset and return list of unified records."""
    raw_dir = Path(raw_dir)

    # the Zenodo zip extracts to a nested BUSBRA/ subdirectory
    nested = raw_dir / "BUSBRA"
    if nested.exists():
        raw_dir = nested

    image_dir = raw_dir / "Images"
    csv_path = raw_dir / "bus_data.csv"

    if not image_dir.exists() or not csv_path.exists():
        logger.warning(
            "BUS-BRA: missing Images/ or bus_data.csv in %s. Skipping.", raw_dir,
        )
        return []

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    id_col = "id" if "id" in df.columns else df.columns[0]

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        image_id = str(row[id_col]).strip()
        image_path = image_dir / f"{image_id}.png"
        if not image_path.exists():
            for ext in (".jpg", ".jpeg", ".bmp"):
                candidate = image_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            else:
                continue

        caption = _build_busi_cot_style_caption(row)
        birads = row.get("birads")
        pathology = str(row.get("pathology", "")).lower() or None

        records.append({
            "id": f"bus_bra_{image_id}",
            "source": "bus_bra",
            "image_path": str(image_path),
            "conversations": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": caption},
            ],
            "metadata": {
                "birads": int(birads) if pd.notna(birads) and str(birads).isdigit() else birads,
                "pathology": pathology,
                "histology": str(row.get("histology", "")).strip() or None,
                "side": str(row.get("side", "")).strip() or None,
                # `Case` is the patient id (1,875 images from 1,064 patients); the
                # grouped split keeps every image of a patient on one side.
                "group": f"bus_bra_{str(row.get('case', image_id)).strip()}",
            },
        })

    logger.info("BUS-BRA: loaded %d records from %s", len(records), raw_dir)
    return records
