"""Perceptual hashing to find one frame reaching several records.

Public breast-ultrasound collections are re-published inside each other: BUS-CoT
aggregates 11 collections (all of BUS-BRA and BUSI among them), and some frames
appear twice in it under different patient ids. Record ids, file paths and
patient ids cannot see such copies; a perceptual hash of the pixels can.
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.datasets.bus_cot import source_image_id

#: Max Hamming distance (of 256 bits) for two frames to count as the same image.
#: Re-encoded or re-cropped copies of one frame land at 0-10.
MAX_DIST = 10
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def dhash(path: str) -> np.ndarray | None:
    """256-bit difference hash, packed to 32 bytes; None if unreadable."""
    try:
        img = Image.open(path).convert("L").resize((17, 16), Image.BILINEAR)
    except OSError:
        return None
    a = np.asarray(img, dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1])


def frame_path(record: dict[str, Any]) -> str:
    """The full frame behind a record: BUS-CoT lesion crops map to their raw source
    image, so a crop and the same frame from another collection compare equal."""
    # Older splits carry no `source_image`; the crop's file name encodes it.
    src = ((record.get("metadata") or {}).get("source_image")
           or source_image_id(record.get("image_path") or ""))
    if record.get("source") == "bus_cot" and src:
        # crops are <BUSCoT>/BUS-Lesion/<split>/<file>, frames <BUSCoT>/BUS-Expert/<src>/,
        # so the frame is found under whatever raw root the split was built from
        bus_cot = Path(record["image_path"]).parents[2]
        return str(bus_cot / "BUS-Expert" / src / f"{src}@raw.png")
    return record["image_path"]


def hash_paths(paths: list[str]) -> list[np.ndarray | None]:
    with Pool(32) as pool:
        return pool.map(dhash, paths, chunksize=64)


def near_duplicates(a: np.ndarray, b: np.ndarray, max_dist: int = MAX_DIST) -> np.ndarray:
    """(i, j) pairs with Hamming(a[i], b[j]) <= max_dist, for packed hash arrays."""
    pairs = []
    for start in range(0, len(a), 256):
        dist = _POPCOUNT[a[start:start + 256, None, :] ^ b[None, :, :]].sum(-1)
        i, j = np.nonzero(dist <= max_dist)
        pairs.append(np.stack([i + start, j], axis=1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=int)
