"""Unified dataset builder.

Merges all source datasets into a single list of records and writes
train/val/test JSONL files. Run via scripts/preprocess/build_unified.py.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data import image_hash
from src.data.datasets import breast, bus_bra, bus_cot, busi

logger = logging.getLogger(__name__)

LOADERS = {
    "bus_cot": bus_cot.load,
    "breast": breast.load,
    "bus_bra": bus_bra.load,
    "busi": busi.load,
}


def build(
    raw_root: str | Path,
    out_dir: str | Path,
    sources: list[str] | None = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    copy_images: bool = False,
    grouped: bool = False,
    drop_histopathology: bool = False,
) -> dict[str, int]:
    """Build unified dataset.

    Args:
        raw_root: Root of raw per-dataset directories (e.g. data/raw/).
        out_dir:  Output directory for unified JSONL + images (e.g. data/unified/).
        sources:  Which sources to include. None = all available.
        train_ratio: Fraction for training split.
        val_ratio:   Fraction for validation split (remainder goes to test).
        seed:     Random seed for splitting.
        copy_images: If True, copy images into out_dir/images/. If False, keep original paths.
        grouped:  Leakage-free protocol: BUS-CoT's official test split is the test
                  set, every other record is split by `metadata.group` (patient
                  for BUS-CoT and BUS-BRA) so no group straddles two splits.
        drop_histopathology: Remove the histopathology sentence from every report
                  target (it cannot be determined from sonographic appearance).

    Returns:
        Dict with split sizes: {"train": N, "val": N, "test": N}.
    """
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        (out_dir / "images").mkdir(exist_ok=True)

    active_sources = sources or list(LOADERS.keys())

    all_records: list[dict[str, Any]] = []
    for source in active_sources:
        loader = LOADERS.get(source)
        if loader is None:
            logger.warning("Unknown source '%s', skipping.", source)
            continue
        source_dir = raw_root / source
        if not source_dir.exists():
            logger.warning("Raw directory not found: %s — skipping %s.", source_dir, source)
            continue
        # The grouped protocol evaluates on BUS-CoT's own held-out split, which the
        # default loader call never reads.
        kwargs = {"splits": ["trainval", "test"]} if (grouped and source == "bus_cot") else {}
        records = loader(source_dir, **kwargs)
        all_records.extend(records)

    if not all_records:
        logger.error("No records loaded. Check raw data directories under %s.", raw_root)
        return {"train": 0, "val": 0, "test": 0}

    logger.info("Total records before split: %d", len(all_records))

    if drop_histopathology:
        for record in all_records:
            for turn in record["conversations"]:
                if turn["role"] == "assistant":
                    turn["content"] = strip_histopathology(turn["content"])

    if grouped:
        all_records = _merge_duplicate_frames(all_records)
        splits = _grouped_split(all_records, train_ratio, val_ratio, seed)
    else:
        splits = _random_split(all_records, train_ratio, val_ratio, seed)

    counts: dict[str, int] = {}
    for split_name, idx_arr in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for i in idx_arr:
                record = all_records[i]
                record["metadata"]["split"] = split_name

                if copy_images:
                    src = Path(record["image_path"])
                    dst = out_dir / "images" / f"{record['id']}{src.suffix}"
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    record["image_path"] = str(dst)

                f.write(json.dumps(record) + "\n")

        counts[split_name] = len(idx_arr)
        logger.info("  %s: %d records → %s", split_name, len(idx_arr), out_path)

    if grouped:
        _write_per_source_test_files(all_records, splits["test"], out_dir)
        _write_split_manifest(all_records, splits, out_dir)

    return counts


_HISTOPATH_RE = re.compile(
    r"\s*(?:The features are consistent with )?histopathology category:\s*[^.]+\.\s*",
    re.IGNORECASE,
)


def strip_histopathology(text: str) -> str:
    """Remove the histopathology sentence from a BUS-CoT / BUS-BRA style report."""
    return _HISTOPATH_RE.sub(" ", text).replace("  ", " ").strip()


def _random_split(
    all_records: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int,
) -> dict[str, np.ndarray]:
    """Record-level stratified split (the pre-revision protocol)."""
    # Stratify split by pathology label where available
    labels = [r["metadata"].get("pathology") or "unknown" for r in all_records]
    indices = np.arange(len(all_records))

    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=1 - train_ratio, random_state=seed, stratify=labels
        )
        temp_labels = [labels[i] for i in temp_idx]
        val_frac = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=1 - val_frac, random_state=seed, stratify=temp_labels
        )
    except ValueError:
        # Fall back to unstratified if classes too small
        logger.warning("Stratified split failed (class too small), using random split.")
        train_idx, temp_idx = train_test_split(
            indices, test_size=1 - train_ratio, random_state=seed
        )
        val_frac = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_frac, random_state=seed)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _grouped_holdout(
    idx: np.ndarray, labels: list[str], groups: list[str], n_folds: int, seed: int,
) -> list[np.ndarray]:
    """Split `idx` into `n_folds` group-disjoint, label-stratified folds."""
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = [labels[i] for i in idx]
    g = [groups[i] for i in idx]
    return [idx[test_pos] for _, test_pos in skf.split(np.zeros(len(idx)), y, g)]


def _merge_duplicate_frames(all_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make `metadata.group` cover copies of one frame, not just one patient.

    Groups are joined when any of their frames are near-duplicates (perceptual
    hash). A joined group that reaches the fixed official test split cannot be
    moved there, so its non-test records are dropped instead.
    """
    paths = [image_hash.frame_path(r) for r in all_records]
    hashes = image_hash.hash_paths(paths)
    missing = [p for p, h in zip(paths, hashes) if h is None]
    if missing:  # an unhashed frame could not be joined to its duplicates
        raise FileNotFoundError(f"{len(missing)} frames unreadable, e.g. {missing[:3]}")
    parent = list(range(len(all_records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_of_group: dict[str, int] = {}
    for i, r in enumerate(all_records):
        g = r["metadata"].get("group") or r["id"]
        parent[find(i)] = find(first_of_group.setdefault(g, i))
    stacked = np.stack(hashes)
    for a, b in image_hash.near_duplicates(stacked, stacked):
        parent[find(a)] = find(b)

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(all_records)):
        members[find(i)].append(i)
    keep: list[dict[str, Any]] = []
    dropped = merged = 0
    for idx in members.values():
        names = sorted({all_records[i]["metadata"].get("group") or all_records[i]["id"]
                        for i in idx})
        merged += len(names) > 1
        has_test = any(all_records[i]["metadata"].get("official_split") == "test" for i in idx)
        for i in idx:
            r = all_records[i]
            if has_test and r["metadata"].get("official_split") != "test":
                dropped += 1
                continue
            r["metadata"]["group"] = names[0]
            keep.append(r)
    logger.info("Duplicate-frame grouping: %d groups joined across patients; %d train/val "
                "records dropped for sharing a frame with the official test split",
                merged, dropped)
    return keep


def _grouped_split(
    all_records: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int,
) -> dict[str, np.ndarray]:
    """Group-disjoint split. BUS-CoT keeps the dataset authors' test split; its
    trainval records are divided train/val by patient (`metadata.group`). Every other
    source is divided train/val/test by its own group key (BUS-BRA: patient)."""
    labels = [r["metadata"].get("pathology") or "unknown" for r in all_records]
    groups = [r["metadata"].get("group") or r["id"] for r in all_records]

    train, val, test = [], [], []
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(all_records):
        if r["metadata"].get("official_split") == "test":
            test.append(i)
        else:
            by_source[r["source"]].append(i)

    # One fold per 10% is exact for the 0.8/0.1/0.1 default and close otherwise.
    n_folds = max(2, round(1 / val_ratio))
    for source, idx in sorted(by_source.items()):
        folds = _grouped_holdout(np.array(idx), labels, groups, n_folds, seed)
        if source == "bus_cot":
            val.extend(folds[0])
            train.extend(np.concatenate(folds[1:]))
        else:
            n_test = max(1, round(n_folds * (1 - train_ratio - val_ratio)))
            test.extend(np.concatenate(folds[:n_test]))
            val.extend(folds[n_test])
            train.extend(np.concatenate(folds[n_test + 1:]))

    splits = {"train": np.array(sorted(train)), "val": np.array(sorted(val)),
              "test": np.array(sorted(test))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = {groups[i] for i in splits[a]} & {groups[i] for i in splits[b]}
        if overlap:
            raise RuntimeError(f"grouped split leaked {len(overlap)} groups between {a}/{b}")
    return splits


def _write_per_source_test_files(
    all_records: list[dict[str, Any]], test_idx: np.ndarray, out_dir: Path,
) -> None:
    """`test_<source>_only.jsonl` per source: the BUS-CoT one is the headline set."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in test_idx:
        by_source[all_records[i]["source"]].append(all_records[i])
    for source, recs in by_source.items():
        path = out_dir / f"test_{source.replace('_', '')}_only.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        logger.info("  test/%s: %d records → %s", source, len(recs), path)


def _write_split_manifest(
    all_records: list[dict[str, Any]], splits: dict[str, np.ndarray], out_dir: Path,
) -> None:
    """Record id, group and split for every record, for release with the paper."""
    manifest = [
        {"id": all_records[i]["id"], "source": all_records[i]["source"],
         "group": all_records[i]["metadata"].get("group"), "split": name}
        for name, idx in splits.items() for i in idx
    ]
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a unified JSONL file into a list of records."""
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
