"""Audit train/val/test contamination at the group level and re-score without it.

Group = the finest available independence unit: for BUS-CoT the source image
(`.../trainval/001755@0.png` -> image 001755; the @N suffix is the lesion-crop
index, 4,338 crops from 4,228 images, so 1.03 crops per image), for BUS-BRA the
patient (`Case`), otherwise the image. The pre-revision image-level split
(data/augmented_v2) let crops of one image straddle train/test; the grouped split
(data/augmented_v4g) must show zero overlap. This script quantifies both and
recomputes the headline metrics with contaminated test groups removed.

Groups are read from `metadata.group` when present (v4g) and derived from the
image path otherwise (v2), where BUS-BRA patients are looked up in bus_data.csv.

Group keys only compare records of one source. BUS-CoT, however, aggregates
11 collections (all of BUS-BRA and BUSI among them), and public benchmarks reuse
the same collections, so the same frame can reach two splits, or an "external"
set, under different ids. `--hash` therefore also compares every image by a
perceptual hash (dHash, 256 bits; BUS-CoT crops are compared through their raw
source frame) across splits and against each `--external` set.

Outputs <output>:
  - per-split record and group counts, train-test / train-val group overlap,
    exact-image overlap, the contaminated test ids
  - with --hash: near-duplicate frames across splits and, per external set, the
    ids whose frame is in train/val (exclude them from external evaluation)
  - Path/Risk F1 for each run, full test set vs contaminated groups excluded

Usage:
    python scripts/leakage_audit.py                                   # v2, historical
    python scripts/leakage_audit.py --data data/augmented_v4g \\
        --output outputs/stats/leakage_v4g.json
    python scripts/leakage_audit.py --data data/unified_v5 --hash \\
        --external u2bench=data/raw/u2bench/breast_eval/breast.jsonl \\
        "breast=data/raw/breast/BrEaST-Lesions_USG-images_and_masks/case???.png" \\
        --output outputs/stats/leakage_v5.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from src.data.image_hash import frame_path, hash_paths, near_duplicates
from src.evaluation.slots import binary_f1, paired_labels

PRED = Path("outputs/weekend")

# `.../BUS-Lesion/trainval/001755@0.png` -> source image 001755 (@N = lesion-crop index)
_STUDY_RE = re.compile(r"/(\d+)@(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
_BUSBRA_RE = re.compile(r"/(bus_\d+-[lr])\.png$")
_BUSBRA_CASE: dict[str, str] | None = None


def _busbra_case(image_id: str) -> str:
    global _BUSBRA_CASE
    if _BUSBRA_CASE is None:
        import csv
        _BUSBRA_CASE = {}
        csv_path = Path("data/raw/bus_bra/BUSBRA/bus_data.csv")
        if csv_path.exists():
            with open(csv_path) as f:
                _BUSBRA_CASE = {r["ID"]: str(r["Case"]) for r in csv.DictReader(f)}
    return _BUSBRA_CASE.get(image_id, image_id)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def study_key(rec: dict) -> tuple[str, str]:
    """Group key: metadata.group if the split recorded one, else derived from the
    path (BUS-CoT source image, BUS-BRA patient, otherwise the image itself)."""
    group = (rec.get("metadata") or {}).get("group")
    if group:
        return ("group", group)
    path = rec.get("image_path") or ""
    m = _STUDY_RE.search(path)
    if m:
        return ("bus_cot", m.group(1))
    b = _BUSBRA_RE.search(path)
    if b:
        return ("bus_bra", _busbra_case(b.group(1)))
    return (rec.get("source", "?"), path)


def audit_splits(data: Path) -> dict:
    train, val, test = (_read_jsonl(data / f"{s}.jsonl") for s in ("train", "val", "test"))
    tr_s = {study_key(r) for r in train}
    va_s = {study_key(r) for r in val}
    te_s = {study_key(r) for r in test}

    leaked_studies = tr_s & te_s
    leaked_ids = sorted(r["id"] for r in test if study_key(r) in leaked_studies)

    tr_img = {r["image_path"] for r in train}
    te_img = {r["image_path"] for r in test}

    return {
        "n_records": {"train": len(train), "val": len(val), "test": len(test)},
        "n_studies": {"train": len(tr_s), "val": len(va_s), "test": len(te_s)},
        "train_test_study_overlap": len(leaked_studies),
        "train_val_study_overlap": len(tr_s & va_s),
        "exact_image_overlap_train_test": len(tr_img & te_img),
        "contaminated_test_records": len(leaked_ids),
        "contaminated_test_fraction": round(len(leaked_ids) / max(len(test), 1), 4),
        "contaminated_ids": leaked_ids,
    }


def _hash_all(items: list[tuple[str, str]]) -> tuple[list[str], np.ndarray]:
    """(id, path) pairs -> ids and their stacked hashes. An unreadable frame would
    silently leave the audit, so any missing image is an error."""
    hashes = hash_paths([p for _, p in items])
    missing = [p for (_, p), h in zip(items, hashes) if h is None]
    if missing:
        raise SystemExit(f"{len(missing)} frames unreadable, audit incomplete, e.g. {missing[:3]}")
    return [i for i, _ in items], np.stack(hashes)


def _matches(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row indices of `a` with a near-duplicate in `b`."""
    return np.unique(near_duplicates(a, b)[:, 0])


def _external_items(spec: str) -> list[tuple[str, str]]:
    """`name=path.jsonl` (id, image_path per line) or `name=glob` (id = file stem)."""
    src = spec.split("=", 1)[1]
    if src.endswith(".jsonl"):
        return [(r["id"], r["image_path"]) for r in _read_jsonl(Path(src))]
    return [(Path(p).stem, p) for p in sorted(glob.glob(src))]


def hash_audit(data: Path, externals: list[str]) -> dict:
    split = {s: _read_jsonl(data / f"{s}.jsonl") for s in ("train", "val", "test")}
    ids, hashes = {}, {}
    for s, recs in split.items():
        ids[s], hashes[s] = _hash_all([(r["id"], frame_path(r)) for r in recs])
    seen = np.concatenate([hashes["train"], hashes["val"]])
    out: dict = { "splits": {}, "external": {}}
    for a, b in (("test", "train"), ("test", "val"), ("val", "train")):
        hit = _matches(hashes[a], hashes[b])
        out["splits"][f"{a}_in_{b}"] = {"n": int(len(hit)),
                                        "ids": [ids[a][i] for i in hit]}
    for spec in externals:
        name = spec.split("=", 1)[0]
        e_ids, e_hash = _hash_all(_external_items(spec))
        hit = _matches(e_hash, seen)
        out["external"][name] = {"n": len(e_ids), "in_train_val": int(len(hit)),
                                 "contaminated_ids": [e_ids[i] for i in hit]}
    return out


def rescore(runs: list[str], excluded: set[str]) -> dict:
    """Path/Risk F1 per run, on the full test set and with `excluded` ids dropped."""
    out: dict = {}
    for name in runs:
        path = PRED / name / "predictions.json"
        if not path.exists():
            continue
        preds = json.load(open(path))
        kept = [p for p in preds if p["id"] not in excluded]
        row: dict = {"n_full": len(preds), "n_clean": len(kept),
                     "n_dropped": len(preds) - len(kept)}
        for endpoint in ("pathology", "risk"):
            row[f"{endpoint}_f1_full"] = round(binary_f1(paired_labels(preds, endpoint)), 4)
            row[f"{endpoint}_f1_clean"] = round(binary_f1(paired_labels(kept, endpoint)), 4)
            row[f"{endpoint}_delta"] = round(
                row[f"{endpoint}_f1_clean"] - row[f"{endpoint}_f1_full"], 4)
        out[name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Run names under outputs/weekend/ (default: all with predictions).")
    ap.add_argument("--output", default="outputs/stats/leakage.json")
    ap.add_argument("--data", default="data/augmented_v2",
                    help="split directory with train/val/test.jsonl")
    ap.add_argument("--hash", action="store_true",
                    help="Also compare frames by perceptual hash across splits and sources.")
    ap.add_argument("--external", nargs="*", default=[], metavar="NAME=JSONL_OR_GLOB",
                    help="External evaluation sets to check against train/val (with --hash).")
    args = ap.parse_args()

    audit = audit_splits(Path(args.data))
    audit["data"] = args.data
    if args.hash:
        audit["hash"] = hash_audit(Path(args.data), args.external)
    runs = args.runs or sorted(p.parent.name for p in PRED.glob("*/predictions.json"))
    excluded = set(audit["contaminated_ids"])
    if args.hash:  # near-duplicate frames leak as much as a shared patient
        for key in ("test_in_train", "test_in_val"):
            excluded |= set(audit["hash"]["splits"][key]["ids"])
    audit["rescored_excluded_ids"] = len(excluded)
    audit["rescored"] = rescore(runs, excluded)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2))

    print(f"=== Group-level split audit: {args.data} ===")
    print(f"  records  train/val/test : {audit['n_records']}")
    print(f"  studies  train/val/test : {audit['n_studies']}")
    print(f"  exact image overlap     : {audit['exact_image_overlap_train_test']}")
    print(f"  train-test study overlap: {audit['train_test_study_overlap']} studies")
    print(f"  contaminated test recs  : {audit['contaminated_test_records']}"
          f" ({audit['contaminated_test_fraction']:.1%})")
    if args.hash:
        h = audit["hash"]
        print("=== Near-duplicate frames (perceptual hash) ===")
        for k, v in h["splits"].items():
            print(f"  {k:22s}: {v['n']}")
        for k, v in h["external"].items():
            print(f"  external {k:13s}: {v['in_train_val']} of {v['n']} frames in train/val")
    print("\n=== Sensitivity: F1 full vs contaminated studies excluded ===")
    for name, r in audit["rescored"].items():
        print(f"  {name:22s} Path {r['pathology_f1_full']:.3f} -> {r['pathology_f1_clean']:.3f}"
              f" ({r['pathology_delta']:+.3f})   "
              f"Risk {r['risk_f1_full']:.3f} -> {r['risk_f1_clean']:.3f}"
              f" ({r['risk_delta']:+.3f})   [-{r['n_dropped']} recs]")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
