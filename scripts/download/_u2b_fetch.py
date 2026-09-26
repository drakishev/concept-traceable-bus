"""Fetch U2-BENCH and dump the breast subset as an eval JSONL.

Streams DolphinAI/u2-bench (train split) row-by-row and writes the breast-anatomy
records (image bytes + labels + report) incrementally to
data/raw/u2bench/breast_eval/breast.jsonl. Streaming (rather than a bulk
load_dataset) is used because the full dataset is 7241 inline-base64 studies and
bulk downloads have repeatedly stalled on this shared node; with streaming and
incremental writes, any breast rows found before a stall are preserved.

Re-runs resume: existing breast.jsonl ids are skipped, so re-running after a
stall picks up where it left off (modulo HF streaming not supporting exact
seeking; duplicates are de-duped by id at load time).
"""
from __future__ import annotations

import base64
import io
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image

OUT = Path("data/raw/u2bench/breast_eval")
OUT.mkdir(parents=True, exist_ok=True)


def _img(v):
    """Decode a possibly-base64 / bytes image field to a PIL image, else None."""
    try:
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(v)))
        if isinstance(v, str) and len(v) > 100:
            return Image.open(io.BytesIO(base64.b64decode(v)))
    except Exception:
        return None
    return None


def _load_existing_ids(path: Path) -> set[str]:
    """ids already in breast.jsonl so a resumed run does not re-save them."""
    ids: set[str] = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def main() -> None:
    out_jsonl = OUT / "breast.jsonl"
    seen = _load_existing_ids(out_jsonl)
    print(f"Streaming U2-BENCH (DolphinAI/u2-bench, split=train); "
          f"{len(seen)} breast rows already on disk", flush=True)

    ds = load_dataset(
        "DolphinAI/u2-bench", split="train",
        cache_dir="data/raw/u2bench", streaming=True,
    )

    n_scanned = 0
    n_breast = 0
    labels: Counter = Counter()
    tasks: Counter = Counter()
    # Append mode so a partial run is preserved across stalls/restarts.
    f = open(out_jsonl, "a")
    try:
        for i, r in enumerate(ds):
            n_scanned = i
            if i == 0:
                continue  # schema-description placeholder row
            anat = str(r.get("anatomy_location", "")).lower()
            ds_name = str(r.get("dataset_name", "")).lower()
            if "breast" not in anat and "breast" not in ds_name and "bus" not in ds_name:
                continue
            rid = f"u2b_{i}"
            if rid in seen:
                continue
            img = _img(r.get("img_data"))
            if img is None:
                continue
            pid = str(r.get("patient_id", f"u2b_{i}"))
            img_path = OUT / f"{pid}_{i}.png"
            if not img_path.exists():
                img.convert("RGB").save(img_path)
            task = str(r.get("classification_task", ""))
            label = str(r.get("class_label", ""))
            f.write(json.dumps({
                "id": rid,
                "image_path": str(img_path),
                "anatomy": anat,
                "dataset_name": ds_name,
                "classification_task": task,
                "class_label": label,
                "options": str(r.get("options", "")),
                "report": str(r.get("report", "")),
                "caption": str(r.get("caption", "")),
                "prompt": str(r.get("prompt", "")),
            }) + "\n")
            f.flush()
            n_breast += 1
            labels[label] += 1
            tasks[task] += 1
            if n_breast % 25 == 0:
                print(f"  scanned={i} breast={n_breast} "
                      f"labels={dict(labels)} tasks={dict(tasks)}", flush=True)
    finally:
        f.close()

    print(f"U2BENCH_BREAST_DONE scanned={n_scanned} breast={n_breast}", flush=True)
    print("LABELS", dict(labels), flush=True)
    print("TASKS", dict(tasks), flush=True)


if __name__ == "__main__":
    main()
