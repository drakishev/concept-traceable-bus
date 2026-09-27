"""External evaluation on the U2-BENCH breast subset (never used in training).

U2-BENCH (arXiv:2505.17782) breast records come in two usable flavours:

  --task malignancy (default, n=471): benign-vs-malignant. Runs ONLY the frozen
      image encoder + predictor concept heads (no decoder generation) and reports
      malignancy F1 / accuracy / sensitivity / specificity / balanced accuracy,
      plus the malignant probability per image for calibration and ROC.
  --task birads (n=109, categories 2/3/4A/4B/4C/5): external check of the
      BI-RADS-like category and risk-group predictions, twice: from the BI-RADS
      concept head, and from the *generated report* (regex-extracted), so this
      is also the only external test of report generation (R1 #12).

Usage:
    python scripts/evaluate_u2bench.py \
        --checkpoint checkpoints/runs/dinov2_cb/final.ckpt \
        --breast_jsonl data/raw/u2bench/breast_eval/breast.jsonl \
        --output_dir outputs/eval_u2bench_breast [--task birads]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.data.slot_labels import BIRADS_CLASSES, LOW_RISK_BIRADS  # noqa: E402
from src.evaluation.slots import binary_f1, extract_slots  # noqa: E402

# Concept head class order (must match src/data/slot_labels.py + the training
# config). pathology idx 0 = benign, 1 = malignant.
PATHOLOGY_CLASSES = ["benign", "malignant"]
BIRADS_NAMES = [k for k, _ in sorted(BIRADS_CLASSES.items(), key=lambda kv: kv[1])]


def label_to_malignant(class_label: str, options: str, task: str) -> int | None:
    """Map a U2-BENCH breast record to {0=benign, 1=malignant}, or None if it is
    not a binary-malignancy record or the label cannot be resolved.

    U2-BENCH labels are dataset-dependent strings; we accept the common spellings
    and the index-into-options convention. Records whose task/labels are not
    benign-vs-malignant are dropped (returned None).
    """
    s = (class_label or "").strip().lower()
    if s in {"benign", "0", "b"}:
        return 0
    if s in {"malignant", "malignancy", "1", "m"}:
        return 1
    # The 'malignant breast cancer' task is 3-way (normal/benign/malignant);
    # 'normal' (no lesion) is not malignant -> 0.
    if s in {"normal", "n"}:
        return 0
    # Some U2-BENCH rows store the answer as an index into `options`.
    if s.isdigit() and options:
        opts = [o.strip().lower() for o in options.split(",")]
        idx = int(s)
        if 0 <= idx < len(opts):
            o = opts[idx]
            if "benign" in o and "malign" not in o:
                return 0
            if "malign" in o:
                return 1
    # If the task is explicitly malignancy and the label text contains a keyword.
    if "malign" in (task or "").lower() or "benign" in (task or "").lower():
        if "benign" in s and "malign" not in s:
            return 0
        if "malign" in s:
            return 1
    return None


class U2BBreastDataset(Dataset):
    def __init__(self, rows, transform, image_root: Path):
        self.rows = rows
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.image_root / row["image_path"]
                         if not Path(row["image_path"]).is_absolute()
                         else row["image_path"]).convert("RGB")
        return {
            "image": self.transform(img),
            "label": row["y"],
            "id": row["id"],
            "class_label": row["class_label"],
        }


def _binary_metrics(preds: list[int], golds: list[int]) -> dict:
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": len(golds),
        "f1": binary_f1(list(zip(golds, preds))),
        "accuracy": (tp + tn) / len(golds) if golds else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _birads_metrics(preds: list[str | None], golds: list[str]) -> dict:
    """Exact / adjacent category accuracy, risk-group F1 and 4A-vs-4B accuracy.
    `preds` may contain None (unextractable report); those count as wrong for
    accuracy and are excluded from the risk-group F1 (coverage is reported)."""
    order = {k: i for i, k in enumerate(BIRADS_NAMES)}
    n = len(golds)
    exact = sum(1 for p, g in zip(preds, golds) if p == g)
    adjacent = sum(1 for p, g in zip(preds, golds)
                   if p is not None and abs(order[p] - order[g]) <= 1)
    risk_pairs = [(0 if g in LOW_RISK_BIRADS else 1, 0 if p in LOW_RISK_BIRADS else 1)
                  for p, g in zip(preds, golds) if p is not None]
    sub = [(p, g) for p, g in zip(preds, golds) if g in ("4A", "4B")]
    confusion: dict[str, dict[str, int]] = {}
    for p, g in zip(preds, golds):
        confusion.setdefault(g, {})
        confusion[g][p or "none"] = confusion[g].get(p or "none", 0) + 1
    return {
        "n": n,
        "coverage": sum(p is not None for p in preds) / n if n else 0.0,
        "exact_accuracy": exact / n if n else 0.0,
        "adjacent_accuracy": adjacent / n if n else 0.0,
        "risk_group": _binary_metrics([p for _, p in risk_pairs], [g for g, _ in risk_pairs]),
        "acc_4a_vs_4b": (sum(1 for p, g in sub if p == g) / len(sub)) if sub else None,
        "n_4a_4b": len(sub),
        "confusion": confusion,
    }


def eval_birads(rows: list[dict], model, loader: DataLoader, device, out: Path,
                max_new_tokens: int, num_beams: int) -> None:
    """BI-RADS category from the concept head and from the generated report."""
    head_preds, text_preds, golds, ids, reports = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="U2-BENCH BI-RADS"):
            images = batch["image"].to(device)
            vision_features = model.x_encoder(images)
            _, aux_logits = model.predictor(vision_features, return_aux=True)
            head_preds.extend(BIRADS_NAMES[i] for i in aux_logits["birads"].argmax(-1).tolist())
            texts = model.forward_generate(images, max_new_tokens=max_new_tokens,
                                           num_beams=num_beams)
            for t in texts:
                b = extract_slots(t).get("birads")
                text_preds.append(b.upper() if b and b.upper() in BIRADS_CLASSES else None)
            reports.extend(texts)
            golds.extend(batch["class_label"])
            ids.extend(batch["id"])

    metrics = {
        "concept_head": _birads_metrics(head_preds, golds),
        "generated_report": _birads_metrics(text_preds, golds),
        "gold_distribution": {g: golds.count(g) for g in BIRADS_NAMES if g in golds},
        "decoding": {"max_new_tokens": max_new_tokens, "num_beams": num_beams},
    }
    logger.info("U2-BENCH breast BI-RADS: %s", json.dumps(metrics, indent=2))
    with open(out / "metrics_birads.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "predictions_birads.json", "w") as f:
        json.dump([{"id": i, "gold": g, "head": h, "report_birads": t, "report": r}
                   for i, g, h, t, r in zip(ids, golds, head_preds, text_preds, reports)],
                  f, indent=2)
    logger.info("Saved BI-RADS metrics + predictions to %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--breast_jsonl", required=True)
    ap.add_argument("--output_dir", default="outputs/eval_u2bench_breast")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--task", choices=["malignancy", "birads"], default="malignancy")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load breast records and filter to the requested task ──────────────────
    raw = [json.loads(line) for line in open(args.breast_jsonl)]
    logger.info("Loaded %d breast records from %s", len(raw), args.breast_jsonl)
    from collections import Counter
    logger.info("Raw tasks: %s", dict(Counter(r["classification_task"] for r in raw)))
    logger.info("Raw class_labels: %s", dict(Counter(r["class_label"] for r in raw)))

    rows = []
    for r in raw:
        if args.task == "birads":
            if r["classification_task"] != "BIRADS" or r["class_label"] not in BIRADS_CLASSES:
                continue
            r["y"] = BIRADS_CLASSES[r["class_label"]]
            rows.append(r)
            continue
        y = label_to_malignant(r["class_label"], r.get("options", ""),
                               r["classification_task"])
        if y is None:
            continue
        r["y"] = y
        rows.append(r)
    logger.info("Kept %d binary-malignancy records (dropped %d); label dist: %s",
                len(rows), len(raw) - len(rows),
                dict(Counter(r["y"] for r in rows)))
    if not rows:
        logger.error("No usable malignancy records; aborting. Inspect raw labels above.")
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.data.datasets.bus_cot_reports import _ultrasound_transform
    from src.model.report_model import ConceptReportModel

    logger.info("Loading checkpoint: %s", args.checkpoint)
    model = ConceptReportModel.load_from_checkpoint(args.checkpoint, stage="finetune")
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    transform = _ultrasound_transform(image_size=args.image_size, train=False)
    ds = U2BBreastDataset(rows, transform, image_root=Path("."))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    if args.task == "birads":
        eval_birads(rows, model, loader, device, out, args.max_new_tokens, args.num_beams)
        return

    # ── Run the pathology concept head ─────────────────────────────────────────
    preds, probs, golds, ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="U2-BENCH eval"):
            images = batch["image"].to(device)
            vision_features = model.x_encoder(images)
            _, aux_logits = model.predictor(vision_features, return_aux=True)
            path_logits = aux_logits["pathology"].float()  # [B, 2]
            preds.extend(path_logits.argmax(-1).cpu().tolist())
            probs.extend(path_logits.softmax(-1)[:, 1].cpu().tolist())
            golds.extend(batch["label"].tolist())
            ids.extend(batch["id"])

    # ── Metrics ────────────────────────────────────────────────────────────────
    bm = _binary_metrics(preds, golds)
    metrics = {
        "n": bm["n"],
        "malignancy_f1": bm["f1"],
        "accuracy": bm["accuracy"],
        "sensitivity": bm["sensitivity"],
        "specificity": bm["specificity"],
        "precision": bm["precision"],
        "balanced_accuracy": bm["balanced_accuracy"],
        "confusion": bm["confusion"],
        "checkpoint": args.checkpoint,
    }
    logger.info("U2-BENCH breast malignancy: %s", json.dumps(metrics, indent=2))

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "predictions.json", "w") as f:
        json.dump([{"id": i, "pred": p, "gold": g, "prob_malignant": pr,
                    "pred_label": PATHOLOGY_CLASSES[p], "gold_label": PATHOLOGY_CLASSES[g]}
                   for i, p, g, pr in zip(ids, preds, golds, probs)], f, indent=2)
    logger.info("Saved metrics + predictions to %s", out)


if __name__ == "__main__":
    main()
