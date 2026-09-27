"""Dump concept-head probabilities and compute malignancy ROC/PR curves.

Reviewer item 11: the manuscript's Limitations claimed that discrete report generation
"does not afford a true malignancy ROC". That is wrong -- the concept heads emit softmax
distributions (the qualitative table already prints them as 99%/51%/91%), so a
probabilistic ROC/PR curve is computable. The saved predictions.json files hold text
only, so this script re-runs the frozen encoder + predictor aux heads (no decoder
generation, no training) and dumps per-sample probabilities.

Also reports the sensitivity/specificity trade-off across thresholds, which the paper
needs in order to discuss the operating point honestly (at the argmax threshold
sensitivity is 0.730, i.e. 27% of malignancies missed).

Usage:
    # internal BUS-CoT test split
    python scripts/dump_concept_probs.py \
        --checkpoint checkpoints/runs/dinov2_cb/final.ckpt \
        --jsonl data/split_augmented/test.jsonl \
        --restrict-to outputs/runs/dinov2_cb/predictions.json \
        --output outputs/stats/probs_internal.json

    # external U2-BENCH breast subset
    python scripts/dump_concept_probs.py \
        --checkpoint checkpoints/runs/dinov2_cb/final.ckpt \
        --u2bench data/raw/u2bench/breast_eval/breast.jsonl \
        --output outputs/stats/submitted_model_u2bench.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
random.seed(0)
N_BOOT = 2000


# ─────────────────────────── metric helpers ────────────────────────────────
def roc_points(scores: list[float], labels: list[int]) -> list[dict]:
    """ROC/PR sweep over every distinct threshold (positive = malignant)."""
    pos = sum(labels)
    neg = len(labels) - pos
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    pts = [{"threshold": 1.01, "tpr": 0.0, "fpr": 0.0, "precision": 1.0, "recall": 0.0}]
    for rank, i in enumerate(order, 1):
        tp += labels[i] == 1
        fp += labels[i] == 0
        pts.append({
            "threshold": round(scores[i], 6),
            "tpr": tp / pos if pos else 0.0,
            "fpr": fp / neg if neg else 0.0,
            "precision": tp / rank,
            "recall": tp / pos if pos else 0.0,
        })
    return pts


def auc_trapz(xs: list[float], ys: list[float]) -> float:
    return sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2 for i in range(1, len(xs)))


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (equals the Mann-Whitney U statistic; ties averaged)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def average_precision(scores: list[float], labels: list[int]) -> float:
    pts = roc_points(scores, labels)
    ap = 0.0
    prev_r = 0.0
    for p in pts[1:]:
        ap += (p["recall"] - prev_r) * p["precision"]
        prev_r = p["recall"]
    return ap


def boot_auc_ci(scores: list[float], labels: list[int], n: int = N_BOOT) -> list[float]:
    m = len(scores)
    vals = []
    for _ in range(n):
        idx = [random.randrange(m) for _ in range(m)]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        if 0 < sum(y) < len(y):
            vals.append(roc_auc(s, y))
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals))], 4)]


def operating_points(scores: list[float], labels: list[int]) -> list[dict]:
    """Sensitivity/specificity at clinically relevant thresholds + Youden's J."""
    out = []
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 1)
        fn = sum(1 for s, y in zip(scores, labels) if s < thr and y == 1)
        tn = sum(1 for s, y in zip(scores, labels) if s < thr and y == 0)
        fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        out.append({"threshold": thr, "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4), "youden_j": round(sens + spec - 1, 4)})
    return out


# ─────────────────────────── inference ────────────────────────────────
def run_internal(args, model, device) -> tuple[list[str], list[float], list[int]]:
    from src.data.datasets.bus_cot_reports import BUSCoTReportDataset, _ultrasound_transform
    from src.data.slot_labels import IGNORE_INDEX

    keep = None
    if args.restrict_to and Path(args.restrict_to).exists():
        keep = {d["id"] for d in json.load(open(args.restrict_to))}
        logger.info("Restricting to %d evaluated ids", len(keep))

    ds = BUSCoTReportDataset(jsonl_path=args.jsonl,
                           transform=_ultrasound_transform(args.image_size, train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    ids, probs, golds = [], [], []
    with torch.no_grad():
        for batch in loader:
            _, aux = model.predictor(model.x_encoder(batch["image"].to(device)),
                                     return_aux=True)
            p = torch.softmax(aux["pathology"].float(), -1)[:, 1].cpu().tolist()
            y = batch["labels"]["pathology"].tolist()
            bid = batch.get("id") or [""] * len(p)
            for i, (pi, yi) in enumerate(zip(p, y)):
                rid = bid[i] if isinstance(bid, list) else ""
                if keep is not None and rid not in keep:
                    continue
                if yi == IGNORE_INDEX:
                    continue
                ids.append(rid)
                probs.append(pi)
                golds.append(int(yi))
    return ids, probs, golds


def run_u2bench(args, model, device) -> tuple[list[str], list[float], list[int]]:
    from scripts.evaluate_u2bench import U2BBreastDataset, label_to_malignant
    from src.data.datasets.bus_cot_reports import _ultrasound_transform

    rows = []
    for line in open(args.u2bench):
        r = json.loads(line)
        y = label_to_malignant(r["class_label"], r.get("options", ""),
                               r["classification_task"])
        if y is None:
            continue
        r["y"] = y
        rows.append(r)
    logger.info("U2-BENCH: %d binary-malignancy records", len(rows))

    ds = U2BBreastDataset(rows, _ultrasound_transform(args.image_size, train=False),
                          image_root=Path("."))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    ids, probs, golds = [], [], []
    with torch.no_grad():
        for batch in loader:
            _, aux = model.predictor(model.x_encoder(batch["image"].to(device)),
                                     return_aux=True)
            probs.extend(torch.softmax(aux["pathology"].float(), -1)[:, 1].cpu().tolist())
            golds.extend(int(v) for v in batch["label"].tolist())
            ids.extend(batch["id"])
    return ids, probs, golds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", default="data/split_augmented/test.jsonl")
    ap.add_argument("--restrict-to", default=None)
    ap.add_argument("--u2bench", default=None,
                    help="If given, evaluate the U2-BENCH breast subset instead.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from src.model.report_model import ConceptReportModel
    logger.info("Loading %s", args.checkpoint)
    model = ConceptReportModel.load_from_checkpoint(args.checkpoint, stage="finetune")
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    if args.u2bench:
        ids, probs, golds = run_u2bench(args, model, device)
        split = "u2bench_breast"
    else:
        ids, probs, golds = run_internal(args, model, device)
        split = "buscot_test"

    if not probs:
        logger.error("No samples scored; aborting.")
        return

    res = {
        "split": split,
        "checkpoint": args.checkpoint,
        "n": len(probs),
        "n_malignant": sum(golds),
        "prevalence_malignant": round(sum(golds) / len(golds), 4),
        "roc_auc": round(roc_auc(probs, golds), 4),
        "roc_auc_ci95": boot_auc_ci(probs, golds),
        "average_precision": round(average_precision(probs, golds), 4),
        "operating_points": operating_points(probs, golds),
        "roc_curve": roc_points(probs, golds),
        "per_sample": [{"id": i, "p_malignant": round(p, 6), "y": y}
                       for i, p, y in zip(ids, probs, golds)],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))

    print(f"=== Malignancy ROC ({split}), n={res['n']}, "
          f"prevalence={res['prevalence_malignant']:.3f} ===")
    print(f"  AUC = {res['roc_auc']:.4f}  95% CI {res['roc_auc_ci95']}")
    print(f"  Average precision = {res['average_precision']:.4f}")
    print("  threshold  sens   spec   J")
    for op in res["operating_points"]:
        print(f"    {op['threshold']:.1f}     {op['sensitivity']:.3f}  "
              f"{op['specificity']:.3f}  {op['youden_j']:+.3f}")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
