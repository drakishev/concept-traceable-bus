"""Slot-accuracy evaluation for BUS-CoT templated reports.

Extracts structured clinical fields from both prediction and reference,
then computes per-slot accuracy and a confusion matrix for BI-RADS.

This is more honest than BLEU/ROUGE for templated data: the model can
score near-perfect on n-gram metrics just by memorising the template
skeleton. Slot accuracy measures whether it predicts the *right values*.

Slots extracted (src/evaluation/slots.py, any of the four training styles):
    orientation   — parallel / not parallel
    margins       — source enum: regular / partiallyregular / irregular
    shape         — oval / round / irregular
    echogenicity  — source enum (7 classes)
    calcification — source enum (6 classes)
    birads        — 2 / 3 / 4A / 4B / 4C / 5 / 6
    pathology     — benign / malignant
    answer        — 0 / 1  (direct classification label)

Usage:
    python scripts/evaluate_slots.py \
        --predictions outputs/eval_jepa/predictions.json \
        --output_dir  outputs/eval_jepa
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ── Slot extractor: the shared, style-agnostic instrument ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.slots import extract_slots  # noqa: E402

# ── Metrics ────────────────────────────────────────────────────────────────

def slot_accuracy(refs: list[dict], preds: list[dict], slot: str) -> dict:
    correct = 0
    total = 0
    missing_ref = 0
    missing_pred = 0

    # per-class TP/FP/FN for macro-F1
    class_tp: dict[str, int] = defaultdict(int)
    class_fp: dict[str, int] = defaultdict(int)
    class_fn: dict[str, int] = defaultdict(int)

    for r, p in zip(refs, preds):
        rv = r.get(slot)
        pv = p.get(slot)
        if rv is None:
            missing_ref += 1
            continue
        total += 1
        rv_l = rv.lower()
        if pv is None:
            missing_pred += 1
            class_fn[rv_l] += 1
            continue
        pv_l = pv.lower()
        if rv_l == pv_l:
            correct += 1
            class_tp[rv_l] += 1
        else:
            class_fn[rv_l] += 1
            class_fp[pv_l] += 1

    # macro-averaged precision, recall, F1 across classes
    all_classes = set(class_tp) | set(class_fn)
    per_class_f1 = []
    per_class_p = []
    per_class_r = []
    for c in all_classes:
        tp, fp, fn = class_tp[c], class_fp[c], class_fn[c]
        p_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        per_class_p.append(p_)
        per_class_r.append(r_)
        per_class_f1.append(f_)

    macro_p  = sum(per_class_p)  / len(per_class_p)  if per_class_p  else 0.0
    macro_r  = sum(per_class_r)  / len(per_class_r)  if per_class_r  else 0.0
    macro_f1 = sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0

    return {
        "accuracy":  correct / total if total > 0 else 0.0,
        "precision": macro_p,
        "recall":    macro_r,
        "f1":        macro_f1,
        "correct":   correct,
        "total":     total,
        "missing_pred": missing_pred,
    }


def confusion_matrix(refs: list[dict], preds: list[dict], slot: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r, p in zip(refs, preds):
        rv = r.get(slot)
        pv = p.get(slot)
        if rv is None or pv is None:
            continue
        matrix[rv.upper()][pv.upper()] += 1
    return {k: dict(v) for k, v in matrix.items()}


def binary_metrics(refs: list[dict], preds: list[dict], slot: str) -> dict:
    tp = fp = tn = fn = 0
    for r, p in zip(refs, preds):
        rv = r.get(slot)
        pv = p.get(slot)
        if rv is None or pv is None:
            continue
        r_pos = rv in ("1", "malignant")
        p_pos = pv in ("1", "malignant")
        if r_pos and p_pos:
            tp += 1
        elif not r_pos and p_pos:
            fp += 1
        elif r_pos and not p_pos:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/eval_jepa/predictions.json")
    parser.add_argument("--output_dir",  default="outputs/eval_jepa")
    parser.add_argument("--output",      default=None, help="alias for --output_dir")
    args = parser.parse_args()
    if args.output:
        args.output_dir = args.output

    with open(args.predictions) as f:
        data = json.load(f)

    ref_slots  = [extract_slots(d["reference"])  for d in data]
    pred_slots = [extract_slots(d["prediction"]) for d in data]

    # ── Per-slot accuracy ──────────────────────────────────────────────────
    slots = ["orientation", "margins", "shape", "echogenicity", "calcification",
             "birads", "pathology", "answer"]
    results: dict = {}

    print("\n=== Slot Metrics ===")
    print(f"{'Slot':<14} {'Accuracy':>8}  {'Precision':>9}  {'Recall':>8}  {'F1':>8}  "
          f"{'Miss-pred':>9}")
    print("-" * 66)
    for s in slots:
        acc = slot_accuracy(ref_slots, pred_slots, s)
        results[s] = acc
        print(
            f"{s:<14} {acc['accuracy']:>8.3f}  {acc['precision']:>9.3f}  "
            f"{acc['recall']:>8.3f}  {acc['f1']:>8.3f}  {acc['missing_pred']:>9d}"
        )

    # ── Binary classification (malignant/benign + 0/1) ─────────────────────
    print("\n=== Binary Classification (answer: 0/1) ===")
    bm_answer = binary_metrics(ref_slots, pred_slots, "answer")
    for k, v in bm_answer.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results["binary_answer"] = bm_answer

    print("\n=== Binary Classification (pathology: malignant/benign) ===")
    bm_path = binary_metrics(ref_slots, pred_slots, "pathology")
    for k, v in bm_path.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results["binary_pathology"] = bm_path

    # ── BI-RADS confusion matrix ───────────────────────────────────────────
    print("\n=== BI-RADS Confusion Matrix (row=ref, col=pred) ===")
    cm = confusion_matrix(ref_slots, pred_slots, "birads")
    all_labels = sorted(set(list(cm.keys()) + [k for v in cm.values() for k in v]))
    header = f"{'':>6}" + "".join(f"{lab:>6}" for lab in all_labels)
    print(header)
    for rl in all_labels:
        row = f"{rl:>6}" + "".join(f"{cm.get(rl, {}).get(pl, 0):>6}" for pl in all_labels)
        print(row)
    results["birads_confusion"] = cm

    # ── BI-RADS malignancy grouping (2-3 = low-risk vs 4-6 = high-risk) ──
    print("\n=== BI-RADS Risk Group (low: 2-3, high: 4A-6) ===")
    low_risk = {"2", "3"}
    tp = fp = tn = fn = 0
    for r, p in zip(ref_slots, pred_slots):
        rv, pv = r.get("birads"), p.get("birads")
        if rv is None or pv is None:
            continue
        r_high = rv.upper() not in low_risk
        p_high = pv.upper() not in low_risk
        if r_high and p_high:
            tp += 1
        elif not r_high and p_high:
            fp += 1
        elif r_high and not p_high:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    grp = {
        "accuracy":  (tp + tn) / total if total > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1":        0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    pr_sum = grp["precision"] + grp["recall"]
    grp["f1"] = 2 * grp["precision"] * grp["recall"] / pr_sum if pr_sum > 0 else 0.0
    for k, v in grp.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results["birads_risk_group"] = grp

    # ── Save ───────────────────────────────────────────────────────────────
    out = Path(args.output_dir) / "slot_metrics.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
