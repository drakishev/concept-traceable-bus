"""Calibration and sensitivity-first operating points for the pathology concept head.

Revision 2, reviewer 1 item 13: F1 at the argmax threshold treats a missed
cancer like a false alarm, and a probability is only useful clinically if it is
calibrated. This script reads the per-sample malignant probabilities dumped by
scripts/dump_concept_probs.py and reports, for each split:

  * Brier score, expected calibration error (equal-width bins) and the
    reliability-diagram bins (for the figure);
  * sensitivity / specificity / PPV / NPV at the argmax threshold (0.5) and at a
    sensitivity-oriented threshold chosen ONCE on the validation split (the
    smallest threshold reaching --target_sensitivity there) and then applied
    unchanged to the internal test split and to every external split, so the
    external operating point is prospective rather than tuned on the test data.

Usage:
    python scripts/calibration.py --val outputs/stats/probs_val.json \\
        --test outputs/stats/probs_internal.json \\
        --external u2bench=outputs/stats/probs_u2bench.json \\
                   breast=outputs/stats/probs_breast.json \\
        --output outputs/stats/calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

N_BINS = 10


def _load(path: str) -> tuple[list[float], list[int]]:
    d = json.load(open(path))
    return [s["p_malignant"] for s in d["per_sample"]], [int(s["y"]) for s in d["per_sample"]]


def brier(p: list[float], y: list[int]) -> float:
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)


def reliability(p: list[float], y: list[int], n_bins: int = N_BINS) -> dict:
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        last = b == n_bins - 1
        members = [(pi, yi) for pi, yi in zip(p, y) if lo <= pi < hi or (last and pi == 1.0)]
        if not members:
            bins.append({"lo": lo, "hi": hi, "n": 0, "confidence": None, "accuracy": None})
            continue
        conf = sum(pi for pi, _ in members) / len(members)
        acc = sum(yi for _, yi in members) / len(members)
        ece += len(members) / len(p) * abs(conf - acc)
        bins.append({"lo": lo, "hi": hi, "n": len(members),
                     "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return {"ece": round(ece, 4), "bins": bins}


def operating_point(p: list[float], y: list[int], thr: float) -> dict:
    tp = sum(1 for pi, yi in zip(p, y) if pi >= thr and yi == 1)
    fn = sum(1 for pi, yi in zip(p, y) if pi < thr and yi == 1)
    tn = sum(1 for pi, yi in zip(p, y) if pi < thr and yi == 0)
    fp = sum(1 for pi, yi in zip(p, y) if pi >= thr and yi == 0)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": round(thr, 6),
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "ppv": round(prec, 4), "npv": round(tn / (tn + fn), 4) if (tn + fn) else None,
        "f1": round(2 * prec * sens / (prec + sens), 4) if (prec + sens) else 0.0,
        "missed_malignancies": fn, "n_malignant": tp + fn,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_for_sensitivity(p: list[float], y: list[int], target: float) -> float:
    """Largest threshold whose sensitivity on (p, y) is at least `target`: the k-th highest
    positive score with k = ceil(target * positives). Rounding k instead gave 158 of 176
    (0.898) for a 0.90 target."""
    pos = sorted((pi for pi, yi in zip(p, y) if yi == 1), reverse=True)
    if not pos:
        return 0.5
    k = max(1, math.ceil(target * len(pos) - 1e-9))
    return pos[min(k, len(pos)) - 1]


def summarize(p: list[float], y: list[int], thr_sens: float, target: float) -> dict:
    return {
        "n": len(p), "n_malignant": sum(y), "prevalence": round(sum(y) / len(y), 4),
        "brier": round(brier(p, y), 4),
        "reliability": reliability(p, y),
        "argmax_threshold": operating_point(p, y, 0.5),
        f"sensitivity_{target:.2f}_threshold_from_val": operating_point(p, y, thr_sens),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", required=True, help="probs JSON on the validation split")
    ap.add_argument("--test", required=True, help="probs JSON on the internal test split")
    ap.add_argument("--external", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--target_sensitivity", type=float, default=0.90)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pv, yv = _load(args.val)
    thr = threshold_for_sensitivity(pv, yv, args.target_sensitivity)
    out = {
        "target_sensitivity": args.target_sensitivity,
        "threshold_chosen_on_val": round(thr, 6),
        "val": summarize(pv, yv, thr, args.target_sensitivity),
        "internal_test": summarize(*_load(args.test), thr, args.target_sensitivity),
        "external": {},
    }
    for spec in args.external:
        name, path = spec.split("=", 1)
        out["external"][name] = summarize(*_load(path), thr, args.target_sensitivity)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    key = f"sensitivity_{args.target_sensitivity:.2f}_threshold_from_val"
    print(f"threshold for {args.target_sensitivity:.0%} sensitivity on val: {thr:.4f}")
    for name, d in [("val", out["val"]), ("internal_test", out["internal_test"]),
                    *out["external"].items()]:
        a, s = d["argmax_threshold"], d[key]
        print(f"{name:14s} n={d['n']:4d} Brier={d['brier']:.3f} "
              f"ECE={d['reliability']['ece']:.3f} | "
              f"argmax sens={a['sensitivity']:.3f} spec={a['specificity']:.3f} "
              f"| @val-thr sens={s['sensitivity']:.3f} spec={s['specificity']:.3f} "
              f"missed={s['missed_malignancies']}/{s['n_malignant']}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
