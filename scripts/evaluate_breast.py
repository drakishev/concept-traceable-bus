"""Descriptor-level external evaluation on BrEaST (Breast-Lesions-USG).

BrEaST is the only public breast-ultrasound set whose lesions carry the full
BI-RADS descriptor lexicon, so it is the external test of whether the
*findings* in a generated report are right, not only the assessment
(revision 2, reviewer 1 items 7 and 12). Nothing here touches a GPU: it scores
a predictions.json produced by

    python scripts/evaluate_jepa.py --checkpoint <ckpt> \\
        --test_jsonl data/raw/breast/breast_eval.jsonl --output_dir outputs/breast/<run> \\
        --train_config configs/train/finetune_jepa_v5.yaml

against the structured fields stored in `metadata` by src/data/datasets/breast.py.

Scored per report (coverage = fraction whose report exposes the slot):
  pathology      benign / malignant                    (F1, sensitivity, specificity)
  BI-RADS        exact, adjacent, risk group, 4A-vs-4B (BrEaST categories 2-5)
  shape          oval / round / irregular
  margin         circumscribed vs not circumscribed    (BUS-CoT has no "spiculated")
  echogenicity   6 classes, BUS-CoT "slightly hypoechoic" folded into hypoechoic
  calcification  absent vs present

Usage:
    python scripts/evaluate_breast.py --predictions outputs/breast/<run>/predictions.json \\
        --eval_jsonl data/raw/breast/breast_eval.jsonl \\
        --output outputs/breast/<run>/descriptors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_u2bench import _binary_metrics, _birads_metrics  # noqa: E402

from src.data.slot_labels import BIRADS_CLASSES  # noqa: E402
from src.evaluation.slots import extract_slots  # noqa: E402

#: BrEaST vocabulary for the gold column (heterogeneous / complex spelled as in
#: the TCIA sheet) and for the BUS-CoT echo enum the shared extractor returns.
ECHO_CANON = {
    "hypoechoic": "hypoechoic", "slightly hypoechoic": "hypoechoic",
    "hyperechoic": "hyperechoic", "isoechoic": "isoechoic", "anechoic": "anechoic",
    "heterogeneous": "heterogeneous", "heterogeneous echo": "heterogeneous",
    "complex cystic and solid": "complex cystic/solid",
    "mixed cystic-solid": "complex cystic/solid",
    "lowecho": "hypoechoic", "slightlylowecho": "hypoechoic", "highecho": "hyperechoic",
    "heterogeneousecho": "heterogeneous", "cysticsolidmixedecho": "complex cystic/solid",
    "noecho": "anechoic",
}


def _pred_fields(text: str) -> dict[str, str | int | None]:
    """Canonical descriptor values read from a generated report (None = absent).

    `extract_slots` returns BUS-CoT enum keys for the descriptors (margins
    regular / partiallyregular / irregular, echo lowecho / ..., calcification
    nocalcification / ...); BrEaST's binary margin column counts only fully
    circumscribed margins as circumscribed."""
    s = extract_slots(text)
    margins = s.get("margins")
    echo = s.get("echogenicity")
    calc = s.get("calcification")
    birads = (s.get("birads") or "").upper() or None
    pathology = s.get("pathology")
    return {
        "pathology": None if pathology is None else int(pathology.lower() == "malignant"),
        "birads": birads if birads in BIRADS_CLASSES else None,
        "shape": (s.get("shape") or "").lower() or None,
        "margin_circumscribed": None if margins is None else int(margins == "regular"),
        "echogenicity": ECHO_CANON.get(echo) if echo else None,
        "calcification": None if calc is None else int(calc != "nocalcification"),
    }


def _categorical(preds: list, golds: list) -> dict:
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
    n = len(golds)
    confusion: dict[str, dict[str, int]] = {}
    for p, g in zip(preds, golds):
        confusion.setdefault(str(g), {})
        confusion[str(g)][str(p)] = confusion[str(g)].get(str(p), 0) + 1
    return {
        "n": n,
        "coverage": len(pairs) / n if n else 0.0,
        "accuracy_on_covered": sum(p == g for p, g in pairs) / len(pairs) if pairs else None,
        "accuracy_missing_wrong": sum(p == g for p, g in pairs) / n if n else 0.0,
        "gold_distribution": dict(Counter(str(g) for g in golds)),
        "confusion": confusion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--eval_jsonl", default="data/raw/breast/breast_eval.jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gold = {}
    for line in open(args.eval_jsonl):
        r = json.loads(line)
        m = r["metadata"]
        gold[r["id"]] = {
            "pathology": int(m["pathology"] == "malignant"),
            "birads": m["birads"],
            "shape": m["shape"],
            "margin_circumscribed": None if m["margin_circumscribed"] is None
            else int(m["margin_circumscribed"]),
            "echogenicity": ECHO_CANON.get(m["echogenicity"], m["echogenicity"]),
            "calcification": 0 if m["calcifications"] == "no" else 1,
        }
    preds = {d["id"]: _pred_fields(d["prediction"]) for d in json.load(open(args.predictions))}
    ids = [i for i in gold if i in preds]

    def col(field: str, src: dict) -> list:
        return [src[i][field] for i in ids]

    path_pairs = [(p, g) for p, g in zip(col("pathology", preds), col("pathology", gold))
                  if p is not None]
    out = {
        "n": len(ids),
        "pathology": {
            "coverage": len(path_pairs) / len(ids) if ids else 0.0,
            **_binary_metrics([p for p, _ in path_pairs], [g for _, g in path_pairs]),
        },
        "birads": _birads_metrics(col("birads", preds), col("birads", gold)),
        "shape": _categorical(col("shape", preds), col("shape", gold)),
        "margin_circumscribed": _categorical(col("margin_circumscribed", preds),
                                             col("margin_circumscribed", gold)),
        "echogenicity": _categorical(col("echogenicity", preds), col("echogenicity", gold)),
        "calcification": _categorical(col("calcification", preds), col("calcification", gold)),
        "predictions": args.predictions,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"BrEaST descriptor evaluation, n={len(ids)}")
    p = out["pathology"]
    print(f"  pathology     cov={p['coverage']:.2f} F1={p['f1']:.3f} "
          f"sens={p['sensitivity']:.3f} spec={p['specificity']:.3f}")
    b = out["birads"]
    print(f"  BI-RADS       cov={b['coverage']:.2f} exact={b['exact_accuracy']:.3f} "
          f"adjacent={b['adjacent_accuracy']:.3f} riskF1={b['risk_group']['f1']:.3f} "
          f"4A/4B={b['acc_4a_vs_4b']}")
    for k in ("shape", "margin_circumscribed", "echogenicity", "calcification"):
        d = out[k]
        acc = d["accuracy_on_covered"]
        print(f"  {k:14s}cov={d['coverage']:.2f} acc(covered)="
              f"{acc if acc is None else round(acc, 3)} acc(missing=wrong)="
              f"{d['accuracy_missing_wrong']:.3f}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
