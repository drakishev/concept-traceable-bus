"""Statistical rigor for the CB-JEPA paper — no training required.

For every eval run under outputs/weekend/<name>/predictions.json:
  - bootstrap 95% CI for malignancy (Path) F1 and BI-RADS-risk F1 (skip-missing
    convention, matching the reported tables) + extraction coverage.
For the headline pairwise claims:
  - paired bootstrap of the F1 difference (CI excluding 0 => significant).
  - McNemar exact test on per-sample correctness.
Plus malignancy ROC/AUC and the BI-RADS confusion matrix for the best model.

Clustering (revision 2, reviewer 1 item 2): BUS-CoT stores up to two lesion
crops per source image, so records are not fully independent. When
--groups_jsonl is given, every bootstrap resamples *groups* (source image /
patient, `metadata.group`) rather than records, and McNemar is additionally run
on one record per group. Both the record-level and the cluster-level numbers are
reported so the reader can see how little they differ.

Usage:
    python scripts/stats.py                       # all runs found
    python scripts/stats.py --runs dec_qwen7b x_dinov2
    python scripts/stats.py --groups_jsonl data/unified_v4g/test_buscot_only.jsonl
Outputs JSON + a LaTeX-ready summary to outputs/stats/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.slots import (  # noqa: E402
    binary_f1,
    coverage,
    extract_slots,
    paired_labels,
    pathology_label,
    risk_label,
)

random.seed(0)
N_BOOT = 2000
OUT = Path("outputs/stats")


def _load(name: str) -> list[dict] | None:
    p = f"outputs/weekend/{name}/predictions.json"
    if not os.path.exists(p):
        p2 = f"outputs/{name}/predictions.json"
        if not os.path.exists(p2):
            return None
        p = p2
    return json.load(open(p))


def _clusters(keys: list, group_of: dict | None) -> list[list[int]] | None:
    """Positions of `keys` grouped by cluster id; None when no grouping is known."""
    if not group_of:
        return None
    by_group: dict[str, list[int]] = {}
    for pos, k in enumerate(keys):
        by_group.setdefault(group_of.get(k, k), []).append(pos)
    return list(by_group.values())


def _resample(m: int, clusters: list[list[int]] | None) -> list[int]:
    """Bootstrap positions: by record, or by whole cluster when clusters are given."""
    if clusters is None:
        return [random.randrange(m) for _ in range(m)]
    c = len(clusters)
    return [pos for _ in range(c) for pos in clusters[random.randrange(c)]]


def boot_ci(pairs: list[tuple[int, int]], stat=binary_f1, n=N_BOOT,
            clusters: list[list[int]] | None = None) -> tuple[float, float, float]:
    if not pairs:
        return (0.0, 0.0, 0.0)
    point = stat(pairs)
    m = len(pairs)
    vals = []
    for _ in range(n):
        vals.append(stat([pairs[i] for i in _resample(m, clusters)]))
    vals.sort()
    return point, vals[int(0.025 * n)], vals[int(0.975 * n)]


def _label_map(pred: list[dict], endpoint: str) -> dict:
    """id -> (ref_label, pred_label|None), extracted ONCE (bootstrap-friendly)."""
    fn = pathology_label if endpoint == "pathology" else risk_label
    out = {}
    for d in pred:
        rl = fn(extract_slots(d["reference"]))
        if rl is None:
            continue
        out[d["id"]] = (rl, fn(extract_slots(d["prediction"])))
    return out


def _f1_from_map(ids_sub, lm) -> float:
    pairs = [(lm[i][0], lm[i][1]) for i in ids_sub if i in lm and lm[i][1] is not None]
    return binary_f1(pairs)


def paired_diff_ci(pred_a: list[dict], pred_b: list[dict], endpoint: str, n=N_BOOT,
                   group_of: dict | None = None):
    """Bootstrap CI of F1(A) - F1(B) over shared ids (paired resample). Labels are
    pre-extracted once, so the bootstrap loop is pure integer lookups."""
    la, lb = _label_map(pred_a, endpoint), _label_map(pred_b, endpoint)
    ids = [i for i in la if i in lb]
    point = _f1_from_map(ids, la) - _f1_from_map(ids, lb)
    m = len(ids)
    clusters = _clusters(ids, group_of)
    diffs = []
    for _ in range(n):
        sub = [ids[i] for i in _resample(m, clusters)]
        diffs.append(_f1_from_map(sub, la) - _f1_from_map(sub, lb))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n)]
    return {"delta_f1": point, "ci95": [lo, hi], "significant": (lo > 0 or hi < 0)}


def mcnemar(pred_a: list[dict], pred_b: list[dict], endpoint: str,
            group_of: dict | None = None) -> dict:
    """Exact McNemar on per-sample correctness (both models, shared ids). With
    `group_of`, only the first record of each group is used (independent units)."""
    from math import comb
    idx_a = {d["id"]: d for d in pred_a}
    idx_b = {d["id"]: d for d in pred_b}
    fn = pathology_label if endpoint == "pathology" else risk_label
    b = c = 0  # b: A right B wrong, c: A wrong B right
    seen_groups: set = set()
    for i in idx_a:
        if i not in idx_b:
            continue
        if group_of is not None:
            g = group_of.get(i, i)
            if g in seen_groups:
                continue
            seen_groups.add(g)
        rl = fn(extract_slots(idx_a[i]["reference"]))
        if rl is None:
            continue
        pa = fn(extract_slots(idx_a[i]["prediction"]))
        pb = fn(extract_slots(idx_b[i]["prediction"]))
        ca = (pa == rl)
        cb = (pb == rl)
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    nn = b + c
    # exact two-sided binomial p
    k = min(b, c)
    p = sum(comb(nn, i) for i in range(0, k + 1)) / (2 ** nn) * 2 if nn else 1.0
    p = min(p, 1.0)
    return {"b": b, "c": c, "p_value": p, "significant": p < 0.05}


def malignancy_auc(pred: list[dict]) -> dict | None:
    """ROC/AUC treating malignant as positive. Since generation is discrete (no
    probabilities), AUC here is the rank-AUC of the binary decision = balanced
    accuracy proxy; we report AUC of the 2-point ROC + sensitivity/specificity."""
    pairs = paired_labels(pred, "pathology", penalize_missing=False)
    if not pairs:
        return None
    tp = sum(1 for r, p in pairs if r == 1 and p == 1)
    fp = sum(1 for r, p in pairs if r == 0 and p == 1)
    fn_ = sum(1 for r, p in pairs if r == 1 and p == 0)
    tn = sum(1 for r, p in pairs if r == 0 and p == 0)
    sens = tp / (tp + fn_) if (tp + fn_) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = (sens + spec) / 2  # area under the single-threshold ROC
    return {"sensitivity": sens, "specificity": spec, "auc_balacc": auc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn_}


def birads_confusion(pred: list[dict]) -> dict:
    from collections import defaultdict
    cm: dict = defaultdict(lambda: defaultdict(int))
    for d in pred:
        rv = extract_slots(d["reference"]).get("birads")
        pv = extract_slots(d["prediction"]).get("birads")
        if rv and pv:
            cm[rv.upper()][pv.upper()] += 1
    return {k: dict(v) for k, v in cm.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="runs to report per-run F1/CI for (default: all under outputs/weekend)")
    ap.add_argument("--compare", nargs="*", default=None, metavar="LABEL:RUN_A:RUN_B",
                    help="pairwise comparisons to run, overriding the built-in headline set")
    ap.add_argument("--groups_jsonl", default=None,
                    help="test JSONL whose metadata.group gives the cluster of each id; "
                         "adds cluster-bootstrap CIs and one-per-group McNemar")
    ap.add_argument("--out", default=str(OUT / "stats.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    group_of: dict | None = None
    if args.groups_jsonl:
        group_of = {}
        for line in open(args.groups_jsonl):
            r = json.loads(line)
            group_of[r["id"]] = r["metadata"].get("group") or r["id"]
        print(f"clusters: {len(set(group_of.values()))} groups over {len(group_of)} records")

    if args.runs:
        names = args.runs
    else:
        names = sorted(os.path.basename(os.path.dirname(p))
                       for p in glob.glob("outputs/weekend/*/predictions.json"))

    per_run = {}
    for name in names:
        pred = _load(name)
        if pred is None:
            continue
        row = {}
        for ep, key in [("pathology", "path_f1"), ("risk", "risk_f1")]:
            pairs = paired_labels(pred, ep, penalize_missing=False)
            pt, lo, hi = boot_ci(pairs)
            row[key] = {"point": pt, "ci95": [lo, hi], "n": len(pairs),
                        "coverage": coverage(pred, ep)}
            if group_of is not None:
                # paired_labels keeps record order, so ids line up with pairs
                fn = pathology_label if ep == "pathology" else risk_label
                ids = [d["id"] for d in pred
                       if fn(extract_slots(d["reference"])) is not None
                       and fn(extract_slots(d["prediction"])) is not None]
                assert len(ids) == len(pairs)
                _, clo, chi = boot_ci(pairs, clusters=_clusters(ids, group_of))
                row[key]["ci95_cluster"] = [clo, chi]
        row["malignancy_roc"] = malignancy_auc(pred)
        per_run[name] = row
        pf = row["path_f1"]
        rf = row["risk_f1"]
        print(f"{name:22s} Path {pf['point']:.3f} [{pf['ci95'][0]:.3f},{pf['ci95'][1]:.3f}] "
              f"cov={pf['coverage']:.2f} | Risk {rf['point']:.3f} "
              f"[{rf['ci95'][0]:.3f},{rf['ci95'][1]:.3f}] cov={rf['coverage']:.2f}")
        if group_of is not None:
            print(f"{'':22s} cluster CIs: Path [{pf['ci95_cluster'][0]:.3f},"
                  f"{pf['ci95_cluster'][1]:.3f}] | Risk [{rf['ci95_cluster'][0]:.3f},"
                  f"{rf['ci95_cluster'][1]:.3f}]")

    # headline pairwise comparisons (only if both runs present)
    default_pairs = [
        ("DINOv2 vs UNI2-h (encoder dominates)", "x_dinov2", "eval_jepa_v2_mt_buscot"),
        ("concept-bottleneck vs opaque (UNI2-h)", "eval_jepa_cb_buscot", "eval_jepa_v2_mt_buscot"),
        ("concept-bottleneck vs opaque (DINOv2)", "dinov2_cb", "x_dinov2"),
        ("DINOv2 CB-JEPA vs Qwen2-VL ceiling", "x_dinov2", "eval_qwen_v2_buscot"),
    ]
    if args.compare:
        pairs_to_test = []
        for spec in args.compare:
            parts = spec.split(":")
            if len(parts) != 3:
                ap.error(f"--compare expects LABEL:RUN_A:RUN_B, got {spec!r}")
            pairs_to_test.append(tuple(parts))
    else:
        pairs_to_test = default_pairs

    comparisons = {}
    for label, a, b in pairs_to_test:
        pa, pb = _load(a), _load(b)
        if pa is None or pb is None:
            missing = [n for n, d in ((a, pa), (b, pb)) if d is None]
            print(f"  (skipping {label!r}: no predictions for {missing})", file=sys.stderr)
            continue
        comparisons[label] = {
            ep: {"paired_bootstrap": paired_diff_ci(pa, pb, ep),
                 "mcnemar": mcnemar(pa, pb, ep)}
            for ep in ("pathology", "risk")
        }
        if group_of is not None:
            for ep in ("pathology", "risk"):
                comparisons[label][ep]["paired_bootstrap_cluster"] = paired_diff_ci(
                    pa, pb, ep, group_of=group_of)
                comparisons[label][ep]["mcnemar_one_per_group"] = mcnemar(
                    pa, pb, ep, group_of=group_of)
        print(f"\n[{label}]")
        for ep in ("pathology", "risk"):
            d = comparisons[label][ep]["paired_bootstrap"]
            mc = comparisons[label][ep]["mcnemar"]
            print(f"  {ep:9s} dF1={d['delta_f1']:+.3f} CI[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] "
                  f"sig={d['significant']} | McNemar b={mc['b']} c={mc['c']} p={mc['p_value']:.3g}")
            if group_of is not None:
                dc = comparisons[label][ep]["paired_bootstrap_cluster"]
                mg = comparisons[label][ep]["mcnemar_one_per_group"]
                print(f"  {'':9s} cluster CI[{dc['ci95'][0]:+.3f},{dc['ci95'][1]:+.3f}] "
                      f"sig={dc['significant']} | McNemar(1/group) b={mg['b']} c={mg['c']} "
                      f"p={mg['p_value']:.3g}")

    json.dump({"per_run": per_run, "comparisons": comparisons,
               "groups_jsonl": args.groups_jsonl},
              open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
