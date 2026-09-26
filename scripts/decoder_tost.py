"""Decoder-scale equivalence on a patient-grouped split (reviewers 1 #3, 2 #4, 3 #3).

For every Qwen2.5 size against the 0.5B at the same seed: paired cluster
bootstrap of the F1 difference (resampling patients, `metadata.group`), a TOST
at the pre-specified +/-0.02 margin (equivalence iff the whole 90% CI lies inside
it) and the smallest symmetric margin the data support. A dose-response check
fits F1 against log10(parameters) over every run of the sweep.

Usage:
    python scripts/decoder_tost.py --prefix h_ \\
        --groups_jsonl data/unified_v5/test_buscot_only.jsonl \\
        --output outputs/stats/tost_decoder_scale_batch7.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.slots import (  # noqa: E402
    binary_f1,
    extract_slots,
    pathology_label,
    risk_label,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_revision import _load  # noqa: E402

MARGIN = 0.02
N_BOOT = 2000
SIZES = {"0_5b": 0.5, "1_5b": 1.5, "3b": 3, "7b": 7, "14b": 14, "32b": 32, "72b": 72}


def _groups(jsonl: str) -> dict[str, str]:
    with open(jsonl) as f:
        return {r["id"]: r["metadata"]["group"] for r in map(json.loads, f)}


def cluster_tost(pred_a: list[dict], pred_b: list[dict], endpoint: str,
                 group_of: dict[str, str]) -> dict:
    fn = pathology_label if endpoint == "pathology" else risk_label
    a, b = {d["id"]: d for d in pred_a}, {d["id"]: d for d in pred_b}
    rows, by_group = [], {}
    for i in sorted(a.keys() & b.keys()):
        ref = fn(extract_slots(a[i]["reference"]))
        pa, pb = fn(extract_slots(a[i]["prediction"])), fn(extract_slots(b[i]["prediction"]))
        if None in (ref, pa, pb):
            continue
        by_group.setdefault(group_of.get(i, i), []).append(len(rows))
        rows.append((ref, pa, pb))
    clusters = list(by_group.values())

    def delta(sample: list[tuple]) -> float:
        return (binary_f1([(r, x) for r, x, _ in sample])
                - binary_f1([(r, y) for r, _, y in sample]))

    obs = delta(rows)
    rng = random.Random(0)
    diffs = sorted(
        delta([rows[p] for _ in clusters for p in clusters[rng.randrange(len(clusters))]])
        for _ in range(N_BOOT))
    lo90, hi90 = diffs[int(0.05 * N_BOOT)], diffs[int(0.95 * N_BOOT)]
    lo95, hi95 = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]
    return {
        "n": len(rows), "n_groups": len(clusters), "delta_f1": obs,
        "ci90": [lo90, hi90], "ci95": [lo95, hi95],
        "p_tost": round(max(sum(d <= -MARGIN for d in diffs),
                            sum(d >= MARGIN for d in diffs)) / N_BOOT, 4),
        "equivalent_at_margin": bool(lo90 > -MARGIN and hi90 < MARGIN),
        "supported_margin": max(abs(lo90), abs(hi90)),
    }


def dose_response(prefix: str, rows: list[dict]) -> dict:
    """OLS slope of F1 on log10(params) over all sweep runs, per endpoint, with a
    t-based 95% interval. The runs share one test set, so this interval treats
    them as independent and is an approximation."""
    out = {}
    for endpoint in ("path_f1", "risk_f1"):
        x = np.log10([r["params_b"] for r in rows])
        y = np.array([r[endpoint] for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        se = np.sqrt(resid @ resid / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean())))
        from scipy.stats import t as student_t
        q = student_t.ppf(0.975, len(x) - 2)
        out[endpoint] = {"slope_per_decade": float(slope),
                         "ci95": [float(slope - q * se), float(slope + q * se)],
                         "n_runs": len(x)}
    return out


def slope_bootstrap(names: list[str], params: list[float], group_of: dict[str, str]) -> dict:
    """Patient-cluster bootstrap of the dose-response slope: every replicate resamples
    patients once and re-scores all runs on that sample, so the shared test set is
    accounted for (the OLS interval above treats runs as independent)."""
    preds = [{d["id"]: d for d in _load(n)} for n in names]
    ids = sorted(set.intersection(*(set(p) for p in preds)))
    by_group: dict[str, list[int]] = {}
    for pos, i in enumerate(ids):
        by_group.setdefault(group_of.get(i, i), []).append(pos)
    clusters = [np.array(v) for v in by_group.values()]
    x = np.log10(params)
    out = {}
    for endpoint, fn in (("path_f1", pathology_label), ("risk_f1", risk_label)):
        ref = np.array([fn(extract_slots(preds[0][i]["reference"])) for i in ids], dtype=float)
        hyp = np.array([[fn(extract_slots(p[i]["prediction"])) for i in ids] for p in preds],
                       dtype=float)

        def f1s(idx: np.ndarray) -> np.ndarray:
            r, h = ref[idx], hyp[:, idx]
            ok = ~np.isnan(r)[None, :] & ~np.isnan(h)
            tp = ((h == 1) & (r == 1) & ok).sum(1)
            fp = ((h == 1) & (r == 0) & ok).sum(1)
            fn_ = ((h == 0) & (r == 1) & ok).sum(1)
            return 2 * tp / np.maximum(2 * tp + fp + fn_, 1)

        rng = np.random.default_rng(0)
        slopes = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(clusters), len(clusters))
            idx = np.concatenate([clusters[k] for k in pick])
            slopes.append(np.polyfit(x, f1s(idx), 1)[0])
        slopes.sort()
        out[endpoint] = {"slope_per_decade": float(np.polyfit(x, f1s(np.arange(len(ids))), 1)[0]),
                         "ci95_patient_bootstrap": [float(slopes[int(0.025 * N_BOOT)]),
                                                    float(slopes[int(0.975 * N_BOOT)])]}
    return out


def main() -> None:
    import csv

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="h_")
    ap.add_argument("--groups_jsonl", required=True)
    ap.add_argument("--results_csv", default="outputs/weekend7_results.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    group_of = _groups(args.groups_jsonl)
    results = {r["name"]: r for r in csv.DictReader(open(args.results_csv))}
    pairs, sweep = {}, []
    for tag, params in SIZES.items():
        for seed in (1, 2, 3):
            name = f"{args.prefix}dec_qwen{tag}_s{seed}"
            if name not in results:
                continue
            sweep.append({"name": name, "params_b": params,
                          "path_f1": float(results[name]["path_f1"]),
                          "risk_f1": float(results[name]["risk_f1"])})
            ref = f"{args.prefix}dec_qwen0_5b_s{seed}"
            if tag == "0_5b":
                continue
            a, b = _load(name), _load(ref)
            pairs[f"{tag}_vs_0_5b_s{seed}"] = {
                ep: cluster_tost(a, b, ep, group_of) for ep in ("pathology", "risk")}
            print(name, {ep: (v["delta_f1"], v["ci90"], v["equivalent_at_margin"])
                         for ep, v in pairs[f"{tag}_vs_0_5b_s{seed}"].items()}, flush=True)
    out = {"margin": MARGIN, "resampling": "patient clusters", "pairs": pairs,
           "dose_response": dose_response(args.prefix, sweep),
           "dose_response_patient_bootstrap": slope_bootstrap(
               [r["name"] for r in sweep], [r["params_b"] for r in sweep], group_of)}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("dose_response", "dose_response_patient_bootstrap")},
                     indent=1))


if __name__ == "__main__":
    main()
