"""Peer-review revision analyses on existing predictions - no training required.

Complements scripts/stats.py (which produced the bootstrap CIs / McNemar tests already
in the paper). This script adds the analyses the reviewers asked for:

  TOST equivalence   (item 10) "interpretability is free" currently accepts the null.
                     Two one-sided paired-bootstrap tests against a pre-specified
                     margin; only a two-sided (1-2a) CI inside +/-margin supports
                     equivalence. Also reports the margin the data CAN support.
  Holm correction    (item 17) family-wise correction over the headline comparisons.
  Class balance      (item 30) reference prevalence for pathology and BI-RADS, so the
                     F1 values are interpretable.
  BI-RADS detail     (item 31) full confusion including 4A vs 4B (the management-
                     relevant boundary) and any 0/1 categories outside the 2-3/4A-6
                     dichotomy.
  Histopathology     (item 6)  accuracy of the `histopathology category` slot, which is
                     generated OUTSIDE the audited concept path.
  Descriptor acc.    (item 37/12 context) per-descriptor agreement.

Outputs outputs/stats/revision.json.

Usage:
    python scripts/stats_revision.py
    python scripts/stats_revision.py --cb_run cb_base3_seed1 --opaque_run x_dinov2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.slots import (  # noqa: E402
    DESCRIPTOR_SLOTS,
    binary_f1,
    extract_slots,
)

random.seed(0)
N_BOOT = 2000
OUT = Path("outputs/stats")
PRED = Path("outputs/runs")

#: Pre-specified equivalence margin for the "interpretability is free" claim,
#: in absolute F1 points (item 10 asks for an explicit margin).
EQUIV_MARGIN = 0.02


def _load(name: str) -> list[dict] | None:
    for p in (PRED / name / "predictions.json", Path("outputs") / name / "predictions.json"):
        if p.exists():
            return json.load(open(p))
    return None


def _aligned(pred_a: list[dict], pred_b: list[dict], endpoint: str):
    """Per-id (ref, pred_a, pred_b) over ids both runs cover, for paired resampling."""
    from src.evaluation.slots import pathology_label, risk_label
    fn = pathology_label if endpoint == "pathology" else risk_label
    a = {d["id"]: d for d in pred_a}
    b = {d["id"]: d for d in pred_b}
    rows = []
    # sorted(): set iteration order over string ids varies with Python's per-process
    # hash seed, which reorders `rows` and so changes the bootstrap draws even under
    # random.seed(0) -- CIs moved in the third decimal between identical invocations.
    for i in sorted(a.keys() & b.keys()):
        rl = fn(extract_slots(a[i]["reference"]))
        if rl is None:
            continue
        pa = fn(extract_slots(a[i]["prediction"]))
        pb = fn(extract_slots(b[i]["prediction"]))
        if pa is None or pb is None:
            continue
        rows.append((rl, pa, pb))
    return rows


def tost(pred_a: list[dict], pred_b: list[dict], endpoint: str,
         margin: float = EQUIV_MARGIN, n: int = N_BOOT) -> dict:
    """Paired-bootstrap TOST. Equivalence is supported only when the whole 90% CI
    (the 1-2*0.05 interval matching two one-sided 5% tests) lies inside +/-margin."""
    rows = _aligned(pred_a, pred_b, endpoint)
    if not rows:
        return {"error": "no aligned samples"}
    m = len(rows)
    obs = binary_f1([(r, x) for r, x, _ in rows]) - binary_f1([(r, y) for r, _, y in rows])
    diffs = []
    for _ in range(n):
        s = [rows[random.randrange(m)] for _ in range(m)]
        diffs.append(binary_f1([(r, x) for r, x, _ in s]) - binary_f1([(r, y) for r, _, y in s]))
    diffs.sort()
    lo90, hi90 = diffs[int(0.05 * n)], diffs[int(0.95 * n)]
    lo95, hi95 = diffs[int(0.025 * n)], diffs[int(0.975 * n)]
    # one-sided bootstrap p-values against the two margin bounds
    p_lower = sum(1 for d in diffs if d <= -margin) / n   # H0: diff <= -margin
    p_upper = sum(1 for d in diffs if d >= margin) / n    # H0: diff >= +margin
    return {
        "n": m,
        "delta_f1": round(obs, 4),
        "margin": margin,
        "ci90": [round(lo90, 4), round(hi90, 4)],
        "ci95": [round(lo95, 4), round(hi95, 4)],
        "p_tost": round(max(p_lower, p_upper), 4),
        "equivalent_at_margin": bool(lo90 > -margin and hi90 < margin),
        # smallest symmetric margin the data would support (the honest number)
        "supported_margin": round(max(abs(lo90), abs(hi90)), 4),
    }


def holm(pvals: dict[str, float]) -> dict[str, dict]:
    """Holm-Bonferroni step-down over a named family of p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(items)
    out: dict[str, dict] = {}
    running = 0.0
    for idx, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (k - idx) * p))
        running = adj
        out[name] = {"p_raw": round(p, 5), "p_holm": round(adj, 5),
                     "sig_holm_005": adj < 0.05}
    return out


def class_balance(pred: list[dict]) -> dict:
    """Reference-label prevalence: F1 at an unstated prevalence is uninterpretable."""
    path = Counter()
    birads = Counter()
    for d in pred:
        s = extract_slots(d["reference"])
        if s.get("pathology"):
            path[s["pathology"].lower()] += 1
        if s.get("birads"):
            birads[s["birads"].upper()] += 1
    n_path = sum(path.values())
    return {
        "n": len(pred),
        "pathology_counts": dict(path),
        "pathology_prevalence_malignant": round(path.get("malignant", 0) / n_path, 4)
        if n_path else None,
        "birads_counts": dict(sorted(birads.items())),
        # categories outside the 2-3 / 4A-6 risk dichotomy (item 31)
        "birads_outside_dichotomy": {k: v for k, v in sorted(birads.items())
                                     if k not in {"2", "3", "4A", "4B", "4C", "5", "6"}},
    }


def birads_detail(pred: list[dict]) -> dict:
    """Full BI-RADS confusion + the 4A-vs-4B sub-problem that drives management."""
    cm: dict[str, Counter] = {}
    exact = tot = 0
    ab_correct = ab_tot = 0
    for d in pred:
        rv = (extract_slots(d["reference"]).get("birads") or "").upper()
        pv = (extract_slots(d["prediction"]).get("birads") or "").upper()
        if not rv or not pv:
            continue
        cm.setdefault(rv, Counter())[pv] += 1
        tot += 1
        exact += int(rv == pv)
        if rv in {"4A", "4B"}:
            ab_tot += 1
            ab_correct += int(rv == pv)
    return {
        "confusion": {k: dict(v) for k, v in sorted(cm.items())},
        "exact_category_accuracy": round(exact / tot, 4) if tot else None,
        "n_scored": tot,
        "birads_4a_4b_accuracy": round(ab_correct / ab_tot, 4) if ab_tot else None,
        "n_4a_4b": ab_tot,
    }


def slot_agreement(pred: list[dict], slot: str) -> dict:
    """Agreement of one non-endpoint slot (e.g. histopathology, margins)."""
    ok = tot = 0
    miss = 0
    for d in pred:
        rv = extract_slots(d["reference"]).get(slot)
        if rv is None:
            continue
        tot += 1
        pv = extract_slots(d["prediction"]).get(slot)
        if pv is None:
            miss += 1
            continue
        ok += int(rv.strip().lower() == pv.strip().lower())
    return {"slot": slot, "n": tot, "accuracy": round(ok / tot, 4) if tot else None,
            "unextractable_predictions": miss}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # These were previously hard-coded, so --runs had no effect and the script
    # silently reported whichever runs were named in the source. That is how a stale
    # 4A-vs-4B figure nearly reached the manuscript; the chosen runs are now echoed
    # into the output so a stale file identifies itself.
    ap.add_argument("--cb_run", default="dinov2_cb",
                    help="concept-bottleneck run for the TOST comparison")
    ap.add_argument("--opaque_run", default="x_dinov2",
                    help="opaque counterpart for the TOST comparison")
    ap.add_argument("--flagship", default=None,
                    help="run used for the descriptive audits (defaults to --cb_run)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="restrict the coverage table to these runs (default: all)")
    ap.add_argument("--output", default="outputs/stats/revision.json")
    args = ap.parse_args()

    flagship_name = args.flagship or args.cb_run
    res: dict = {"equivalence_margin": EQUIV_MARGIN,
                 "runs": {"cb": args.cb_run, "opaque": args.opaque_run,
                          "flagship": flagship_name}}

    # --- item 10: TOST for "interpretability is free" (CB vs opaque) ---
    cb, opaque = _load(args.cb_run), _load(args.opaque_run)
    if cb and opaque:
        res["tost_cb_vs_opaque_dinov2"] = {
            ep: tost(cb, opaque, ep) for ep in ("pathology", "risk")
        }
    else:
        missing = [n for n, d in ((args.cb_run, cb), (args.opaque_run, opaque)) if not d]
        print(f"WARNING: no predictions for {missing}; skipping TOST", file=sys.stderr)

    # --- item 30/31/6: descriptive audits on the flagship run ---
    flagship = _load(flagship_name)
    if not flagship:
        print(f"WARNING: no predictions for flagship {flagship_name!r}; "
              "skipping descriptive audits", file=sys.stderr)
    if flagship:
        res["class_balance"] = class_balance(flagship)
        res["birads_detail"] = birads_detail(flagship)
        res["histopathology_slot"] = slot_agreement(flagship, "histopath")
        res["descriptor_slots"] = {s: slot_agreement(flagship, s) for s in DESCRIPTOR_SLOTS}

    # --- item 26: per-run F1 with CIs already in stats.py; here: coverage of every run ---
    from src.evaluation.slots import coverage
    cov = {}
    names = sorted(p.parent.name for p in PRED.glob("*/predictions.json"))
    if args.runs:
        names = [n for n in names if n in set(args.runs)]
    for name in names:
        p = _load(name)
        if p:
            cov[name] = {ep: round(coverage(p, ep), 4) for ep in ("pathology", "risk")}
    res["coverage_all_runs"] = cov

    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))

    # ---- console summary ----
    if "tost_cb_vs_opaque_dinov2" in res:
        print(f"=== TOST equivalence, {args.cb_run} vs {args.opaque_run}, "
              f"margin +/-{EQUIV_MARGIN} F1 ===")
        for ep, t in res["tost_cb_vs_opaque_dinov2"].items():
            ci = t["ci90"]
            print(f"  {ep:9s} dF1={t['delta_f1']:+.4f} "
                  f"90%CI[{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f" equivalent={t['equivalent_at_margin']}"
                  f" | data supports margin +/-{t['supported_margin']:.3f}")
    if "class_balance" in res:
        b = res["class_balance"]
        print(f"\n=== Class balance (n={b['n']}) ===")
        print(f"  pathology: {b['pathology_counts']} (malignant prevalence "
              f"{b['pathology_prevalence_malignant']:.3f})")
        print(f"  BI-RADS  : {b['birads_counts']}")
        if b["birads_outside_dichotomy"]:
            print(f"  OUTSIDE 2-3/4A-6 dichotomy: {b['birads_outside_dichotomy']}")
    if "birads_detail" in res:
        d = res["birads_detail"]
        print(f"\n=== BI-RADS (n={d['n_scored']}) exact-category acc "
              f"{d['exact_category_accuracy']} | 4A-vs-4B acc {d['birads_4a_4b_accuracy']}"
              f" (n={d['n_4a_4b']}) ===")
    if "histopathology_slot" in res:
        h = res["histopathology_slot"]
        print("\n=== Histopathology slot (generated OUTSIDE the audited path) ===")
        print(f"  accuracy {h['accuracy']} on n={h['n']}")
        for s, v in res["descriptor_slots"].items():
            print(f"  descriptor {s:13s} accuracy {v['accuracy']}")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
