"""Collect every number the revised manuscript cites into one JSON (revision 2).

The manuscript, supplement and response letter quote from this file only, so
each number can be traced to the artifact it was computed from (the
`source` next to each block). Nothing here trains or evaluates a model.

Pre-specified comparison family (Holm within each seed, two endpoints each):
DINOv2-L vs UNI2-h, DINOv2-B vs USFM, CB-3 vs opaque, CB-9 vs opaque. The p-value
is McNemar on one record per patient, so it respects the patient clustering; CIs
are paired patient-cluster bootstrap intervals of the F1 difference.

Usage:
    python scripts/revision2_numbers.py --out outputs/stats/revision2_numbers.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

A7 = Path("outputs/analysis7")
STATS = Path("outputs/stats")
SEEDS = (1, 2, 3)
FAMILY = ("DINOv2vsUNI2h", "DINOv2BvsUSFM", "CB3vsOpaque", "CB9vsOpaque")


def _j(path: Path) -> dict:
    return json.loads(path.read_text())


def _msd(v: list[float]) -> dict:
    # Full precision: rounding here and again at display time double-rounds
    # (0.75850031 -> 0.7585 -> "0.758" instead of "0.759").
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else None,
            "min": min(v), "max": max(v), "per_seed": list(v)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = round(running, 6)
    return out


def data_block() -> dict:
    def load(p):
        return [json.loads(line) for line in open(p)]
    train, val = load("data/unified_v5/train.jsonl"), load("data/unified_v5/val.jsonl")
    test = load("data/unified_v5/test_buscot_only.jsonl")
    aug = sum(1 for _ in open("data/augmented_v5/train.jsonl"))
    path = Counter(r["metadata"]["pathology"] for r in test)
    from src.evaluation.slots import extract_slots
    birads = Counter(extract_slots(r["conversations"][1]["content"]).get("birads") or "none"
                     for r in test)
    leak = _j(STATS / "leakage_v5.json")
    return {
        "source": "data/unified_v5/*.jsonl, outputs/stats/leakage_v5.json",
        "n_train": len(train), "n_train_augmented": aug, "n_val": len(val), "n_test": len(test),
        "groups": leak["n_studies"],
        "test_pathology": dict(path),
        "test_malignant_prevalence": round(path["malignant"] / len(test), 4),
        "test_birads": dict(birads),
        "origin_datasets_test": dict(Counter(r["metadata"].get("origin_dataset") for r in test)),
        "hash_audit": {k: v["n"] for k, v in leak["hash"]["splits"].items()},
        "external_in_train_val": {k: {"n": v["n"], "in_train_val": v["in_train_val"]}
                                  for k, v in leak["hash"]["external"].items()},
    }


def _test_ids(version: str) -> set[str]:
    """Official-test record ids of an earlier split: from its data directory, or from the
    id list shipped in the public repository's splits/ folder."""
    data = Path(f"data/unified_{version}/test_buscot_only.jsonl")
    if data.exists():
        return {json.loads(line)["id"] for line in open(data)}
    return set(Path(f"splits/test_ids_{version}.txt").read_text().split())


def leakage_history() -> dict:
    """Near-duplicate-frame leakage of the two earlier splits, for the response letter:
    v2 = the submitted image-level split, v4g = the batch-6 grouped split."""
    from sklearn.metrics import roc_auc_score
    out = {"source": "outputs/stats/leakage_{v2,v4g}_hash.json (scripts/leakage_audit.py --hash); "
                     "outputs/stats/probs_u2bench.json (submitted checkpoint)"}
    for v in ("v2", "v4g"):
        h = _j(STATS / f"leakage_{v}_hash.json")["hash"]
        ids = _test_ids(v)
        leak = set(h["splits"]["test_in_train"]["ids"]) | set(h["splits"]["test_in_val"]["ids"])
        out[v] = {"n_test_buscot": len(ids), "test_frames_in_train_or_val": len(ids & leak),
                  "u2bench_frames_in_train_or_val": h["external"]["u2bench"]["in_train_val"]}
    probs = _j(STATS / "probs_u2bench.json")
    rec = probs[list(probs)[-1]]
    bad = set(_j(STATS / "leakage_v2_hash.json")["hash"]["external"]["u2bench"]["contaminated_ids"])

    def auc(rows: list[dict]) -> float:
        return roc_auc_score([r["y"] for r in rows], [r["p_malignant"] for r in rows])
    clean = [r for r in rec if r["id"] not in bad]
    dirty = [r for r in rec if r["id"] in bad]
    out["submitted_u2bench_auroc"] = {"all": auc(rec), "n_all": len(rec), "clean": auc(clean),
                                      "n_clean": len(clean), "contaminated": auc(dirty),
                                      "n_contaminated": len(dirty)}
    return out


def excluded_seed() -> dict:
    """R3.5: the nine-head seed excluded in the submitted version (earlier split), scored as
    submitted (outputs/weekend_results.csv, template-A-only extractor) and re-extracted with
    the current extractor, beside its two retained sibling seeds."""
    from src.evaluation.slots import binary_f1, coverage, paired_labels
    submitted = {r["name"]: r for r in csv.DictReader(open("outputs/weekend_results.csv"))
                 if r["status"] == "ok"}
    out: dict = {"source": "outputs/weekend/cb_desc9_s{1,2,3}/predictions.json"}
    for s in (1, 2, 3):
        name = f"cb_desc9_s{s}"
        pred = _j(Path("outputs/weekend") / name / "predictions.json")
        out[name] = {"submitted": {"path_f1": float(submitted[name]["path_f1"]),
                                   "risk_f1": float(submitted[name]["risk_f1"])},
                     "current_extractor": {ep: {"f1": binary_f1(paired_labels(pred, ep)),
                                                "coverage": coverage(pred, ep)}
                                           for ep in ("pathology", "risk")}}
    return out


def _constant(golds: list[str], pathology: list[str] | None = None) -> dict:
    """Scores of an image-blind report that always states the same answer: malignant, high
    risk, the most frequent category, and the more frequent of 4A and 4B on the 4A/4B
    references (the endpoints' own definitions; F1 of the positive class)."""
    from src.data.slot_labels import LOW_RISK_BIRADS
    cats = Counter(g for g in golds if g)
    n_high = sum(v for k, v in cats.items() if k not in LOW_RISK_BIRADS)
    ab = {k: cats.get(k, 0) for k in ("4A", "4B")}
    out = {"risk_f1_always_high": 2 * n_high / (sum(cats.values()) + n_high),
           "exact_most_frequent": max(cats.values()) / sum(cats.values()),
           "most_frequent_category": max(cats, key=cats.get),
           "ab_more_frequent": max(ab.values()) / sum(ab.values()),
           "ab_more_frequent_category": max(ab, key=ab.get)}
    if pathology is not None:
        n_mal = sum(x == "malignant" for x in pathology)
        out["path_f1_always_malignant"] = 2 * n_mal / (len(pathology) + n_mal)
    return out


def constant_baselines() -> dict:
    from src.evaluation.slots import extract_slots
    test = [json.loads(line) for line in open("data/unified_v5/test_buscot_only.jsonl")]
    golds = [extract_slots(r["conversations"][1]["content"]).get("birads") for r in test]
    u2b = _j(A7 / "u2b_birads" / "cb3_s1" / "predictions_birads.json")
    return {"source": "data/unified_v5/test_buscot_only.jsonl; outputs/analysis7/u2b_birads "
                      "(gold categories, identical for every model)",
            "internal": _constant(golds, [r["metadata"]["pathology"] for r in test]),
            "u2bench_birads": _constant([p["gold"] for p in u2b])}


def submitted_encoder_check() -> dict:
    """The submitted version's DINOv2 vs UNI2-h comparison (earlier split, 440 test records),
    re-evaluated from its two checkpoints (scripts/evaluate_jepa.py): with resize only, which
    reproduces the stored predictions, and with the training preprocessing."""
    from src.evaluation.slots import binary_f1, paired_labels
    root = Path("outputs/submitted_preprocessing_check")
    stored = {"dinov2": "outputs/weekend/x_dinov2/predictions.json",
              "uni2h": "outputs/eval_jepa_v2_mt_buscot/predictions.json"}
    runs = {"dinov2": "x_dinov2", "uni2h": "v2_mt_uni2h"}
    out: dict = {"source": f"{root}/<run>_<basic|ultrasound>/predictions.json"}
    for m in ("basic", "ultrasound"):
        f1 = {}
        for enc, run in runs.items():
            pred = _j(root / f"{run}_{m}" / "predictions.json")
            f1[enc] = {ep: binary_f1(paired_labels(pred, ep)) for ep in ("pathology", "risk")}
            if m == "basic":
                old = {d["id"]: d["prediction"] for d in _j(Path(stored[enc]))}
                f1[enc]["reproduces_stored"] = all(old[d["id"]] == d["prediction"] for d in pred)
        out[m] = {**f1, "delta": {ep: f1["dinov2"][ep] - f1["uni2h"][ep]
                                  for ep in ("pathology", "risk")}}
    return out


def label_consistency() -> dict:
    """How often BUS-CoT's structured margin annotation (LesionEdge, the CB-9 margins label)
    agrees with the margin word in the same record's report text."""
    from src.data.slot_labels import _struct_lookup, lesion_key
    from src.evaluation.slots import extract_slots
    struct = _struct_lookup()
    agree = total = 0
    for split in ("train", "val", "test_buscot_only"):
        for line in open(f"data/unified_v5/{split}.jsonl"):
            r = json.loads(line)
            edge = (struct.get(lesion_key(r["image_path"]) or "", {}).get("us_report", {})
                    .get("LesionEdge"))
            text = extract_slots(r["conversations"][1]["content"]).get("margins")
            if edge and text:
                total += 1
                agree += edge.lower() == text
    return {"source": "BUS-CoT lesion_dataset.json vs report text, data/unified_v5",
            "margin_enum_vs_text_agreement": agree / total, "n": total}


#: BrEaST descriptor entries of descriptors.json (scripts/evaluate_breast.py)
BREAST_DESCRIPTORS = {"shape": "shape", "margin": "margin_circumscribed",
                      "echogenicity": "echogenicity", "calcification": "calcification"}


def _gold_counts(entry: dict) -> dict[str, int]:
    return entry.get("gold_distribution") or {g: sum(r.values())
                                              for g, r in entry["confusion"].items()}


def balanced_accuracy(entry: dict) -> float:
    """Mean per-class recall over the gold classes; a missing prediction counts as wrong."""
    gold = _gold_counts(entry)
    return st.mean(entry["confusion"].get(g, {}).get(g, 0) / n for g, n in gold.items() if n)


def breast_baselines(d: dict) -> dict:
    """What a report that ignores the image would score on BrEaST: always the most frequent
    annotated class (accuracy), chance balanced accuracy, and always "malignant" (F1)."""
    out = {}
    for name, key in {**BREAST_DESCRIPTORS, "birads_exact": "birads"}.items():
        gold = _gold_counts(d[key])
        out[name] = {"majority_accuracy": max(gold.values()) / sum(gold.values()),
                     "majority_class": max(gold, key=gold.get),
                     "chance_balanced_accuracy": 1 / len(gold), "n_classes": len(gold)}
    c = d["pathology"]["confusion"]
    pos, n = c["tp"] + c["fn"], sum(c.values())
    out["path_f1_always_malignant"] = 2 * pos / (n + pos)
    return out


def main_table() -> dict:
    t = _j(STATS / "batch7_table.json")["table"]
    out = {}
    for group, rows in t.items():
        for label, e in rows.items():
            out[label] = {"group": group, "runs": e["runs"],
                          **{k: _msd([x for x in e[k] if x is not None])
                             for k in ("path", "risk", "exact", "ab", "bleu")}}
    return {"source": "outputs/stats/batch7_table.json (scripts/batch_table.py)", "rows": out}


def comparisons() -> dict:
    s = _j(STATS / "stats_batch7.json")
    comp, fam = {}, {}
    for name, v in s["comparisons"].items():
        row = {}
        for ep in ("pathology", "risk"):
            b = v[ep]["paired_bootstrap_cluster"]
            row[ep] = {"delta": b["delta_f1"], "ci95_cluster": list(b["ci95"]),
                       "sig_cluster": b["significant"],
                       "mcnemar_p": round(v[ep]["mcnemar"]["p_value"], 6),
                       "mcnemar_p_one_per_patient": round(
                           v[ep]["mcnemar_one_per_group"]["p_value"], 6)}
        comp[name] = row
    for seed in SEEDS:
        p = {f"{c}_{ep}": comp[f"{c}_s{seed}"][ep]["mcnemar_p_one_per_patient"]
             for c in FAMILY for ep in ("pathology", "risk")}
        fam[f"s{seed}"] = holm(p)
    cov = [min(r["path_f1"]["coverage"], r["risk_f1"]["coverage"]) for r in s["per_run"].values()]
    return {"source": "outputs/stats/stats_batch7.json (scripts/stats.py, patient clusters)",
            "pairs": comp, "holm_family": {"members": list(FAMILY), "adjusted_p": fam},
            "min_extraction_coverage": min(cov), "n_runs": len(cov)}


def decoder() -> dict:
    t = _j(STATS / "tost_decoder_scale_batch7.json")
    pairs = t["pairs"]
    n_eq = sum(v[ep]["equivalent_at_margin"] for v in pairs.values() for ep in v)
    n_tot = sum(len(v) for v in pairs.values())
    sup = [v[ep]["supported_margin"] for v in pairs.values() for ep in v]
    return {"source": "outputs/stats/tost_decoder_scale_batch7.json (scripts/decoder_tost.py)",
            "dose_response": t["dose_response"],
            "dose_response_patient_bootstrap": t["dose_response_patient_bootstrap"],
            "tost_margin": t["margin"],
            "tost_equivalent": f"{n_eq}/{n_tot}", "supported_margin_range": [min(sup), max(sup)],
            "pairs": pairs}


def per_model_analyses() -> dict:
    from sklearn.metrics import roc_auc_score
    out = {}
    for m in ("cb3", "cb9"):
        auc, ece, op, arg, u2b, u2bb, br, iv, cv = ({} for _ in range(9))
        for s in SEEDS:
            t = f"{m}_s{s}"
            for key in ("test", "u2bench", "breast"):
                d = _j(A7 / "probs" / t / f"{key}.json")
                rec = d[list(d)[-1]]
                auc.setdefault(key, []).append(
                    roc_auc_score([r["y"] for r in rec], [r["p_malignant"] for r in rec]))
            cal = _j(A7 / "probs" / t / "calibration.json")
            for key, d in (("test", cal["internal_test"]), ("u2bench", cal["external"]["u2bench"]),
                           ("breast", cal["external"]["breast"])):
                ece.setdefault(key, []).append(d["reliability"]["ece"])
                o = d["sensitivity_0.90_threshold_from_val"]
                op.setdefault(key, {}).setdefault("sens", []).append(o["sensitivity"])
                op[key].setdefault("spec", []).append(o["specificity"])
                a = d["argmax_threshold"]
                arg.setdefault(key, {}).setdefault("sens", []).append(a["sensitivity"])
                arg[key].setdefault("spec", []).append(a["specificity"])
            mal = _j(A7 / "u2b_malig" / t / "metrics.json")
            for k in ("malignancy_f1", "sensitivity", "specificity"):
                u2b.setdefault(k, []).append(mal[k])
            bi = _j(A7 / "u2b_birads" / t / "metrics_birads.json")
            for src in ("concept_head", "generated_report"):
                b = bi[src]
                for k, v in (("exact", b["exact_accuracy"]), ("adjacent", b["adjacent_accuracy"]),
                             ("risk_f1", b["risk_group"]["f1"]),
                             ("risk_balanced_accuracy", b["risk_group"]["balanced_accuracy"]),
                             ("ab", b["acc_4a_vs_4b"]),
                             ("coverage", b["coverage"])):
                    u2bb.setdefault(src, {}).setdefault(k, []).append(v)
            pred = {p["id"]: p for p in json.load(open(A7 / "u2b_birads" / t /
                                                        "predictions_birads.json"))}
            agree = [p["head"] == p["report_birads"] for p in pred.values() if p["report_birads"]]
            u2bb.setdefault("head_report_agreement", []).append(sum(agree) / len(agree))
            d = _j(A7 / "breast" / t / "descriptors.json")
            for k, v in (("path_f1", d["pathology"]["f1"]),
                         ("path_sens", d["pathology"]["sensitivity"]),
                         ("path_spec", d["pathology"]["specificity"]),
                         ("birads_exact", d["birads"]["exact_accuracy"]),
                         ("birads_ab", d["birads"]["acc_4a_vs_4b"]),
                         ("shape", d["shape"]["accuracy_missing_wrong"]),
                         ("margin", d["margin_circumscribed"]["accuracy_missing_wrong"]),
                         ("echogenicity", d["echogenicity"]["accuracy_missing_wrong"]),
                         ("calcification", d["calcification"]["accuracy_missing_wrong"]),
                         ("birads_balanced", balanced_accuracy(d["birads"])),
                         *((f"{k}_balanced", balanced_accuracy(d[e]))
                           for k, e in BREAST_DESCRIPTORS.items())):
                br.setdefault(k, []).append(v)
            for h, v in _j(A7 / "intervention" / f"{t}.json")["agreement"].items():
                if v is not None:
                    iv.setdefault(h, []).append(v)
            c = _j(A7 / "covariation" / f"{t}.json")
            for k, v in (("adopt", c["flip"]["forced_label_adopted_rate"]),
                         ("flip_change", c["flip"]["any_descriptor_changed_rate"]),
                         ("control_change", c["control"]["any_descriptor_changed_rate"])):
                cv.setdefault(k, []).append(v)
        conv = {k: _msd(v) for k, v in auc.items()}
        out[m] = {
            "auroc": conv, "ece": {k: _msd(v) for k, v in ece.items()},
            "op90": {k: {kk: _msd(vv) for kk, vv in v.items()} for k, v in op.items()},
            "argmax": {k: {kk: _msd(vv) for kk, vv in v.items()} for k, v in arg.items()},
            "u2b_malignancy_argmax": {k: _msd(v) for k, v in u2b.items()},
            "u2b_birads": {k: ({kk: _msd(vv) for kk, vv in v.items()} if isinstance(v, dict)
                               else _msd(v)) for k, v in u2bb.items()},
            "breast": {k: _msd(v) for k, v in br.items()},
            "intervention": {k: _msd(v) for k, v in iv.items()},
            "covariation": {k: _msd(v) for k, v in cv.items()},
        }
    baselines = breast_baselines(_j(A7 / "breast" / "cb3_s1" / "descriptors.json"))
    probe = _j(A7 / "linear_probe" / "cb3_s1.json")["endpoints"]
    gap = _j(STATS / "modality_gap_batch7.json")
    gap = gap if isinstance(gap, dict) else {}
    keys = ("linear_cka", "i2t_R@1", "t2i_R@1", "i2t_median_rank", "t2i_median_rank",
            "chance_median_rank")
    return {"source": "outputs/analysis7/ (scripts/run_batch_analyses.sh)", "models": out,
            "breast_baselines": baselines,
            "linear_probe": {ep: {"f1": v["f1"], "ci95": v["f1_ci95"]} for ep, v in probe.items()},
            "modality_gap": {k: _msd([r[k] for r in gap.values() if isinstance(r, dict)])
                             for k in keys}}


def vlm_bleu() -> dict:
    # latest row per run: a regenerated run is appended, its earlier row stays in the CSV
    latest = {r["name"]: r for r in csv.DictReader(open("outputs/weekend7_results.csv"))
              if r["status"] == "ok"}
    ours = [r for n, r in latest.items() if not n.startswith("h_vlm")]
    dec = [r for n, r in latest.items() if n.startswith("h_dec_")]

    def rng(rows: list[dict], k: str) -> list[float]:
        return [round(min(float(r[k]) for r in rows), 3), round(max(float(r[k]) for r in rows), 3)]
    return {"source": "outputs/weekend7_results.csv (latest row per run)",
            "bleu4_range_all_runs": rng(ours, "bleu4"),
            "path_f1_range_all_runs": rng(ours, "path_f1"),
            "bleu4_range_decoder_sweep": rng(dec, "bleu4"),
            "path_f1_range_decoder_sweep": rng(dec, "path_f1")}


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/stats/revision2_numbers.json")
    args = ap.parse_args()
    from u2bench_labels import require_labels
    require_labels()
    ext = _j(STATS / "extractor_validation_v5.json")
    out = {"data": data_block(), "leakage_history": leakage_history(),
           "main_table": main_table(), "comparisons": comparisons(),
           "label_consistency": label_consistency(), "excluded_seed": excluded_seed(),
           "submitted_encoder_check": submitted_encoder_check(),
           "constant_baselines": constant_baselines(),
           "decoder": decoder(), "analyses": per_model_analyses(), "bleu": vlm_bleu(),
           "extractor_validation": {"source": "outputs/stats/extractor_validation_v5.json",
                                    "summary": ext.get("summary", ext)}}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
