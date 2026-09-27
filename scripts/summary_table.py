"""Aggregate the runs into the paper's result table.

Reads the results CSV (latest row per run), the per-run
predictions.json (for BI-RADS exact accuracy and 4A-vs-4B accuracy, computed
with the shared extractor) and, when present, the stats JSON
(bootstrap CIs and extraction coverage). Applies the pre-specified coverage
guard (D17: coverage < 0.50 on either endpoint = excluded; 0.50-0.90 = flagged)
and reports mean +/- sd over seeds per configuration.

Usage:
    python scripts/summary_table.py --results outputs/runs.csv \
        --stats outputs/stats/statistics.json --json_out outputs/stats/summary_table.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.slots import extract_slots  # noqa: E402

PRED = Path("outputs/runs")

EXCLUDE_BELOW, FLAG_BELOW = 0.50, 0.90

#: run-name pattern (without a prefix) -> (group, display label)
GROUPS = [
    (r"cb3_dinov2_seed(\d)", "A flagship", "DINOv2-L + 3-head concept bottleneck"),
    (r"cb9_dinov2_seed(\d)", "B nine-head", "DINOv2-L + 9-head (finding-level) bottleneck"),
    (r"enc_dinov2_L_seed(\d)", "C encoders", "DINOv2 ViT-L/14 (opaque)"),
    (r"enc_dinov3_L_seed(\d)", "C encoders", "DINOv3 ViT-L/16"),
    (r"enc_in21k_L_seed(\d)", "C encoders", "ImageNet-21k ViT-L/16 (supervised)"),
    (r"enc_eva02_L_seed(\d)", "C encoders", "EVA-02 ViT-L/14 (MIM)"),
    (r"enc_siglip_L_seed(\d)", "C encoders", "SigLIP ViT-L/16 (image-text)"),
    (r"enc_dinov2_B_seed(\d)", "C encoders", "DINOv2 ViT-B/14"),
    (r"enc_raddino_seed(\d)", "C encoders", "RAD-DINO ViT-B/14 (chest X-ray)"),
    (r"enc_usfm_seed(\d)", "C encoders", "USFM ViT-B/16 (ultrasound)"),
    (r"enc_uni2h_seed(\d)", "C encoders", "UNI2-h ViT-H/14 (histopathology)"),
    (r"dec_qwen0_5b_seed(\d)", "D decoders", "Qwen2.5-0.5B"),
    (r"dec_qwen1_5b_seed(\d)", "D decoders", "Qwen2.5-1.5B"),
    (r"dec_qwen3b_seed(\d)", "D decoders", "Qwen2.5-3B"),
    (r"dec_qwen7b_seed(\d)", "D decoders", "Qwen2.5-7B"),
    (r"dec_qwen14b_seed(\d)", "D decoders", "Qwen2.5-14B"),
    (r"dec_qwen32b_seed(\d)", "D decoders", "Qwen2.5-32B"),
    (r"dec_qwen72b_seed(\d)", "D decoders", "Qwen2.5-72B (4-bit)"),
    (r"vlm_qwen25vl_7b_seed\d", "E VLM reference baselines", "Qwen2.5-VL-7B (end-to-end LoRA)"),
    (r"vlm_internvl3_8b_seed\d", "E VLM reference baselines", "InternVL3-8B (end-to-end LoRA)"),
    (r"vlm_qwen2vl_7b_seed\d", "E VLM reference baselines", "Qwen2-VL-7B (end-to-end LoRA)"),
]


def latest_rows(results: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for r in csv.DictReader(open(results)):
        if r["status"] in ("ok", "skip") or r["name"] not in rows:
            rows[r["name"]] = r
    return rows


def birads_accuracies(run: str) -> tuple[float | None, float | None]:
    """Exact BI-RADS-like category accuracy and 4A-vs-4B accuracy over records
    whose reference has a category (unextractable predictions count as wrong)."""
    path = PRED / run / "predictions.json"
    if not path.exists():
        return None, None
    n = exact = n_ab = exact_ab = 0
    for d in json.load(open(path)):
        ref = extract_slots(d["reference"]).get("birads")
        if ref is None:
            continue
        pred = extract_slots(d["prediction"]).get("birads")
        n += 1
        exact += pred == ref
        if ref in ("4A", "4B"):
            n_ab += 1
            exact_ab += pred == ref
    return (exact / n if n else None), (exact_ab / n_ab if n_ab else None)


def mean_sd(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} +/- {statistics.stdev(vals):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="", help="run-name prefix")
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--latex", default=None)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    rows = latest_rows(args.results)
    stats = json.load(open(args.stats))["per_run"] if args.stats.exists() else {}
    table: dict[str, dict[str, dict]] = defaultdict(dict)
    notes: list[str] = []
    for name, r in rows.items():
        if r["status"] not in ("ok", "skip"):
            notes.append(f"{name}: status {r['status']} (not reported)")
            continue
        for pat, group, label in GROUPS:
            m = re.fullmatch(re.escape(args.prefix) + pat, name)
            if not m:
                continue
            st = stats.get(name, {})
            cov = min(st.get("path_f1", {}).get("coverage", 1.0),
                      st.get("risk_f1", {}).get("coverage", 1.0))
            if cov < EXCLUDE_BELOW:
                notes.append(f"{name}: EXCLUDED by the coverage guard (coverage {cov:.2f})")
                continue
            if cov < FLAG_BELOW:
                notes.append(f"{name}: flagged, coverage {cov:.2f}")
            exact, ab = birads_accuracies(name)
            entry = table[group].setdefault(label, {"runs": [], "path": [], "risk": [],
                                                    "exact": [], "ab": [], "bleu": []})
            entry["runs"].append(name)
            entry["path"].append(float(r["path_f1"]))
            entry["risk"].append(float(r["risk_f1"]))
            entry["exact"].append(exact)
            entry["ab"].append(ab)
            entry["bleu"].append(float(r["bleu4"]))
            break

    lines = ["| Group | Configuration | n | Path F1 | Risk F1 | BI-RADS exact | 4A vs 4B "
             "| BLEU-4 |", "|---|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{llccccc}", "\\toprule",
           "Configuration & seeds & Malignancy F1 & Risk-group F1 & Exact category & "
           "4A vs 4B & BLEU-4 \\\\", "\\midrule"]
    for group in sorted(table):
        tex.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{group[2:]}}}}} \\\\")
        for label, e in table[group].items():
            cells = [mean_sd(e["path"]), mean_sd(e["risk"]),
                     mean_sd([x for x in e["exact"] if x is not None]),
                     mean_sd([x for x in e["ab"] if x is not None]), mean_sd(e["bleu"])]
            lines.append(f"| {group} | {label} | {len(e['runs'])} | "
                         + " | ".join(cells) + " |")
            tex.append(f"{label} & {len(e['runs'])} & " + " & ".join(
                c.replace("+/-", "$\\pm$") for c in cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]

    print("\n".join(lines))
    if notes:
        print("\nNotes:\n  " + "\n  ".join(notes))
    if args.latex:
        Path(args.latex).write_text("\n".join(tex) + "\n")
        print(f"\nLaTeX -> {args.latex}")
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps({"table": table, "notes": notes}, indent=2))


if __name__ == "__main__":
    main()
