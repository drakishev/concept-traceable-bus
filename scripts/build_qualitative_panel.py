"""Build a paper-ready qualitative figure + LaTeX table from
scripts/qualitative_examples.py output (panel_selected.json).

Figure: image grid (one column per example) with ground-truth vs. predicted
pathology/risk printed under each, color-coded green/red by correctness.
Table: LaTeX rows with truncated generated vs. reference report text and, for
concept-bottleneck checkpoints, the top predicted concept probabilities.

Usage:
    python scripts/build_qualitative_panel.py \
        --panel outputs/qualitative/h_cb9_dinov2_s2/panel_selected.json \
        --out_fig paper/frontiers/revision2/supp_fig_qualitative.png \
        --out_tex paper/frontiers/revision2/tables/tab_qualitative.tex \
        --model_name "CB-9 (seed 2)"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import slot_labels as sl  # noqa: E402


def _names(classes: dict[str, int], surface: dict[str, tuple] | None = None) -> list[str]:
    """Class index -> readable name; enum classes use their first report synonym."""
    by_key = {k.lower(): v[0] for k, v in (surface or {}).items()}
    return [by_key.get(k, k) for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]


CONCEPT_CLASS_NAMES = {
    "pathology": _names(sl.PATHOLOGY_CLASSES),
    "risk": _names(sl.RISK_CLASSES),
    "birads": _names(sl.BIRADS_CLASSES),
    "margins": _names(sl.MARGINS_CLASSES, sl.EDGE_MAP),
    "shape": _names(sl.SHAPE_CLASSES),
    "orientation": _names(sl.ORIENTATION_CLASSES),
    "echogenicity": _names(sl.ECHO_CLASSES, sl.ECHO_MAP),
    "boundary": _names(sl.BOUNDARY_CLASSES, sl.BOUNDARY_MAP),
    "calcification": _names(sl.CALCIFICATION_CLASSES, sl.CALC_MAP),
}
HEAD_LABELS = {"pathology": "Path", "risk": "Risk", "birads": "BI-RADS-like",
               "margins": "Margins", "shape": "Shape", "orientation": "Orientation",
               "echogenicity": "Echo", "boundary": "Boundary", "calcification": "Calc."}
#: Frontiers forbids red/green indicator pairs; the manuscript's validated pair.
C_OK, C_BAD = "#4C78A8", "#F58518"
_TAG_RE = re.compile(r"</?reasoning>|<answer>\s*\d\s*</answer>")


def plain(report: str) -> str:
    """Drop the <reasoning>/<answer> markup of the BUS-CoT template for display."""
    return " ".join(_TAG_RE.sub(" ", report).split())


def top_concept(concept_probs: dict | None, head: str) -> str:
    if not concept_probs or head not in concept_probs:
        return "--"
    probs = concept_probs[head]
    names = CONCEPT_CLASS_NAMES[head]
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return f"{names[idx]} ({probs[idx]*100:.0f}\\%)"


def esc(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_")
             .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out_fig", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--max_chars", type=int, default=340)
    ap.add_argument("--model_name", default="the DINOv2 + concept-bottleneck model")
    ap.add_argument("--table_indices", default="0,3,6",
                    help="Comma-separated indices into the panel to include in the LaTeX table "
                         "(the figure shows the full panel; the table shows one representative "
                         "example per correctness bucket by default: both/partial/miss).")
    args = ap.parse_args()

    panel = json.load(open(args.panel))
    table_idx = [int(i) for i in args.table_indices.split(",") if i.strip() != ""]
    table_panel = [panel[i] for i in table_idx if i < len(panel)]

    # ── Figure: image grid with correctness-coded captions ─────────────────────
    n = len(panel)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 5.0 * nrows), dpi=300)
    axes = axes.flatten() if n > 1 else [axes]

    # Reserve fixed vertical room below each image for the title + 2 caption
    # lines so rows never collide, regardless of tight_layout's own estimate.
    for ax in axes:
        ax.set_anchor("N")

    for ax, ex in zip(axes, panel):
        if ex.get("panel_image") and Path(ex["panel_image"]).exists():
            img = Image.open(ex["panel_image"])
            ax.imshow(img, cmap="gray")
        ax.axis("off")
        path_color = C_OK if ex["path_correct"] else C_BAD
        risk_color = C_OK if ex["risk_correct"] else C_BAD
        meta = ex.get("metadata", {})
        gt_path = meta.get("pathology", "?")
        ax.set_title(f"GT: {gt_path}", fontsize=10, pad=8)
        mark_path = "correct" if ex["path_correct"] else "wrong"
        mark_risk = "correct" if ex["risk_correct"] else "wrong"
        ax.text(0.5, -0.10, f"pathology: {mark_path}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=path_color)
        ax.text(0.5, -0.19, f"risk: {mark_risk}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=risk_color)

    for ax in axes[len(panel):]:
        ax.axis("off")

    fig.suptitle(f"Qualitative test-set examples ({args.model_name})", fontsize=12, y=0.995)
    fig.subplots_adjust(hspace=0.55, wspace=0.1, top=0.94)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {args.out_fig}")

    # ── LaTeX table: id, top concepts, truncated pred/ref ──────────────────────
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering\scriptsize")
    lines.append(r"\caption{Representative test-set examples from " + esc(args.model_name)
                 + r", one per correctness bucket (Supplementary Figure~S2 shows the full "
                 r"nine-example panel). The predicted concepts (top class and probability of each "
                 r"head) are the decoder's only conditioning signal; \checkmark/$\times$\ mark "
                 r"slot-extracted correctness against the reference. Reports are shown verbatim "
                 r"apart from the template's reasoning/answer tags, including a missing space "
                 r"(``hypoechoicwith'') that is present in the BUS-CoT source reports.}")
    lines.append(r"\label{tab:qualitative}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    # ragged-right cells: justified text in narrow columns stretches word gaps (needs array)
    rr = r">{\raggedright\arraybackslash}"
    lines.append(r"\begin{tabular}{" + rr + r"p{0.24\textwidth}" + rr + r"p{0.34\textwidth}"
                 + rr + r"p{0.34\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"Predicted concepts (decoder's only input) & Generated report "
                 r"& Reference report \\")
    lines.append(r"\midrule")
    for ex in table_panel:
        cp = ex.get("concept_probs")
        path_mark = r"\checkmark" if ex["path_correct"] else r"$\times$"
        risk_mark = r"\checkmark" if ex["risk_correct"] else r"$\times$"
        marks = {"pathology": f" {path_mark}", "risk": f" {risk_mark}"}
        concepts = "; ".join(f"{HEAD_LABELS[h]}: {top_concept(cp, h)}{marks.get(h, '')}"
                             for h in CONCEPT_CLASS_NAMES if cp and h in cp)
        pred = esc(textwrap.shorten(plain(ex["prediction"]), width=args.max_chars,
                                    placeholder="..."))
        ref = esc(textwrap.shorten(plain(ex["reference"]), width=args.max_chars,
                                   placeholder="..."))
        lines.append(f"{concepts} & {pred} & {ref} \\\\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    Path(args.out_tex).write_text("\n".join(lines))
    print(f"Saved table -> {args.out_tex}")


if __name__ == "__main__":
    main()
