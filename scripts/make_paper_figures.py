"""Regenerate every result figure at Frontiers print specification.

Frontiers requires: 300 dpi at final size, RGB, width 85 mm (one column) or 180 mm
(two columns), smallest text >= 8 pt, lines >= 2 pt wide, panel labels (A)/(B) outside
the image, no information carried by colour alone, and no red/green indicator pairs.
It also asks authors NOT to draw figures with LaTeX, so the architecture schematic is
rendered here rather than in TikZ.

Every value is read from the analysis artifacts (outputs/stats/*.json,
outputs/submitted_version/runs.csv, outputs/**/predictions.json) so no number is hand-copied
into a figure.

Palette: blue/orange/purple/teal/yellow, no green. Adjacent-pair separation was
validated in OKLab under deuteranopia/protanopia/tritanopia simulation; the
blue+orange pair used for the two clinical endpoints separates by dE > 22 under all
three. Green was removed because orange-green failed protanopia (dE 3.5).

Usage:
    python scripts/make_paper_figures.py --outdir paper/frontiers/submission
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Frontiers geometry / typography ────────────────────────────────────────
MM = 1 / 25.4
W1, W2 = 85 * MM, 180 * MM          # single- and double-column widths (inches)
# 400, not 300: every figure is saved with bbox_inches="tight", which trims the
# surrounding whitespace and therefore yields FEWER pixels than the nominal
# figsize x dpi. Included at (a fraction of) \linewidth, a 300-dpi save landed at
# 264-286 effective dpi -- under Frontiers' "300 dpi at final size" floor. Saving
# at 400 puts every figure at 350+ effective dpi with headroom; it does not change
# any font size, since point sizes are defined against the figure's inch geometry.
DPI = 400
C_PATH, C_RISK = "#4C78A8", "#F58518"      # validated pair for the two endpoints
C_ALT, C_ALT2 = "#B279A2", "#72B7B2"
C_MUTED, INK, GRID = "#8C8C8C", "#222222", "#D9D9D9"

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    # Bumped from 8/9pt: at full \linewidth inclusion, 8pt design text renders
    # right at Frontiers' legibility floor and reads as small next to ~10-11pt
    # body text. 10/12pt gives clear headroom while staying well inside limits.
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 0.8, "lines.linewidth": 2.0,      # >= 2 pt data lines
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _r3(x: float) -> str:
    """Round to 3 dp half-UP, matching the manuscript text.

    Python's format() rounds half-to-even against the binary value, so the external
    AUC of 0.8965 renders as "0.896" while every mention in the text says 0.897.
    Decimal with ROUND_HALF_UP removes that figure-versus-text discrepancy.
    """
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))


def _load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _csv_rows() -> dict[str, dict]:
    """Latest OK row per run name from the master results CSV."""
    out: dict[str, dict] = {}
    with open("outputs/submitted_version/runs.csv") as f:
        for r in csv.DictReader(f):
            if r.get("status", "").startswith("ok") and r.get("path_f1"):
                out[r["name"]] = r
    return out


def _f1(pred_dir: str, endpoint: str) -> float | None:
    from src.evaluation.slots import binary_f1, paired_labels
    p = Path(pred_dir) / "predictions.json"
    if not p.exists():
        return None
    return binary_f1(paired_labels(json.load(open(p)), endpoint))


# ── Figure 1: architecture ────────────────────────────────────────────────
def fig_architecture(outdir: Path,
                     concept_note: str = "pathology / BI-RADS risk /\nBI-RADS category") -> None:
    fig, ax = plt.subplots(figsize=(W2, 0.42 * W2))
    # Boxes are drawn with boxstyle "round,pad=0.008", which expands each patch
    # 0.008 data-units OUTWARD from its nominal rectangle. The leftmost box sits
    # at x=0.005, so its padded edge lands at -0.003 and was being clipped flat
    # by an xlim of exactly (0, 1) -- the left rounded corners and outline of the
    # "Breast US image" box went missing. Give the axes a margin wider than the
    # pad so every patch outline falls inside the drawing area.
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, label, fc, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=1.0, edgecolor=INK, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=INK, linespacing=1.35)

    def arrow(x0, y0, x1, y1, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=9, linewidth=1.4, color=INK, linestyle=ls))

    # Layout note. Each box's drawn outline sits PAD beyond its nominal rectangle
    # (boxstyle pad), so the visible gap between two boxes is (nominal gap - 2*PAD)
    # and an arrow drawn edge-to-edge on nominal coordinates starts and ends
    # *underneath* the two outlines. Previously the s-hat -> concept-bottleneck
    # nominal gap was 0.0175, i.e. only 0.0015 of visible clearance, and every
    # arrowhead touched the box it pointed at. Boxes are therefore placed on a
    # uniform GAP and arrows are inset by PAD + CLEAR at both ends.
    PAD, CLEAR, GAP = 0.008, 0.006, 0.050
    yt, h = 0.60, 0.20

    def chain(specs, y, height):
        """Place boxes left-to-right on a uniform gap; return their spans."""
        spans, x = [], specs[0][0]
        for i, (_, w, label, fc) in enumerate(specs):
            box(x, y, w, height, label, fc)
            spans.append((x, x + w))
            x += w + GAP
        return spans

    # s-hat (top) and s (bottom) are centred on the same x so the vertical
    # InfoNCE arrow meets both squarely.
    top = [(0.000, 0.108, "Breast US\nimage $x$", "#EDEDED"),
           (None, 0.124, "X-encoder $f_X$\n(frozen)", "#DCE6F1"),
           (None, 0.096, "Predictor\n$f_P$", "#FCE3CC"),
           (None, 0.065, "$\\hat{s}$", "#FFFFFF"),
           (None, 0.140, "Concept\nbottleneck", "#FCE3CC"),
           (None, 0.095, "Y-decoder\n$f_D$", "#FCE3CC"),
           (None, 0.072, "Report\n$\\hat{t}$", "#EDEDED")]
    spans = chain(top, yt, h)
    for (_, x_end), (x_next, _) in zip(spans[:-1], spans[1:]):
        arrow(x_end + PAD + CLEAR, yt + h / 2, x_next - PAD - CLEAR, yt + h / 2)

    shat_c = sum(spans[3]) / 2
    concept_c = sum(spans[4]) / 2
    ax.text(concept_c, yt - 0.055, concept_note,
            ha="center", va="top", fontsize=7, color=C_MUTED, linespacing=1.3)

    yb, hb = 0.16, h * 0.8
    s_w = 0.075
    bot = [(shat_c - s_w / 2, s_w, "$s$", "#FFFFFF"),
           (None, 0.150, "Y-encoder $f_Y$\n(0.05x LR)", "#DCE6F1"),
           (None, 0.140, "Report text $t$", "#EDEDED")]
    bspans = chain(bot, yb, hb)
    # text path flows right-to-left: report text -> Y-encoder -> s
    for (x_start, _), (_, x_prev_end) in zip(bspans[1:], bspans[:-1]):
        arrow(x_start - PAD - CLEAR, yb + hb / 2, x_prev_end + PAD + CLEAR, yb + hb / 2)
    arrow(shat_c, yb + hb + PAD + CLEAR, shat_c, yt - PAD - CLEAR,
          style="<|-|>", ls="--")
    ax.text(shat_c - 0.010, (yb + hb + yt) / 2, "InfoNCE  ", ha="right",
            va="center", fontsize=7.5, color=INK)

    ax.text(0.000, 0.94, "Stage 1: cross-modal alignment", fontsize=8,
            color="#2F5597", style="italic")
    ax.text(concept_c - 0.075, 0.94, "Stage 2: generation (alignment path frozen)",
            fontsize=8, color="#B26100", style="italic")
    ax.text(0.005, 0.02, "Blue = frozen   Orange = trainable", fontsize=7, color=C_MUTED)
    _save(fig, outdir, "fig_architecture")


# ── Figure 2: decoder scale, with the confounds shown ─────────────────────
def _seed_stats(prefix: str) -> tuple[float, float, float, float, int] | None:
    """(path_mean, path_sd, risk_mean, risk_sd, n) over all seeds of a run prefix."""
    import statistics as st
    rows = _csv_rows()
    vals = [(float(r["path_f1"]), float(r["risk_f1"]))
            for name, r in rows.items() if name.startswith(prefix + "_seed")]
    if not vals:
        return None
    p = [v[0] for v in vals]
    k = [v[1] for v in vals]
    return (st.mean(p), st.stdev(p) if len(p) > 1 else 0.0,
            st.mean(k), st.stdev(k) if len(k) > 1 else 0.0, len(p))


def fig_decoder_scale(outdir: Path) -> None:
    """Condition-matched decoder sweep: only decoder size varies (batch 5).

    Replaces the earlier 3-point plot, whose points differed in text encoder,
    data version, preprocessing and adaptation method as well as size.
    """
    sizes = [("2.7B", "dec_m_biomedlm2b"), ("4B", "dec_m_medgemma4b"),
             ("7B", "dec_m_qwen7b"), ("8B", "dec_m_openbiollm8b"),
             ("27B", "dec_m_medgemma27b"), ("70B", "dec_m_openbiollm70b")]
    pts = [(lbl, _seed_stats(pre)) for lbl, pre in sizes]
    pts = [(lbl, s) for lbl, s in pts if s]
    if len(pts) < 4:
        print("  ! skip fig_decoder_scale (matched sweep incomplete)")
        return

    xs = list(range(len(pts)))
    pm = [s[0] for _, s in pts]
    psd = [s[1] for _, s in pts]
    km = [s[2] for _, s in pts]
    ksd = [s[3] for _, s in pts]

    # Full two-column width. Frontiers: "a width that corresponds to one column
    # (85 mm) or two columns (180 mm)" -- an intermediate width is not one of the
    # two accepted sizes, so this is W2 and is included at \linewidth.
    fig, ax = plt.subplots(figsize=(W2, 0.52 * W2))
    ax.errorbar(xs, pm, yerr=psd, fmt="o-", color=C_PATH, label="Malignancy F1",
                markersize=7, markeredgecolor="white", markeredgewidth=1.0,
                capsize=4, linewidth=2.2)
    ax.errorbar(xs, km, yerr=ksd, fmt="s--", color=C_RISK, label="BI-RADS risk F1",
                markersize=7, markeredgecolor="white", markeredgewidth=1.0,
                capsize=4, linewidth=2.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _ in pts])
    ax.set_xlabel("Report decoder size (matched conditions)")
    ax.set_ylabel("F1")
    ax.set_ylim(0.68, 0.90)
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=9.5)
    # single-seed points carry no error bar; say so rather than imply precision
    for i, (_, s) in enumerate(pts):
        if s[4] == 1:
            ax.annotate("1 seed", (i, pm[i]), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=8.5, color=C_MUTED)
    _save(fig, outdir, "fig_decoder_scale")


# ── Figure 3: encoder comparison ──────────────────────────────────────────
def fig_encoder_compare(outdir: Path) -> None:
    """Encoder comparison. The six contested encoders carry 3-seed error bars
    (batch 5); the remaining rows are single-seed legacy runs, drawn without
    error bars and marked, so the reader can see which orderings are resolvable."""
    rows = _csv_rows()
    multi = [("DINOv3 ViT-L/16", "enc_dinov3_L"), ("ImageNet-21k ViT-L/16", "enc_in21k_L"),
             ("EVA-02 ViT-L/14", "enc_eva02_L"), ("DINOv2 ViT-g/14", "enc_dinov2_giant"),
             ("RAD-DINO", "enc_raddino"), ("SigLIP ViT-L/16", "enc_siglip_L")]
    single = [("DINOv2 ViT-L/14 +reg", "enc_dinov2reg_L"),
              ("AIMv2 ViT-L/14", "enc_aimv2_L"), ("CLIP ViT-L/14", "enc_clip_L")]
    # the 5-seed DINOv2 replication: its runs do not share a "<prefix>_s" pattern,
    # so it is aggregated explicitly rather than through _seed_stats
    DINOV2_5 = ["dinov2_seed1", "dinov2_seed2", "dinov2_seed3", "dinov2_seed4", "x_dinov2"]

    data = []  # (label, path_mean, path_sd, risk_mean, risk_sd, n)
    for lbl, pre in multi:
        st_ = _seed_stats(pre)
        if st_:
            data.append((lbl, st_[0], st_[1], st_[2], st_[3], st_[4]))
    import statistics as _st
    _p = [float(rows[k]["path_f1"]) for k in DINOV2_5 if k in rows]
    _k = [float(rows[k]["risk_f1"]) for k in DINOV2_5 if k in rows]
    if len(_p) > 1:
        data.append(("DINOv2 ViT-L/14", _st.mean(_p), _st.stdev(_p),
                     _st.mean(_k), _st.stdev(_k), len(_p)))
    for lbl, k in single:
        if k in rows:
            data.append((lbl, float(rows[k]["path_f1"]), 0.0,
                         float(rows[k]["risk_f1"]), 0.0, 1))
    up, ur = (_f1("outputs/submitted_version/uni2h", "pathology"),
              _f1("outputs/submitted_version/uni2h", "risk"))
    if up is not None:
        data.append(("UNI2-h ViT-H/14 (histopath.)", up, 0.0, ur, 0.0, 1))
    data.sort(key=lambda t: -t[1])

    # Row pitch bumped 0.055->0.075*W2 and value-label font 7->9.5pt so the
    # eleven rows aren't cramped when printed at full \linewidth.
    fig, ax = plt.subplots(figsize=(W2 * 0.97, 0.075 * W2 * len(data) + 0.9))
    y = range(len(data))
    hh = 0.38
    ax.barh([i + hh / 2 for i in y], [d[1] for d in data], height=hh,
            xerr=[d[2] for d in data], error_kw=dict(ecolor=INK, lw=1.2, capsize=2.5),
            color=C_PATH, label="Malignancy F1", edgecolor="white", linewidth=0.8)
    ax.barh([i - hh / 2 for i in y], [d[3] for d in data], height=hh,
            xerr=[d[4] for d in data], error_kw=dict(ecolor=INK, lw=1.2, capsize=2.5),
            color=C_RISK, label="BI-RADS risk F1", edgecolor="white", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{d[0]}" + ("" if d[5] > 1 else " *") for d in data])
    ax.invert_yaxis()
    ax.set_xlabel("F1 on the BUS-CoT held-out subset   (* = single seed, no error bar)")
    ax.set_xlim(0.4, 0.94)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    ax.grid(axis="y", visible=False)
    for i, d in enumerate(data):
        ax.text(d[1] + d[2] + 0.008, i + hh / 2, f"{d[1]:.3f}", va="center", fontsize=9.5)
        ax.text(d[3] + d[4] + 0.008, i - hh / 2, f"{d[3]:.3f}", va="center", fontsize=9.5)
    _save(fig, outdir, "fig_encoder_compare")


# ── Figure 4: bottleneck spectrum ─────────────────────────────────────────
def fig_bottleneck_spectrum(outdir: Path) -> None:
    rows = _csv_rows()
    spec = [("Single vector\n(opaque)", "eval_submitted_version_uni2h"),
            ("Multi-query\nK=4", "mq_k4"), ("Multi-query\nK=8", "mq_k8"),
            ("Multi-query\nK=32", "mq_k32"),
            ("Concept\nbottleneck", "submitted_version/cb3"),
            ("Concept +\nresidual", "cb_resid256")]
    data = []
    for lbl, key in spec:
        if key.startswith("eval_"):
            p, r = _f1(f"outputs/{key}", "pathology"), _f1(f"outputs/{key}", "risk")
        elif key in rows:
            p, r = float(rows[key]["path_f1"]), float(rows[key]["risk_f1"])
        else:
            continue
        if p is not None:
            data.append((lbl, p, r))

    fig, ax = plt.subplots(figsize=(W2, 0.46 * W2))
    xs = range(len(data))
    ax.plot(xs, [d[1] for d in data], "o-", color=C_PATH, label="Malignancy F1",
            markersize=7, markeredgecolor="white", markeredgewidth=1.0, linewidth=2.2)
    ax.plot(xs, [d[2] for d in data], "s--", color=C_RISK, label="BI-RADS risk F1",
            markersize=7, markeredgecolor="white", markeredgewidth=1.0, linewidth=2.2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel("F1")
    ax.set_xlabel("Information reaching the decoder  (narrow $\\rightarrow$ wide)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, outdir, "fig_bottleneck_spectrum")


# ── Figure 5: faithfulness + descriptor co-variation ──────────────────────
def fig_faithfulness(outdir: Path) -> None:
    base = _load_json("outputs/submitted_version/cb3/intervention.json")
    # Prefer the batch-5 3-head measurement (corrected supervision); fall back to
    # the original run. The 9-head arm is deliberately NOT plotted: that model
    # generates a prose surface form the style-A slot regex cannot parse, so its
    # zeros are an instrument limitation, not a measurement.
    cov = (_load_json("outputs/stats/intervention_covariation_base3.json")
           or _load_json("outputs/stats/intervention_covariation.json"))
    if not base:
        print("  ! skip fig_faithfulness")
        return
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.5 * W2))
    fig.subplots_adjust(wspace=0.38)

    ax = axes[0]
    heads = [("Pathology", "pathology"), ("BI-RADS\nrisk", "risk"),
             ("BI-RADS\ncategory", "birads")]
    vals = [base["agreement"][k] for _, k in heads if base["agreement"].get(k) is not None]
    labs = [lbl for lbl, k in heads if base["agreement"].get(k) is not None]
    bars = ax.bar(labs, vals, color=[C_PATH, C_RISK, C_ALT][:len(vals)],
                  edgecolor="white", linewidth=0.8, width=0.6)
    ax.axhline(0.5, color=C_MUTED, linestyle="--", linewidth=1.2,
               label="chance (2-way heads)")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Forced-concept agreement")
    ax.set_title("(A) The forced concept is adopted", loc="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}",
                ha="center", fontsize=9.5)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    if cov:
        short = {"orientation": "orient.", "margins": "margins",
                 "shape": "shape", "echogenicity": "echo."}
        slots = list(cov["flip"]["per_slot_change_rate"].keys())
        flip = [cov["flip"]["per_slot_change_rate"][s] or 0 for s in slots]
        ctrl = [cov["control"]["per_slot_change_rate"][s] or 0 for s in slots]
        x = range(len(slots))
        w = 0.38
        ax.bar([i - w / 2 for i in x], flip, width=w, color=C_PATH,
               label="label flipped", edgecolor="white", linewidth=0.8)
        ax.bar([i + w / 2 for i in x], ctrl, width=w, color=C_MUTED,
               label="control (no flip)", edgecolor="white", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([short.get(s, s) for s in slots])
        ax.set_ylabel("Descriptor change rate")
        ax.set_ylim(0, max(flip + ctrl) * 1.45 + 0.02)
        ax.legend(frameon=False, loc="upper right", fontsize=9.5)
        ax.set_title("(B) Descriptors partly follow", loc="left")
        ax.grid(axis="x", visible=False)
    _save(fig, outdir, "fig_faithfulness")


# ── Figure 6 (new): malignancy ROC, internal + external ───────────────────
def fig_roc(outdir: Path) -> None:
    a = _load_json("outputs/stats/probs_internal.json")
    b = _load_json("outputs/stats/submitted_model_u2bench.json")
    if not a:
        print("  ! skip fig_roc (run scripts/dump_concept_probs.py)")
        return
    # Full two-column width (see fig_decoder_scale for the 85/180 mm rule). The
    # axes keep a square aspect, as is conventional for an ROC, so the saved
    # figure is close to square; at 180 mm wide it is ~170 mm tall, well inside
    # the 236 mm text height, i.e. still under one page.
    fig, ax = plt.subplots(figsize=(W2, 0.95 * W2))
    for src, colour, style, label in ((a, C_PATH, "-", "BUS-CoT held-out"),
                                      (b, C_RISK, "--", "U2-BENCH (external)")):
        if not src:
            continue
        ax.plot([p["fpr"] for p in src["roc_curve"]],
                [p["tpr"] for p in src["roc_curve"]],
                style, color=colour, linewidth=2.2,
                label=f"{label}\nAUC {_r3(src['roc_auc'])} "
                      f"[{_r3(src['roc_auc_ci95'][0])}, {_r3(src['roc_auc_ci95'][1])}]")
    ax.plot([0, 1], [0, 1], color=C_MUTED, linewidth=1.0, linestyle=":")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.set_aspect("equal")
    _save(fig, outdir, "fig_roc")


# ── Figure 7: UMAP, two rows so each panel is legible ─────────────────────
# Re-derived from the same embeddings.npz (raw, high-dim) that
# scripts/visualize_embeddings.py already wrote, with the identical UMAP
# hyperparameters (n_neighbors=15, min_dist=0.1, random_state=42) it used, so
# the layout is the one already reported, just re-rendered: previously three
# panels were squeezed into one flat row (aspect ~3.5:1), which made every
# point and legend illegible. Two rows roughly doubles each panel's linear
# size. Panel C's axis limits are clipped to the 1st-99th percentile of the
# layout: a handful of far UMAP outliers were otherwise stretching the frame
# so far that the main image/text clusters collapsed into a corner.
_BIRADS_PALETTE = {
    "2": "#C6DBEF", "3": "#9ECAE1", "4A": "#6BAED6", "4B": "#4C78A8",
    "4C": "#F9A85B", "5": "#F58518", "6": "#B34D00",
}
_PATHOLOGY_PALETTE = {"benign": C_PATH, "malignant": C_RISK}


def fig_umap(outdir: Path, npz_path: str = "outputs/viz/v2/embeddings.npz",
             birads_label: str = "BI-RADS category") -> None:
    import numpy as np

    p = Path(npz_path)
    if not p.exists():
        print(f"  ! skip fig_umap ({npz_path} not found)")
        return
    import umap

    d = np.load(p, allow_pickle=True)
    img_embs, txt_embs = d["img_embs"], d["txt_embs"]
    birads = [b if b else None for b in d["birads"]]
    pathology = [x if x else None for x in d["pathology"]]

    reducer_img = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_img = reducer_img.fit_transform(img_embs)
    all_embs = np.concatenate([img_embs, txt_embs], axis=0)
    is_text = np.concatenate([np.zeros(len(img_embs), bool), np.ones(len(txt_embs), bool)])
    reducer_combined = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_combined = reducer_combined.fit_transform(all_embs)

    fig = plt.figure(figsize=(W2, 0.92 * W2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.20)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    def _scatter(ax, coords, labels, palette, title, ncol):
        plotted = set()
        for value, color in palette.items():
            mask = np.array([lab == value for lab in labels])
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=16, alpha=0.75,
                       edgecolors="white", linewidths=0.3, label=f"{value} (n={int(mask.sum())})")
            plotted.add(value)
        mask_unk = np.array([lab is None or lab not in plotted for lab in labels])
        if mask_unk.sum() > 0:
            ax.scatter(coords[mask_unk, 0], coords[mask_unk, 1], c="lightgray", s=10,
                       alpha=0.4, edgecolors="white", linewidths=0.2,
                       label=f"unknown (n={int(mask_unk.sum())})")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        # Legend goes BELOW the panel, laid out horizontally. Because these axes
        # use aspect="equal" with adjustable="datalim", the embedding fills the
        # whole axes box, so any in-axes placement (including "best", which is
        # what was used before) necessarily covers part of the scatter -- panel B's
        # seven-entry legend hid a wide strip of the band. Moving it out of the
        # axes is the only placement that leaves the data fully visible.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=ncol,
                  fontsize=7.5, frameon=True, framealpha=0.9, markerscale=1.3,
                  borderaxespad=0.0, columnspacing=1.4, handletextpad=0.4)
        ax.set_aspect("equal", adjustable="datalim")

    _scatter(axA, umap_img, pathology, _PATHOLOGY_PALETTE,
             "(A) Image embeddings by pathology", ncol=2)
    _scatter(axB, umap_img, birads, _BIRADS_PALETTE,
             f"(B) Image embeddings by {birads_label}", ncol=2)

    axC.scatter(umap_combined[~is_text, 0], umap_combined[~is_text, 1], c=C_PATH, s=14,
                alpha=0.7, edgecolors="white", linewidths=0.3,
                label=f"image (n={int((~is_text).sum())})")
    axC.scatter(umap_combined[is_text, 0], umap_combined[is_text, 1], c=C_RISK, s=14,
                alpha=0.7, edgecolors="white", linewidths=0.3,
                label=f"text (n={int(is_text.sum())})")
    # robust limits: a few far outliers otherwise dominate the frame
    lo, hi = 1, 99
    xlo, xhi = np.percentile(umap_combined[:, 0], [lo, hi])
    ylo, yhi = np.percentile(umap_combined[:, 1], [lo, hi])
    xpad, ypad = 0.08 * (xhi - xlo), 0.08 * (yhi - ylo)
    axC.set_xlim(xlo - xpad, xhi + xpad)
    axC.set_ylim(ylo - ypad, yhi + ypad)
    axC.set_title("(C) Image vs. text embeddings (shared space)", fontsize=11, loc="left")
    axC.set_xticks([])
    axC.set_yticks([])
    # below the panel, matching A and B, so no legend sits on top of the embedding
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=2,
               fontsize=7.5, frameon=True, framealpha=0.9, markerscale=1.3,
               borderaxespad=0.0, columnspacing=1.4, handletextpad=0.4)
    # adjustable="box" (not "datalim"): keeps the robust xlim/ylim above in
    # effect by resizing the axes box instead of re-expanding the data limits
    # back out to the outliers.
    axC.set_aspect("equal", adjustable="box")

    _save(fig, outdir, "fig_umap")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="paper/frontiers/submission")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing 300-dpi figures to {outdir}")
    fig_architecture(outdir)
    fig_decoder_scale(outdir)
    fig_encoder_compare(outdir)
    fig_bottleneck_spectrum(outdir)
    fig_faithfulness(outdir)
    fig_roc(outdir)
    fig_umap(outdir)
    print("done")


if __name__ == "__main__":
    main()
