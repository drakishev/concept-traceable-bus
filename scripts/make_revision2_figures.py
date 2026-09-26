"""Figures of the revised manuscript (revision 2), from batch-7 artifacts only.

Same print specification and palette as make_paper_figures.py (whose style,
architecture schematic and UMAP panel are reused). Every value is read from the
batch-7 outputs, so no number is hand-copied into a figure:
  outputs/weekend7_results.csv, outputs/stats/stats_batch7.json,
  outputs/stats/batch7_table.json, outputs/stats/tost_decoder_scale_batch7.json,
  outputs/analysis7/{intervention,covariation,probs,viz}/.

Usage:
    python scripts/make_revision2_figures.py --outdir paper/frontiers/revision2
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figures as base  # noqa: E402  (sets rcParams on import)
from make_paper_figures import C_MUTED, C_PATH, C_RISK, INK, W2, _save  # noqa: E402

plt = base.plt
PREFIX = "h_"
RESULTS = Path("outputs/weekend7_results.csv")
A7 = Path("outputs/analysis7")
SEEDS = (1, 2, 3)
MODELS = (("cb3", "CB-3 (assessment concepts)", C_PATH),
          ("cb9", "CB-9 (assessment + finding concepts)", C_RISK))


def _rows() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "skip"):
                out[r["name"]] = r
    return out


def _seed_vals(rows: dict, stem: str) -> tuple[list[float], list[float]]:
    names = [n for n in rows if n.startswith(PREFIX + stem + "_s")]
    return ([float(rows[n]["path_f1"]) for n in names],
            [float(rows[n]["risk_f1"]) for n in names])


def _msd(v: list[float]) -> tuple[float, float]:
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


# ── Figure 1: architecture ────────────────────────────────────────────────
def fig_architecture(outdir: Path) -> None:
    base.fig_architecture(outdir, concept_note="pathology / risk group /\n"
                                              "BI-RADS-like category;\nCB-9: + 6 finding heads")


# ── Figure 2: single-family decoder sweep ─────────────────────────────────
def fig_decoder_scale(outdir: Path) -> None:
    rows = _rows()
    sizes = [("0.5B", "0_5b", 0.5), ("1.5B", "1_5b", 1.5), ("3B", "3b", 3),
             ("7B", "7b", 7), ("14B", "14b", 14), ("32B", "32b", 32), ("72B", "72b", 72)]
    pts = []
    for label, tag, params in sizes:
        p, k = _seed_vals(rows, f"dec_qwen{tag}")
        pts.append((label, params, *_msd(p), *_msd(k), len(p)))
    x = [q[1] for q in pts]
    fig, ax = plt.subplots(figsize=(W2, 0.50 * W2))
    ax.errorbar(x, [q[2] for q in pts], yerr=[q[3] for q in pts], fmt="o-", color=C_PATH,
                label="Malignancy F1", markersize=7, markeredgecolor="white",
                markeredgewidth=1.0, capsize=4, linewidth=2.2)
    ax.errorbar(x, [q[4] for q in pts], yerr=[q[5] for q in pts], fmt="s--", color=C_RISK,
                label="BI-RADS-like risk-group F1", markersize=7, markeredgecolor="white",
                markeredgewidth=1.0, capsize=4, linewidth=2.2)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([q[0] for q in pts])
    ax.minorticks_off()
    ax.set_xlabel("Report decoder size, Qwen2.5-Instruct (log scale)")
    ax.set_ylabel("F1 (official BUS-CoT test set, n = 873)")
    ax.set_ylim(0.70, 0.84)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    for q in pts:
        if q[6] == 1:
            ax.annotate("1 seed", (q[1], q[2]), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, color=C_MUTED)
    _save(fig, outdir, "fig_decoder_scale")


# ── Figure 3: encoder comparison ──────────────────────────────────────────
ENCODERS = [("DINOv2 ViT-L/14", "enc_dinov2_L"), ("DINOv3 ViT-L/16", "enc_dinov3_L"),
            ("RAD-DINO ViT-B/14\n(chest X-ray)", "enc_raddino"),
            ("DINOv2 ViT-B/14", "enc_dinov2_B"), ("EVA-02 ViT-L/14", "enc_eva02_L"),
            ("USFM ViT-B/16\n(ultrasound)", "enc_usfm"),
            ("SigLIP ViT-L/16", "enc_siglip_L"),
            ("ImageNet-21k ViT-L/16\n(supervised)", "enc_in21k_L"),
            ("UNI2-h ViT-H/14\n(histopathology)", "enc_uni2h")]


def fig_encoder_compare(outdir: Path) -> None:
    rows = _rows()
    data = []
    for label, stem in ENCODERS:
        p, k = _seed_vals(rows, stem)
        data.append((label, *_msd(p), *_msd(k), p, k))
    data.sort(key=lambda d: d[1])
    y = np.arange(len(data))
    h = 0.38
    fig, ax = plt.subplots(figsize=(W2, 0.62 * W2))
    ax.barh(y + h / 2, [d[1] for d in data], h, xerr=[d[2] for d in data], color=C_PATH,
            label="Malignancy F1", capsize=3, error_kw={"linewidth": 1.2})
    ax.barh(y - h / 2, [d[3] for d in data], h, xerr=[d[4] for d in data], color=C_RISK,
            label="BI-RADS-like risk-group F1", capsize=3, error_kw={"linewidth": 1.2},
            hatch="//", edgecolor="white")
    for i, d in enumerate(data):
        ax.plot(d[5], [i + h / 2] * len(d[5]), "o", color=INK, markersize=2.5)
        ax.plot(d[6], [i - h / 2] * len(d[6]), "o", color=INK, markersize=2.5)
        ax.text(0.905, i, f"{d[1]:.3f} / {d[3]:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=9)
    ax.set_xlim(0.55, 0.97)
    ax.set_xlabel("F1, mean $\\pm$ SD over 3 seeds (dots: individual seeds)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=9)
    ax.grid(axis="y", visible=False)
    _save(fig, outdir, "fig_encoder_compare")


# ── Figure 4: concept intervention ────────────────────────────────────────
HEADS = [("pathology", "Pathology"), ("risk", "Risk group"), ("birads", "BI-RADS-like cat."),
         ("orientation", "Orientation"), ("margins", "Margins"), ("shape", "Shape"),
         ("echogenicity", "Echogenicity"), ("calcification", "Calcification")]


def fig_faithfulness(outdir: Path) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W2, 0.56 * W2),
                                   gridspec_kw={"width_ratios": [2.6, 1]})
    x = np.arange(len(HEADS))
    w = 0.38
    for j, (m, label, color) in enumerate(MODELS):
        per_seed = [json.load(open(A7 / "intervention" / f"{m}_s{s}.json"))["agreement"]
                    for s in SEEDS]
        means, xs = [], []
        for i, (head, _) in enumerate(HEADS):
            vals = [a[head] for a in per_seed if a.get(head) is not None]
            if not vals:
                continue
            xs.append(i + (j - 0.5) * w)
            means.append(st.mean(vals))
            axA.plot([i + (j - 0.5) * w] * len(vals), vals, "o", color=INK, markersize=2.5,
                     zorder=3)
        axA.bar(xs, means, w, color=color, label=label,
                hatch=None if j == 0 else "//", edgecolor="white")
    axA.set_xticks(x)
    axA.set_xticklabels([h[1] for h in HEADS], fontsize=8.5, rotation=35, ha="right",
                        rotation_mode="anchor")
    axA.set_ylim(0, 1.16)
    axA.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axA.set_ylabel("Agreement with forced concept")
    axA.set_title("(A) Forced-concept agreement", loc="left", fontsize=11)
    axA.axvline(2.5, color=C_MUTED, linewidth=0.8, linestyle=":")
    axA.text(1.0, 1.07, "assessment heads", fontsize=8, color=C_MUTED, ha="center")
    axA.text(5.0, 1.07, "finding heads (CB-9 only)", fontsize=8, color=C_MUTED, ha="center")
    axA.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2,
               fontsize=8.5)
    axA.grid(axis="x", visible=False)

    for j, (m, _, color) in enumerate(MODELS):
        cov = [json.load(open(A7 / "covariation" / f"{m}_s{s}.json")) for s in SEEDS]
        flip = [c["flip"]["any_descriptor_changed_rate"] for c in cov]
        ctrl = [c["control"]["any_descriptor_changed_rate"] for c in cov]
        for k, (vals, hatch) in enumerate(((flip, None), (ctrl, ".."))):
            xpos = j + (k - 0.5) * w
            axB.bar(xpos, st.mean(vals), w, color=color, hatch=hatch, edgecolor="white",
                    alpha=1.0 if k == 0 else 0.55)
            axB.plot([xpos] * len(vals), vals, "o", color=INK, markersize=2.5, zorder=3)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["CB-3", "CB-9"])
    axB.set_ylim(0, 1.16)
    axB.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axB.set_ylabel("Any descriptor changes")
    axB.set_title("(B) Pathology flip vs. control", loc="left", fontsize=11)
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor=C_MUTED, label="forced flip"),
                        Patch(facecolor=C_MUTED, alpha=0.55, hatch="..", label="control")],
               frameon=False, loc="upper right", fontsize=8.5)
    axB.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, outdir, "fig_faithfulness")


# ── Figure 5: ROC, internal and two external sets ─────────────────────────
def _roc(y: list[int], p: list[float]) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr


def fig_roc(outdir: Path) -> None:
    from sklearn.metrics import roc_auc_score
    sets = [("test", "(A) Internal: BUS-CoT test"), ("u2bench", "(B) External: U2-BENCH"),
            ("breast", "(C) External: BrEaST")]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 0.40 * W2), sharey=True)
    for ax, (key, title) in zip(axes, sets):
        n = None
        for m, label, color in MODELS:
            aucs = []
            for s in SEEDS:
                d = json.load(open(A7 / "probs" / f"{m}_s{s}" / f"{key}.json"))
                rec = d["records"] if "records" in d else d[list(d)[-1]]
                y = [r["y"] for r in rec]
                p = [r["p_malignant"] for r in rec]
                n = len(y)
                fpr, tpr = _roc(y, p)
                aucs.append(roc_auc_score(y, p))
                ax.plot(fpr, tpr, color=color, linewidth=1.3, alpha=0.8,
                        linestyle="-" if m == "cb3" else "--")
            auc = f"AUC {st.mean(aucs):.2f} $\\pm$ {st.stdev(aucs):.2f}"
            ax.plot([], [], color=color, linestyle="-" if m == "cb3" else "--",
                    label=f"{label.split(' ')[0]}  {auc}")
        ax.plot([0, 1], [0, 1], ":", color=C_MUTED, linewidth=1.0)
        ax.set_title(f"{title}\n(n = {n})", loc="left", fontsize=10)
        ax.set_xlabel("1 - specificity")
        ax.set_aspect("equal")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel("Sensitivity")
    fig.tight_layout()
    _save(fig, outdir, "fig_roc")


# ── Figure 6: embeddings ──────────────────────────────────────────────────
def fig_umap(outdir: Path) -> None:
    base.fig_umap(outdir, npz_path=str(A7 / "viz" / "cb3_s1" / "embeddings.npz"),
                  birads_label="BI-RADS-like category")


# ── Supplementary: reliability diagrams ───────────────────────────────────
def fig_reliability(outdir: Path) -> None:
    sets = [("internal_test", "Internal test"), ("u2bench", "U2-BENCH"), ("breast", "BrEaST")]
    fig, axes = plt.subplots(2, 3, figsize=(W2, 0.66 * W2), sharex=True, sharey=True)
    for r, (m, label, color) in enumerate(MODELS):
        for c, (key, title) in enumerate(sets):
            ax = axes[r, c]
            eces = []
            for s in SEEDS:
                cal = json.load(open(A7 / "probs" / f"{m}_s{s}" / "calibration.json"))
                d = cal[key] if key == "internal_test" else cal["external"][key]
                bins = [b for b in d["reliability"]["bins"] if b["n"] > 0]
                ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-",
                        color=color, linewidth=1.2, markersize=3, alpha=0.85)
                eces.append(d["reliability"]["ece"])
            ax.plot([0, 1], [0, 1], ":", color=C_MUTED, linewidth=1.0)
            ax.set_title(f"{label.split(' ')[0]}, {title}\nECE {min(eces):.2f}-{max(eces):.2f}",
                         fontsize=9, loc="left")
            ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel("Predicted P(malignant)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed fraction")
    fig.tight_layout()
    _save(fig, outdir, "supp_fig_reliability")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="paper/frontiers/revision2")
    args = ap.parse_args()
    from u2bench_labels import require_labels
    require_labels()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fn in (fig_architecture, fig_decoder_scale, fig_encoder_compare, fig_faithfulness,
               fig_roc, fig_umap, fig_reliability):
        fn(outdir)


if __name__ == "__main__":
    main()
