"""Quantify the cross-modal modality gap from saved embeddings.

Reports linear CKA between image and predicted-text embeddings (low => the two
modalities occupy dissimilar subspaces) and cross-modal retrieval R@k (how often
the matching text is retrieved for an image and vice versa). Uses the
embeddings.npz files already produced by scripts/visualize_embeddings.py - no
model re-run required.

Results are printed and, when --output is given, persisted as JSON so the paper's
modality-gap numbers are artifact-backed rather than only appearing in the .tex.

Usage:
    python scripts/modality_gap.py outputs/viz/v2/embeddings.npz \
        --output outputs/stats/modality_gap.json
    # or over all viz dirs at once:
    python scripts/modality_gap.py outputs/viz/*/embeddings.npz \
        --output outputs/stats/modality_gap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def retrieval_at_k(img: np.ndarray, txt: np.ndarray, ks=(1, 5, 10)) -> dict:
    img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
    txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8)
    sim = img @ txt.T                      # [N, N]; diagonal = matching pair
    n = sim.shape[0]
    out = {}
    for name, S in [("i2t", sim), ("t2i", sim.T)]:
        ranks = (S >= S[np.arange(n), np.arange(n)][:, None]).sum(1)  # rank of the true match
        for k in ks:
            out[f"{name}_R@{k}"] = float((ranks <= k).mean())
        out[f"{name}_medrank"] = float(np.median(ranks))
    return out


def mean_gap(img: np.ndarray, txt: np.ndarray) -> float:
    """Euclidean distance between the two modality centroids (Liang et al. 2022)."""
    ic = img.mean(0) / (np.linalg.norm(img.mean(0)) + 1e-8)
    tc = txt.mean(0) / (np.linalg.norm(txt.mean(0)) + 1e-8)
    return float(np.linalg.norm(ic - tc))


def gap_for(path: str) -> dict:
    """Compute all modality-gap metrics for one embeddings.npz file."""
    d = np.load(path)
    img, txt = d["img_embs"].astype(np.float64), d["txt_embs"].astype(np.float64)
    r = retrieval_at_k(img, txt)
    return {
        "n": int(img.shape[0]),
        "dim": int(img.shape[1]),
        "linear_cka": linear_cka(img, txt),
        "centroid_gap": mean_gap(img, txt),
        "i2t_R@1": r["i2t_R@1"],
        "i2t_R@5": r["i2t_R@5"],
        "i2t_R@10": r["i2t_R@10"],
        "t2i_R@1": r["t2i_R@1"],
        "t2i_R@5": r["t2i_R@5"],
        "t2i_R@10": r["t2i_R@10"],
        "i2t_median_rank": r["i2t_medrank"],
        "t2i_median_rank": r["t2i_medrank"],
        "chance_median_rank": img.shape[0] / 2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="One or more embeddings.npz files.")
    ap.add_argument("--output", default=None,
                    help="JSON file to persist results (default: print only).")
    args = ap.parse_args()

    results = {}
    for path in args.paths:
        m = gap_for(path)
        # Key by the parent dir name (e.g. outputs/viz/v2/embeddings.npz -> "v2").
        key = Path(path).parent.name
        results[key] = m
        print(f"{path}  (n={m['n']}, dim={m['dim']})")
        print(f"  linear CKA(image, text) = {m['linear_cka']:.3f}")
        print(f"  modality-gap distance   = {m['centroid_gap']:.3f}")
        print("  cross-modal retrieval:")
        for k in ("i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"):
            print(f"    {k} = {m[k]:.3f}")
        print(f"    median rank i2t={m['i2t_median_rank']:.0f}  "
              f"t2i={m['t2i_median_rank']:.0f}  (chance median ~{m['chance_median_rank']:.0f})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
