"""Extract image+text embeddings from a trained report model and visualize with t-SNE/UMAP.

Generates 2D scatter plots colored by:
    - BI-RADS category (extracted from reference text via regex)
    - Pathology (malignant / benign, from metadata)
    - Modality (image vs text - both projected into shared embedding space)

Output:
    outputs/viz/<run_name>/
        embeddings.npz                  - raw embeddings + labels
        tsne_birads.png
        tsne_pathology.png
        tsne_modality.png
        umap_birads.png
        umap_pathology.png
        umap_modality.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets.bus_cot_reports import BUSCoTReportDataset, preprocess_mode_of
from src.model.report_model import ConceptReportModel


# ── Slot extraction (reuse from evaluate_slots.py) ─────────────────────────
def extract_birads(text: str) -> str | None:
    m = re.search(r"BIRADS\s+(\w+)", text, re.IGNORECASE)
    return m.group(1).strip().upper() if m else None


def extract_pathology(text: str) -> str | None:
    m = re.search(r"(malignant|benign)\s+lesion", text, re.IGNORECASE)
    return m.group(1).strip().lower() if m else None


# ── Plotting ───────────────────────────────────────────────────────────────
# BI-RADS is ordinal, so it gets a single-hue light->dark sequential ramp rather than
# a rainbow. Frontiers forbids conveying information by a red/green pair, so the old
# green->red ramp was replaced by blue->orange (validated for deuteranopia,
# protanopia and tritanopia separation in OKLab).
BIRADS_PALETTE = {
    "2":  "#C6DBEF",  # low risk  (light blue)
    "3":  "#9ECAE1",
    "4A": "#6BAED6",
    "4B": "#4C78A8",
    "4C": "#F9A85B",
    "5":  "#F58518",  # high risk (orange)
    "6":  "#B34D00",
}
PATHOLOGY_PALETTE = {"benign": "#4C78A8", "malignant": "#F58518"}


def scatter_plot(
    coords: np.ndarray,
    labels: list[str | None],
    title: str,
    palette: dict,
    out_path: Path,
    point_size: int = 24,
):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    plotted = set()
    for label_value, color in palette.items():
        mask = np.array([lab == label_value for lab in labels])
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=point_size, alpha=0.75,
            edgecolors="white", linewidths=0.4,
            label=f"{label_value} (n={int(mask.sum())})",
        )
        plotted.add(label_value)

    # unknown / unlabeled
    mask_unk = np.array([lab is None or lab not in plotted for lab in labels])
    if mask_unk.sum() > 0:
        ax.scatter(
            coords[mask_unk, 0], coords[mask_unk, 1],
            c="lightgray", s=point_size * 0.6, alpha=0.4,
            edgecolors="white", linewidths=0.3,
            label=f"unknown (n={int(mask_unk.sum())})",
        )

    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def scatter_modality(
    coords: np.ndarray,
    is_text: np.ndarray,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    ax.scatter(
        coords[~is_text, 0], coords[~is_text, 1],
        c="#1f77b4", s=24, alpha=0.7, edgecolors="white", linewidths=0.4,
        label=f"image (n={int((~is_text).sum())})",
    )
    ax.scatter(
        coords[is_text, 0], coords[is_text, 1],
        c="#ff7f0e", s=24, alpha=0.7, edgecolors="white", linewidths=0.4,
        label=f"text (n={int(is_text.sum())})",
    )
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=10, frameon=True)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to .ckpt")
    parser.add_argument("--test_jsonl", default="data/unified/test.jsonl")
    parser.add_argument("--train_config", required=True,
                        help="Stage-2 training config (its data.preprocess_mode is applied)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", default="pretrain",
                        help="model stage to load (pretrain or finetune)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model = ConceptReportModel.load_from_checkpoint(
        args.checkpoint, stage=args.stage, map_location=args.device
    ).eval()

    # ── Build dataset ──────────────────────────────────────────────────────
    test_ds = BUSCoTReportDataset(jsonl_path=args.test_jsonl, image_size=224,
                                preprocess_mode=preprocess_mode_of(args.train_config))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Extract embeddings ─────────────────────────────────────────────────
    img_embs: list[np.ndarray] = []
    txt_embs: list[np.ndarray] = []
    ids: list[str] = []
    refs: list[str] = []

    print(f"Extracting embeddings on {len(test_ds)} samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(args.device)
            texts = batch["report_text"]

            pred_emb, target_emb = model.forward_embedding(images, texts)
            img_embs.append(pred_emb.float().cpu().numpy())
            txt_embs.append(target_emb.float().cpu().numpy())
            ids.extend(batch["id"])
            refs.extend(texts)

    img_embs = np.concatenate(img_embs, axis=0)  # [N, D]
    txt_embs = np.concatenate(txt_embs, axis=0)  # [N, D]
    print(f"Image embeddings: {img_embs.shape}, Text embeddings: {txt_embs.shape}")

    # ── Extract labels from reference text + metadata ──────────────────────
    metadata = {}
    with open(args.test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            metadata[rec["id"]] = rec.get("metadata", {})

    birads = [extract_birads(r) for r in refs]
    pathology = [
        extract_pathology(r) or metadata.get(i, {}).get("pathology")
        for r, i in zip(refs, ids)
    ]

    # save raw
    np.savez(
        output_dir / "embeddings.npz",
        img_embs=img_embs, txt_embs=txt_embs,
        ids=np.array(ids),
        birads=np.array([b or "" for b in birads]),
        pathology=np.array([p or "" for p in pathology]),
    )

    # ── Combined image+text for modality plot ──────────────────────────────
    all_embs = np.concatenate([img_embs, txt_embs], axis=0)
    is_text = np.concatenate([np.zeros(len(img_embs), bool), np.ones(len(txt_embs), bool)])

    # ── t-SNE ──────────────────────────────────────────────────────────────
    print("Running t-SNE on image embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    tsne_img = tsne.fit_transform(img_embs)
    print("Running t-SNE on combined embeddings...")
    tsne_combined = tsne.fit_transform(all_embs)

    scatter_plot(tsne_img, birads, "t-SNE: image embeddings by BI-RADS", BIRADS_PALETTE,
                 output_dir / "tsne_birads.png")
    scatter_plot(tsne_img, pathology, "t-SNE: image embeddings by pathology", PATHOLOGY_PALETTE,
                 output_dir / "tsne_pathology.png")
    scatter_modality(tsne_combined, is_text, "t-SNE: image vs text embeddings (shared space)",
                     output_dir / "tsne_modality.png")

    # ── UMAP ───────────────────────────────────────────────────────────────
    import umap
    print("Running UMAP on image embeddings...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_img = reducer.fit_transform(img_embs)
    print("Running UMAP on combined embeddings...")
    umap_combined = reducer.fit_transform(all_embs)

    scatter_plot(umap_img, birads, "UMAP: image embeddings by BI-RADS", BIRADS_PALETTE,
                 output_dir / "umap_birads.png")
    scatter_plot(umap_img, pathology, "UMAP: image embeddings by pathology", PATHOLOGY_PALETTE,
                 output_dir / "umap_pathology.png")
    scatter_modality(umap_combined, is_text, "UMAP: image vs text embeddings (shared space)",
                     output_dir / "umap_modality.png")

    print(f"\nDone. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
