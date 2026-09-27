"""Linear probe on frozen encoder features -- the missing lower bound.

Reviewer item 4: every clinical number in the paper is a label recovered by regex from
generated text. A logistic-regression probe on the frozen image encoder's pooled
features, trained on the same labels and the same split, is the natural lower bound. If
the probe matches the full pipeline, then the predictor, the InfoNCE alignment stage and
the 2.7B decoder contribute nothing measurable and the paper's real result is about
encoder selection. The reviewers call this decisive either way, so we run it.

The probe deliberately uses ONLY the frozen x-encoder (no predictor, no alignment, no
decoder), so it isolates what is linearly decodable from the pretrained features.

Usage:
    python scripts/linear_probe.py \
        --checkpoint checkpoints/runs/dinov2_cb/final.ckpt \
        --output outputs/stats/linear_probe.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
random.seed(0)
np.random.seed(0)
N_BOOT = 2000


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def boot_ci(y_true: np.ndarray, y_pred: np.ndarray, clusters: list[list[int]],
            n: int = N_BOOT) -> list[float]:
    """95% interval of F1, resampling whole patient clusters (as scripts/stats.py)."""
    c = len(clusters)
    vals = []
    for _ in range(n):
        idx = [i for k in np.random.randint(0, c, c) for i in clusters[k]]
        vals.append(f1_binary(y_true[idx], y_pred[idx]))
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)]


def patient_clusters(ids: list[str], test_jsonl: str) -> list[list[int]]:
    """Positions of `ids` grouped by the split's patient/frame group."""
    group_of = {}
    for line in open(test_jsonl):
        r = json.loads(line)
        group_of[r["id"]] = (r.get("metadata") or {}).get("group") or r["id"]
    by_group: dict[str, list[int]] = {}
    for pos, i in enumerate(ids):
        by_group.setdefault(group_of.get(i, i), []).append(pos)
    return list(by_group.values())


@torch.no_grad()
def extract(model, jsonl: str, device, image_size: int, batch_size: int,
            num_workers: int, restrict: set[str] | None = None):
    """Pooled frozen-encoder features + concept labels for one split."""
    from src.data.datasets.bus_cot_reports import BUSCoTReportDataset, _ultrasound_transform

    ds = BUSCoTReportDataset(jsonl_path=jsonl, root_dir=Path("."),
                           transform=_ultrasound_transform(image_size, train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, path_y, risk_y, ids = [], [], [], []
    for batch in tqdm(loader, desc=f"features {Path(jsonl).stem}"):
        out = model.x_encoder(batch["image"].to(device))
        # [B, N, D] patch tokens -> mean-pool; [B, D] already pooled
        pooled = out.mean(dim=1) if out.dim() == 3 else out
        feats.append(pooled.float().cpu().numpy())
        path_y.extend(batch["labels"]["pathology"].tolist())
        risk_y.extend(batch["labels"]["risk"].tolist())
        bid = batch.get("id")
        ids.extend(bid if isinstance(bid, list) else [""] * pooled.shape[0])

    X = np.concatenate(feats, 0)
    keep = np.ones(len(X), bool)
    if restrict is not None:
        keep &= np.array([i in restrict for i in ids])
    return X[keep], np.array(path_y)[keep], np.array(risk_y)[keep], \
        [i for i, k in zip(ids, keep) if k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_jsonl", default="data/split_augmented/train.jsonl")
    ap.add_argument("--test_jsonl", default="data/split_augmented/test.jsonl")
    ap.add_argument("--restrict-to", default="outputs/runs/dinov2_cb/predictions.json")
    ap.add_argument("--output", default="outputs/stats/linear_probe.json")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src.data.slot_labels import IGNORE_INDEX
    from src.model.report_model import ConceptReportModel

    logger.info("Loading %s (using its frozen x-encoder only)", args.checkpoint)
    model = ConceptReportModel.load_from_checkpoint(args.checkpoint, stage="finetune")
    model.eval().to(args.device)
    device = torch.device(args.device)

    keep = None
    if args.restrict_to and Path(args.restrict_to).exists():
        keep = {d["id"] for d in json.load(open(args.restrict_to))}

    Xtr, ptr, rtr, _ = extract(model, args.train_jsonl, device, args.image_size,
                               args.batch_size, args.num_workers)
    Xte, pte, rte, te_ids = extract(model, args.test_jsonl, device, args.image_size,
                               args.batch_size, args.num_workers, restrict=keep)
    logger.info("features: train %s test %s", Xtr.shape, Xte.shape)

    res: dict = {"checkpoint": args.checkpoint, "feature_dim": int(Xtr.shape[1]),
                 "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                 "endpoints": {}, "bootstrap": "patient-cluster (split group)",
                 "predictions": {}}

    for name, ytr_all, yte_all in (("pathology", ptr, pte), ("risk", rtr, rte)):
        mtr = ytr_all != IGNORE_INDEX
        mte = yte_all != IGNORE_INDEX
        if mtr.sum() == 0 or mte.sum() == 0:
            continue
        scaler = StandardScaler().fit(Xtr[mtr])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(scaler.transform(Xtr[mtr]), ytr_all[mtr])
        pred = clf.predict(scaler.transform(Xte[mte]))
        y = yte_all[mte]
        ids = [i for i, k in zip(te_ids, mte) if k]
        clusters = patient_clusters(ids, args.test_jsonl)
        f1 = f1_binary(y, pred)
        res["predictions"][name] = dict(zip(ids, pred.tolist()))
        res["endpoints"][name] = {
            "n_train": int(mtr.sum()), "n_test": int(mte.sum()), "n_clusters": len(clusters),
            "f1": round(f1, 4), "f1_ci95": boot_ci(y, pred, clusters),
            "accuracy": round(float((y == pred).mean()), 4),
            "positive_prevalence_test": round(float((y == 1).mean()), 4),
        }
        print(f"  linear probe {name:9s} F1 {f1:.4f} "
              f"CI {res['endpoints'][name]['f1_ci95']} "
              f"(n_test={int(mte.sum())})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
