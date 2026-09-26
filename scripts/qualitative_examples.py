"""Generate a qualitative results panel for the paper: real test-set images with
their generated report, the ground-truth report, and (for concept-bottleneck
checkpoints) the predicted concept probabilities the report was conditioned on.

Selects a diverse, non-cherry-picked set by slot-correctness bucket (both
pathology+risk correct / pathology correct only / a genuine miss) so the panel
is representative, not sanitized.

Usage:
    python scripts/qualitative_examples.py \
        --checkpoint checkpoints/weekend/dinov2_cb/epoch=05-val/loss=XXXX.ckpt \
        --test_jsonl data/unified_v2/test_buscot_only.jsonl \
        --output_dir outputs/qualitative/dinov2_cb --n_per_bucket 3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.slots import extract_slots, pathology_label, risk_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONCEPT_CLASS_NAMES = {
    "pathology": ["benign", "malignant"],
    "risk": ["low", "high"],
    "birads": ["2", "3", "4A", "4B", "4C", "5", "6"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_jsonl", default="data/unified_v2/test_buscot_only.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_per_bucket", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    from src.data.datasets.bus_cot_jepa import BUSCoTJEPADataset
    from src.model.vl_jepa import HistoVLJEPA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model = HistoVLJEPA.load_from_checkpoint(args.checkpoint, stage="finetune")
    model.eval().to(device)
    has_cb = getattr(model, "concept_bottleneck_enabled", False)
    logger.info("Concept bottleneck enabled: %s", has_cb)

    records = [json.loads(line) for line in open(args.test_jsonl) if line.strip()]
    id_to_record = {r["id"]: r for r in records}

    dataset = BUSCoTJEPADataset(jsonl_path=args.test_jsonl, root_dir=Path("."),
                                image_size=args.image_size)

    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = sample["id"]
            rec = id_to_record[sid]
            image = sample["image"].unsqueeze(0).to(device)

            vision_features = model.x_encoder(image)
            concept_probs = None
            if has_cb:
                predicted, aux_logits = model.predictor(vision_features, return_aux=True)
                concept_probs = {
                    head: torch.softmax(logits, dim=-1)[0].cpu().tolist()
                    for head, logits in aux_logits.items()
                }
            cond = model._decoder_conditioning(vision_features)
            report = model.y_decoder.generate(cond, max_new_tokens=args.max_new_tokens,
                                              num_beams=args.num_beams)[0]

            ref_text = sample["report_text"]
            ref_slots = extract_slots(ref_text)
            pred_slots = extract_slots(report)
            ref_path, pred_path = pathology_label(ref_slots), pathology_label(pred_slots)
            ref_risk, pred_risk = risk_label(ref_slots), risk_label(pred_slots)
            path_ok = ref_path is not None and ref_path == pred_path
            risk_ok = ref_risk is not None and ref_risk == pred_risk

            results.append({
                "id": sid,
                "image_path": rec["image_path"],
                "reference": ref_text,
                "prediction": report,
                "path_correct": path_ok,
                "risk_correct": risk_ok,
                "concept_probs": concept_probs,
                "metadata": rec.get("metadata", {}),
            })
            if (i + 1) % 50 == 0:
                logger.info("  %d/%d generated", i + 1, len(dataset))

    with open(out / "all_generations.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Bucket selection: both-correct / partial / miss ────────────────────────
    both = [r for r in results if r["path_correct"] and r["risk_correct"]]
    partial = [r for r in results if r["path_correct"] != r["risk_correct"]]
    miss = [r for r in results if not r["path_correct"] and not r["risk_correct"]]
    logger.info("Buckets: both=%d partial=%d miss=%d (of %d)",
                len(both), len(partial), len(miss), len(results))

    for group in (both, partial, miss):
        random.shuffle(group)
    selected = both[:args.n_per_bucket] + partial[:args.n_per_bucket] + miss[:args.n_per_bucket]

    panel_dir = out / "panel"
    panel_dir.mkdir(exist_ok=True)
    panel = []
    for j, r in enumerate(selected):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            dst = panel_dir / f"ex{j:02d}_{Path(r['image_path']).stem}.png"
            img.save(dst)
        except Exception as e:
            logger.warning("image copy failed for %s: %s", r["id"], e)
            dst = None
        panel.append({**r, "panel_image": str(dst) if dst else None})

    with open(out / "panel_selected.json", "w") as f:
        json.dump(panel, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d generations -> %s", len(results), out / "all_generations.json")
    logger.info("Saved %d panel examples -> %s", len(panel), out / "panel_selected.json")
    for r in panel:
        logger.info("\n[%s] path_ok=%s risk_ok=%s", r["id"], r["path_correct"], r["risk_correct"])
        logger.info("  PRED: %s", r["prediction"][:180])
        logger.info("  REF : %s", r["reference"][:180])


if __name__ == "__main__":
    main()
