"""Evaluation entrypoint for ultrasound VL-JEPA (Stage 2 finetune checkpoint).

Generates reports for every test sample using the Y-Decoder, then computes
BLEU-1/2/3/4, METEOR, and ROUGE-L against ground-truth BUS-CoT annotations.

Usage:
    python scripts/evaluate_jepa.py \
        --checkpoint checkpoints/ultrasound_jepa_finetune/final.ckpt \
        --test_jsonl data/unified/test.jsonl \
        --output_dir outputs/eval_jepa
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_reports(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[list[str], list[str], list[str]]:
    """Run generation over the dataloader.

    Returns:
        (predictions, references, ids)
    """
    model.eval()
    model.to(device)

    predictions, references, ids = [], [], []

    for batch in tqdm(dataloader, desc="Generating"):
        images = batch["image"].to(device)
        generated = model.forward_generate(
            images,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        predictions.extend(generated)
        references.extend(batch["report_text"])
        ids.extend(batch["id"])

    return predictions, references, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to finetune .ckpt")
    parser.add_argument("--test_jsonl", default="data/unified/test.jsonl")
    parser.add_argument("--output_dir", default="outputs/eval_jepa")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    from src.model.vl_jepa import HistoVLJEPA

    logger.info("Loading checkpoint: %s", args.checkpoint)
    # HistoVLJEPA.on_load_checkpoint() filters bitsandbytes quant buffers and
    # loads non-strict, so QLoRA (4-bit) checkpoints load with a single model
    # instance (no double-load → no OOM).
    model = HistoVLJEPA.load_from_checkpoint(args.checkpoint, stage="finetune")
    device = torch.device(args.device)

    # ── Test dataset ───────────────────────────────────────────────────────────
    from src.data.datasets.bus_cot_jepa import BUSCoTJEPADataset

    test_dataset = BUSCoTJEPADataset(
        jsonl_path=args.test_jsonl,
        root_dir=Path("."),
        image_size=args.image_size,
    )
    logger.info("Test samples: %d", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Generate ───────────────────────────────────────────────────────────────
    predictions, references, ids = generate_reports(
        model, test_loader, device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    from src.evaluation.metrics import compute_metrics

    logger.info("Computing metrics over %d samples...", len(predictions))
    metrics = compute_metrics(predictions, references)
    logger.info("\n%s", metrics)

    # ── Save outputs ───────────────────────────────────────────────────────────
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info("Metrics saved to %s/metrics.json", output_dir)

    results = [
        {"id": i, "prediction": p, "reference": r}
        for i, p, r in zip(ids, predictions, references)
    ]
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Predictions saved to %s/predictions.json", output_dir)

    # Print a few examples
    logger.info("\n--- Sample predictions ---")
    for ex in results[:3]:
        logger.info("ID: %s", ex["id"])
        logger.info("  REF : %s", ex["reference"][:120])
        logger.info("  PRED: %s", ex["prediction"][:120])
        logger.info("")


if __name__ == "__main__":
    main()
