"""Concept-intervention faithfulness test for the concept-bottleneck VL-JEPA.

For each test image we:
  1. generate the baseline report (predicted concepts),
  2. for each concept head, FORCE every class value and regenerate,
  3. check whether the generated report's extracted slot matches the forced value.

A faithful bottleneck must change its output to agree with an intervened concept.
We report, per head, the "intervention agreement" = fraction of forced cases where
the regenerated report's slot equals the forced class. High agreement = the report
genuinely depends on the concepts (interpretability is real, not post-hoc).

Usage:
    python scripts/concept_intervention.py \
        --checkpoint checkpoints/ultrasound_jepa_finetune_cb/epoch=07-val/loss=0.0854.ckpt \
        --test_jsonl data/unified_v2/test_buscot_only.jsonl \
        --train_config configs/train/finetune_jepa_v5.yaml \
        --n 80 --output outputs/eval_jepa_cb_buscot/intervention.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets.bus_cot_jepa import BUSCoTJEPADataset, preprocess_mode_of
from src.data.slot_labels import (
    BIRADS_CLASSES,
    CALCIFICATION_CLASSES,
    ECHO_CLASSES,
    LOW_RISK_BIRADS,
    MARGINS_CLASSES,
    ORIENTATION_CLASSES,
    PATHOLOGY_CLASSES,
    RISK_CLASSES,
    SHAPE_CLASSES,
)
from src.evaluation.slots import extract_slots
from src.model.vl_jepa import HistoVLJEPA

# inverse maps: class index -> the value the generated report should express
INV = {
    "pathology": {v: k for k, v in PATHOLOGY_CLASSES.items()},          # 0->benign,1->malignant
    "risk": {v: k for k, v in RISK_CLASSES.items()},                    # 0->low,1->high
    "birads": {v: k for k, v in BIRADS_CLASSES.items()},               # 0->'2',...
}
# Finding-level heads of the nine-head model. The extractor reads descriptors back
# as the same source enums the heads are trained on. `boundary` has no report slot.
DESCRIPTOR_HEADS = {
    "margins": MARGINS_CLASSES, "shape": SHAPE_CLASSES, "orientation": ORIENTATION_CLASSES,
    "echogenicity": ECHO_CLASSES, "calcification": CALCIFICATION_CLASSES,
}
INV.update({h: {v: k for k, v in classes.items()} for h, classes in DESCRIPTOR_HEADS.items()})


def report_slot_value(text: str, head: str) -> str | None:
    """Extract the value the report expresses for a given concept head (any style)."""
    slots = extract_slots(text)
    if head == "pathology":
        return (slots.get("pathology") or "").lower() or None
    if head == "birads":
        b = slots.get("birads")
        return b.upper() if b else None
    if head == "risk":
        b = slots.get("birads")
        if not b:
            return None
        return "low" if b.upper() in LOW_RISK_BIRADS else "high"
    return slots.get(head) if head in DESCRIPTOR_HEADS else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--train_config", required=True,
                    help="Stage-2 training config (its data.preprocess_mode is applied)")
    ap.add_argument("--n", type=int, default=80, help="number of test images to probe")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = HistoVLJEPA.load_from_checkpoint(args.checkpoint, stage="finetune")
    model = model.eval().to(args.device)
    assert model.concept_bottleneck_enabled, "checkpoint is not a concept-bottleneck model"

    ds = BUSCoTJEPADataset(jsonl_path=args.test_jsonl, root_dir=Path("."), image_size=224,
                           preprocess_mode=preprocess_mode_of(args.train_config))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    heads = [h for h in model.aux_heads if h in INV]
    # per head: reports generated under an override, how many state the slot at
    # all, and how many state the forced value. A report that omits the slot has
    # not followed the override, so agreement is over all generated reports.
    agree = {h: 0 for h in heads}
    total = {h: 0 for h in heads}
    extracted = {h: 0 for h in heads}
    examples = []

    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Intervening"):
            if seen >= args.n:
                break
            images = batch["image"].to(args.device)
            B = images.shape[0]
            for h in heads:
                n_classes = model.concept_bottleneck.embeds[h].num_embeddings
                for cls in range(n_classes):
                    if INV[h].get(cls) is None:
                        continue
                    ov = {h: torch.full((B,), cls, dtype=torch.long, device=args.device)}
                    gen = model.forward_generate(images, overrides=ov,
                                                 max_new_tokens=args.max_new_tokens)
                    target_val = INV[h][cls]
                    for k, text in enumerate(gen):
                        got = report_slot_value(text, h)
                        total[h] += 1
                        if got is None:
                            continue
                        extracted[h] += 1
                        ok = (got.lower() == str(target_val).lower())
                        agree[h] += int(ok)
                        if len(examples) < 12 and h == "pathology":
                            examples.append({"forced": f"{h}={target_val}",
                                             "report": text[:160], "match": ok})
            seen += B

    results = {
        "n_images": seen,
        "agreement": {h: (agree[h] / total[h] if total[h] else None) for h in heads},
        "counts": {h: {"agree": agree[h], "total": total[h], "extracted": extracted[h]}
                   for h in heads},
        "examples": examples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Concept-intervention agreement (faithfulness) ===")
    for h in heads:
        a = results["agreement"][h]
        c = results["counts"][h]
        print(f"  {h:10s}: {a:.3f}" if a is not None else f"  {h:10s}: n/a",
              f"  ({c['agree']}/{c['total']})")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
