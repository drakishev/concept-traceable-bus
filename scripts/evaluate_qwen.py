"""Evaluate a LoRA-fine-tuned Qwen2-VL-7B baseline on the ultrasound test set.

Produces predictions.json + metrics.json in the SAME format as
scripts/evaluate_reports.py, so scripts/evaluate_slots.py works unchanged and
results drop straight into the comparison table.

Usage:
    python scripts/evaluate_qwen.py \
        --adapter checkpoints/qwen2vl_lora_v2/final \
        --test_jsonl data/split/test_buscot_only.jsonl \
        --output_dir outputs/eval_qwen_v2_buscot
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_metrics  # BLEU/METEOR/ROUGE-L
from src.model.qwen2vl import generate_report, load_model_and_processor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Describe the breast ultrasound findings and provide a BI-RADS assessment."


@torch.inference_mode()
def generate_report_generic(model, processor, image, prompt, max_new_tokens=256):
    """Model-agnostic single-image generation for any HF image-text-to-text VLM."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _pn = type(processor).__name__.lower()
    imgs = [[image]] if any(k in _pn for k in ("mllama", "gemma3")) else [image]
    inputs = processor(text=[text], images=imgs, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=processor.tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to trained LoRA adapter dir")
    ap.add_argument("--hf_id", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--generic", action="store_true",
                    help="Load via the generic AutoModelForImageTextToText path "
                         "(non-Qwen2-VL VLMs)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading VLM %s with adapter: %s", args.hf_id, args.adapter)
    if args.generic:
        from src.model.vlm import load_vlm_and_processor
        model, processor = load_vlm_and_processor(
            hf_id=args.hf_id, attn_implementation="sdpa",
            adapter_path=args.adapter, device_map="auto",
        )
    else:
        model, processor = load_model_and_processor(
            hf_id=args.hf_id, attn_implementation="sdpa",
            adapter_path=args.adapter, device_map="auto",
        )
    model.eval()

    records = []
    with open(args.test_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Test samples: %d", len(records))

    predictions = []
    for rec in tqdm(records, desc="Generating"):
        # use the dataset's own user prompt if present, else default
        convs = rec.get("conversations", [])
        user_text = args.prompt
        if convs and convs[0].get("content"):
            user_text = convs[0]["content"].replace("<image>", "").strip() or args.prompt
        reference = convs[1]["content"] if len(convs) >= 2 else ""

        image = Image.open(rec["image_path"]).convert("RGB")
        try:
            if args.generic:
                pred = generate_report_generic(
                    model, processor, image, user_text, max_new_tokens=args.max_new_tokens)
            else:
                pred = generate_report(
                    model, processor, image,
                    prompt=user_text,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
        except Exception as e:  # keep going on a single-sample failure
            logger.warning("gen failed for %s: %s", rec["id"], e)
            pred = ""
        predictions.append({"id": rec["id"], "reference": reference, "prediction": pred})

    with open(out / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Predictions saved to %s", out / "predictions.json")

    refs = [p["reference"] for p in predictions]
    hyps = [p["prediction"] for p in predictions]
    metrics = compute_metrics(hyps, refs)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info("Metrics:\n%s", metrics)


if __name__ == "__main__":
    main()
