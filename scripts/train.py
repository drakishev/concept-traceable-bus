"""LoRA SFT training entrypoint for Qwen2-VL on breast ultrasound data.

Uses a multimodal data collator so images are properly fed through the
vision encoder at every step.

Usage (single GPU, debug):
    python scripts/train.py --config configs/train/lora_sft.yaml \
        train.per_device_train_batch_size=1 train.num_train_epochs=1

Usage (multi-GPU with Accelerate):
    accelerate launch --num_processes 8 scripts/train.py \
        --config configs/train/lora_sft.yaml

Usage (resume from checkpoint):
    accelerate launch --num_processes 8 scripts/train.py \
        --config configs/train/lora_sft.yaml \
        train.resume_from_checkpoint=checkpoints/qwen2vl_lora/checkpoint-400
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    import datasets

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl_records(jsonl_path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_dataset(records: list[dict]) -> "datasets.Dataset":
    """Convert unified JSONL records to a HuggingFace Dataset.

    Each row stores the raw strings needed by the collator:
        image_path, user_text, assistant_text
    The collator opens the image and calls the processor at batch time.
    """
    from datasets import Dataset

    rows = []
    for rec in records:
        convs = rec["conversations"]
        if len(convs) < 2:
            continue
        user_content = convs[0]["content"]
        # Strip the <image> placeholder — the collator inserts the real image object
        user_text = user_content.replace("<image>", "").strip()
        assistant_text = convs[1]["content"]
        rows.append(
            {
                "image_path": rec["image_path"],
                "user_text": user_text,
                "assistant_text": assistant_text,
                "id": rec["id"],
            }
        )

    logger.info("Built HF dataset with %d rows", len(rows))
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    # ── Load and merge configs ─────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if cfg.get("defaults"):
        for default in cfg.defaults:
            for key, val in default.items():
                sub_cfg_path = Path(f"configs/{key}/{val}.yaml")
                if sub_cfg_path.exists():
                    sub_cfg = OmegaConf.load(sub_cfg_path)
                    cfg = OmegaConf.merge(sub_cfg, cfg)
    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    train_cfg = cfg.train
    model_cfg = cfg.model
    data_cfg = cfg.data
    # 42 is the TrainingArguments default, so runs without train.seed are unchanged.
    seed = int(train_cfg.get("seed", 42))
    # Seed before the model is built so LoRA adapter initialization follows the seed
    # too; the Trainer re-seeds only after construction.
    from transformers import set_seed
    set_seed(seed)

    # ── Load model + processor ─────────────────────────────────────────────────
    lora_cfg = None
    if cfg.get("lora", {}).get("enabled", False):
        lora_cfg = OmegaConf.to_container(cfg.lora)

    generic = model_cfg.get("generic", False)
    if generic:
        # Any HF image-text-to-text VLM (Qwen2.5-VL, InternVL3, Llama-Vision, ...)
        from src.model.vlm import load_vlm_and_processor
        model, processor = load_vlm_and_processor(
            hf_id=model_cfg.hf_id,
            dtype=torch.bfloat16,
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            lora_config=lora_cfg,
            device_map=None,
        )
    else:
        from src.model.qwen2vl import load_model_and_processor
        model, processor = load_model_and_processor(
            hf_id=model_cfg.hf_id,
            dtype=torch.bfloat16,
            attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
            lora_config=lora_cfg,
            device_map=None,  # Accelerate handles device placement
        )

    # ── Load datasets ──────────────────────────────────────────────────────────
    logger.info("Loading training dataset from %s", data_cfg.unified_jsonl)
    train_records = load_jsonl_records(data_cfg.unified_jsonl)
    train_dataset = build_hf_dataset(train_records)
    logger.info("Training samples: %d", len(train_dataset))

    eval_dataset = None
    val_path = data_cfg.get("val_jsonl", "")
    if val_path and Path(val_path).exists():
        logger.info("Loading validation dataset from %s", val_path)
        val_records = load_jsonl_records(val_path)
        eval_dataset = build_hf_dataset(val_records)
        logger.info("Validation samples: %d", len(eval_dataset))

    # ── Data collator ──────────────────────────────────────────────────────────
    if generic:
        from src.data.vlm_collator import GenericVLMCollator
        collator = GenericVLMCollator(
            processor=processor,
            max_length=train_cfg.get("max_seq_length", 2048),
        )
    else:
        from src.data.collator import Qwen2VLCollator
        collator = Qwen2VLCollator(
            processor=processor,
            max_length=train_cfg.get("max_seq_length", 2048),
        )

    # ── Training arguments ─────────────────────────────────────────────────────
    from transformers import Trainer, TrainingArguments

    class NaNSafeTrainer(Trainer):
        """Skips optimizer steps whose gradients are non-finite.

        bf16 VLM training occasionally produces a NaN/Inf gradient on a single
        hard batch (attention overflow). Plain bf16 has no GradScaler, so the
        NaN gradient would be applied and poison all weights. We zero such
        gradients so the step is effectively skipped and training continues.
        """

        def training_step(self, *args, **kwargs):
            loss = super().training_step(*args, **kwargs)
            model = args[0]
            # after backward (grads populated for this micro-step), neutralise
            # non-finite grads so the upcoming optimizer step is a safe no-op
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = None
            return loss

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.lr,
        lr_scheduler_type=train_cfg.lr_scheduler,
        warmup_steps=train_cfg.warmup_steps,
        weight_decay=train_cfg.weight_decay,
        bf16=train_cfg.get("bf16", True),
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_cfg.get("optim", "adamw_torch"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=eval_dataset is not None,
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name", "ultrasound_vl"),
        dataloader_num_workers=data_cfg.get("num_workers", 4),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=seed,
    )

    trainer = NaNSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    resume = train_cfg.get("resume_from_checkpoint", None)
    logger.info("Starting training (resume=%s)...", resume or "no")
    trainer.train(resume_from_checkpoint=resume)

    # ── Save final adapter ─────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    logger.info("Saving final adapter to %s", final_dir)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
