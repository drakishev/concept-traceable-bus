"""Qwen2-VL model wrapper with LoRA.

Loads Qwen2-VL-7B-Instruct from HuggingFace, applies PEFT LoRA adapters,
and exposes a clean generate() interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_model_and_processor(
    hf_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    lora_config: dict[str, Any] | None = None,
    adapter_path: str | Path | None = None,
    device_map: str = "auto",
) -> tuple[Any, Any]:
    """Load Qwen2-VL model + processor, optionally with LoRA.

    Args:
        hf_id: HuggingFace model ID.
        dtype: Model dtype (bfloat16 recommended for H200).
        attn_implementation: "flash_attention_2" or "sdpa".
        lora_config: If provided, apply LoRA with these settings (dict of peft.LoraConfig args).
        adapter_path: Path to trained LoRA adapter weights to load.
        device_map: "auto" for multi-GPU, or "cuda:0" for single GPU.

    Returns:
        (model, processor) tuple ready for inference or training.
    """
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    logger.info("Loading Qwen2-VL processor from %s", hf_id)
    # Cap visual tokens: Qwen2-VL's default max_pixels (~12.8M) can expand a
    # single high-res ultrasound frame into thousands of visual tokens, which
    # overflows bf16 attention and produces NaN gradients on hard batches.
    # 768x768 ≈ 768 visual tokens is ample for breast ultrasound.
    processor = AutoProcessor.from_pretrained(
        hf_id, trust_remote_code=True,
        min_pixels=256 * 28 * 28,   # ~200k px floor
        max_pixels=768 * 768,       # ~590k px ceiling
    )

    logger.info("Loading Qwen2-VL model from %s (dtype=%s, attn=%s)",
                hf_id, dtype, attn_implementation)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=True,
    )

    if adapter_path is not None:
        # Load trained LoRA weights (inference mode)
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # merge for faster inference
    elif lora_config is not None:
        # Apply LoRA for training
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("rank", 64),
            lora_alpha=lora_config.get("alpha", 128),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, peft_config)
        # Gradient checkpointing on a LoRA model (frozen base) requires the
        # input embeddings to produce gradients, otherwise no gradient flows
        # through the checkpointed activations → NaN grad_norm / loss=0.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    return model, processor


def format_conversation(
    conversations: list[dict[str, str]],
    processor: Any,
    image: Image.Image | None = None,
    add_generation_prompt: bool = False,
) -> dict[str, torch.Tensor]:
    """Format a conversation list into model inputs.

    Args:
        conversations: List of {"role": ..., "content": ...} dicts.
        processor: Qwen2-VL processor.
        image: PIL image (required if "<image>" token in conversation).
        add_generation_prompt: True for inference (appends the assistant-turn
            cue so the model starts generating). Must be False for training,
            where the assistant text is part of the labelled sequence.

    Returns:
        Tokenized inputs dict with input_ids, attention_mask, pixel_values, etc.
    """
    # Build messages in Qwen2-VL chat format
    messages = []
    for turn in conversations:
        role = turn["role"]
        content = turn["content"]
        if "<image>" in content and image is not None:
            # Replace placeholder with proper image content format
            text_parts = content.split("<image>")
            content_list: list[dict] = []
            for j, part in enumerate(text_parts):
                if j > 0:
                    content_list.append({"type": "image", "image": image})
                if part.strip():
                    content_list.append({"type": "text", "text": part})
            messages.append({"role": role, "content": content_list})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=add_generation_prompt)
    images = [image] if image is not None else []

    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    )
    return inputs


@torch.inference_mode()
def generate_report(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str = "Describe the breast ultrasound findings and provide a BI-RADS assessment.",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate a diagnostic report for a single image.

    Args:
        model: Loaded Qwen2-VL model.
        processor: Qwen2-VL processor.
        image: Input PIL image.
        prompt: User prompt text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to use sampling.

    Returns:
        Generated report as a string.
    """
    conversations = [{"role": "user", "content": f"<image>\n{prompt}"}]
    inputs = format_conversation(conversations, processor, image=image, add_generation_prompt=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()
