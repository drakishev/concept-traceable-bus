"""Generic vision-language model loader for end-to-end SFT baselines.

Loads any HuggingFace image-text-to-text model (Qwen2.5-VL, InternVL3,
Llama-3.2-Vision, MedGemma, ...) via the unified `AutoModelForImageTextToText`
+ `AutoProcessor` interface, optionally applying PEFT LoRA. This is the
end-to-end counterpart to the latent-conditioned pipeline - the image is attended
to directly by the decoder rather than squeezed through a predicted embedding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]


def load_vlm_and_processor(
    hf_id: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    lora_config: dict[str, Any] | None = None,
    adapter_path: str | Path | None = None,
    device_map: str | None = None,
    max_pixels: int = 768 * 768,
    min_pixels: int = 256 * 28 * 28,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a generic VLM + processor, optionally with LoRA.

    Caps visual tokens for processors that support it (Qwen family) to avoid
    bf16 attention overflow on high-res ultrasound frames. Prefers native
    transformers processors (trust_remote_code=False) so the unified processor
    API (.tokenizer, apply_chat_template) is available.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Some processors (Qwen2.5-VL) accept min/max_pixels; others reject them.
    try:
        processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
    except (TypeError, ValueError):
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and getattr(tok, "pad_token_id", None) is None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass

    logger.info("Loading VLM %s (dtype=%s, attn=%s)", hf_id, dtype, attn_implementation)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    if adapter_path is not None:
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
        return model, processor

    if lora_config is not None:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("rank", 64),
            lora_alpha=lora_config.get("alpha", 128),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            target_modules=lora_config.get("target_modules", DEFAULT_LORA_TARGETS),
        )
        model = get_peft_model(model, peft_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    return model, processor
