"""Y-Decoder: Text generator conditioned on predicted embeddings.

Takes predicted embeddings from the Predictor and generates report text.
Only used during Stage 2 fine-tuning and inference - NOT during Stage 1 pretraining.

Supports small models (GPT-2, BioMedLM) with full fine-tuning and large models
(MedGemma-27B) with LoRA to keep optimizer state within GPU memory.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_LARGE_MODEL_PREFIXES = (
    "google/medgemma", "google/gemma",
    "meta-llama", "mistralai",
    "aaditya/llama3", "m42-health/llama3",  # biomedical Llama-3 70B variants
    "nousresearch/", "unsloth/",            # open Llama mirrors
    "qwen/",                                 # Qwen2.5 7B+ instruct decoders
    "biomistral/",                           # BioMistral-7B
    "epfl-llm/",                             # Meditron 7B/70B
)


def _is_large_model(model_name: str) -> bool:
    return any(model_name.lower().startswith(p) for p in _LARGE_MODEL_PREFIXES)


# Models that require 4-bit QLoRA to fit on a single H200 (140 GB)
_NEEDS_QLORA_PREFIXES = (
    "meta-llama/llama-3.1-70b", "meta-llama/llama-3.3-70b", "meta-llama/llama-3-70b",
    "nousresearch/meta-llama-3.1-70b", "unsloth/meta-llama-3.1-70b",
    "aaditya/llama3-openbiollm-70b", "m42-health/llama3-med42-70b",
    "epfl-llm/meditron-70b",
    "qwen/qwen2.5-72b",
)


def _needs_qlora(model_name: str) -> bool:
    n = model_name.lower()
    return any(n.startswith(p) for p in _NEEDS_QLORA_PREFIXES)


class YDecoder(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        shared_embed_dim: int = 768,
        num_soft_prompt_tokens: int = 8,
        max_length: int = 512,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        force_lora: bool = False,
    ):
        super().__init__()
        self.max_length = max_length
        self.num_soft_prompt_tokens = num_soft_prompt_tokens
        self.model_name = model_name

        # `force_lora` exists for the matched-condition decoder-scale sweep: by
        # default only models in _LARGE_MODEL_PREFIXES get LoRA, so a small
        # decoder (BioMedLM-2.7B) would be adapted differently from a 27B/70B
        # one, confounding "decoder size" with "adaptation method". Setting it
        # puts every size on the same LoRA protocol.
        large = _is_large_model(model_name) or force_lora
        use_qlora = _needs_qlora(model_name)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load LM:
        #  - tiny models     → fp32 (trainer casts later)
        #  - large bf16      → bf16 + device_map=auto (MedGemma-27B path)
        #  - 70B-class       → QLoRA: 4-bit NF4 base + LoRA adapters
        load_kwargs: dict = {}
        if use_qlora:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["device_map"] = "auto"
        elif large:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"

        self.lm = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.lm.config.pad_token_id = self.tokenizer.pad_token_id

        # apply LoRA for large models so optimizer state fits in memory
        if large:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            if use_qlora:
                # QLoRA: prepare frozen 4-bit base for adapter training
                self.lm = prepare_model_for_kbit_training(
                    self.lm, use_gradient_checkpointing=True
                )
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                bias="none",
            )
            self.lm = get_peft_model(self.lm, lora_cfg)
            self.lm.print_trainable_parameters()

        cfg = self.lm.config
        if hasattr(cfg, "n_embd"):
            lm_hidden_dim = cfg.n_embd
        elif hasattr(cfg, "hidden_size"):
            lm_hidden_dim = cfg.hidden_size
        elif hasattr(cfg, "text_config"):          # Gemma3 nests it under text_config
            lm_hidden_dim = cfg.text_config.hidden_size
        else:
            raise ValueError(f"Cannot determine hidden_size from {type(cfg)}")

        # embedding-to-prompt bridge: maps predicted embedding → soft prompt tokens
        self.embed_to_prompt = nn.Sequential(
            nn.Linear(shared_embed_dim, lm_hidden_dim * num_soft_prompt_tokens),
            nn.GELU(),
            nn.LayerNorm(lm_hidden_dim * num_soft_prompt_tokens),
        )
        self.lm_hidden_dim = lm_hidden_dim
        # multi-query path (B1): project each of K conditioning vectors to LM dim
        # and use them directly as K soft-prompt tokens.
        self.multi_query_proj = nn.Sequential(
            nn.Linear(shared_embed_dim, lm_hidden_dim), nn.GELU(),
            nn.LayerNorm(lm_hidden_dim),
        )

    def _make_soft_prompt(self, predicted_embedding: torch.Tensor) -> torch.Tensor:
        """Convert conditioning into soft-prompt tokens [B, K, D_LM].

        Accepts either [B, D] (single vector → bridge expands to K_soft tokens) or
        [B, K, D] (multi-query → one soft-prompt token per query vector).
        """
        if predicted_embedding.dim() == 3:
            return self.multi_query_proj(predicted_embedding)  # [B, K, D_LM]
        prompt_flat = self.embed_to_prompt(predicted_embedding)
        return prompt_flat.view(-1, self.num_soft_prompt_tokens, self.lm_hidden_dim)

    def _get_input_embeddings(self):
        """Works for both base models and PEFT-wrapped models."""
        model = self.lm.base_model if hasattr(self.lm, "base_model") else self.lm
        return model.get_input_embeddings()

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        target_text: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward (teacher forcing).

        Args:
            predicted_embedding: [B, shared_embed_dim]
            target_text: list of B target strings

        Returns:
            dict with 'loss' and 'logits'
        """
        B = predicted_embedding.shape[0]
        device = predicted_embedding.device

        soft_prompt = self._make_soft_prompt(predicted_embedding)  # [B, K, D]
        n_prompt = soft_prompt.shape[1]

        encoded = self.tokenizer(
            target_text,
            padding=True,
            truncation=True,
            max_length=self.max_length - n_prompt,
            return_tensors="pt",
        ).to(device)

        token_embeds = self._get_input_embeddings()(encoded["input_ids"])  # [B, T, D]

        # cast soft prompt to match token embedding dtype (important for bf16 models)
        soft_prompt = soft_prompt.to(token_embeds.dtype)

        inputs_embeds = torch.cat([soft_prompt, token_embeds], dim=1)  # [B, K+T, D]

        prompt_mask = torch.ones(B, n_prompt, device=device, dtype=encoded["attention_mask"].dtype)
        attention_mask = torch.cat([prompt_mask, encoded["attention_mask"]], dim=1)

        labels = encoded["input_ids"].clone()
        prompt_labels = torch.full((B, n_prompt), -100, device=device, dtype=labels.dtype)
        labels = torch.cat([prompt_labels, labels], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        predicted_embedding: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> list[str]:
        """Generate report text from predicted embedding."""
        device = predicted_embedding.device
        B = predicted_embedding.shape[0]

        soft_prompt = self._make_soft_prompt(predicted_embedding)  # [B, K, D]

        # cast to LM dtype
        lm_dtype = next(self.lm.parameters()).dtype
        soft_prompt = soft_prompt.to(lm_dtype)

        prompt_mask = torch.ones(B, soft_prompt.shape[1], device=device, dtype=torch.long)

        outputs = self.lm.generate(
            inputs_embeds=soft_prompt,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            # Targets are trained with unmasked padding, so the pad token is what
            # the decoder learns to emit after a report. For instruct tokenizers
            # (Qwen2.5: eos <|im_end|>, pad <|endoftext|>) stopping on eos alone
            # lets generation run on to max_new_tokens.
            eos_token_id=sorted({self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}),
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
