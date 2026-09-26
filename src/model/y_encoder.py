"""Y-Encoder: Text target encoder.

Maps report text into a continuous embedding in the shared latent space.
This is the target that the Predictor learns to predict.
Trained with a reduced learning rate (0.05x) for stability.

Supports two backbones:
    - BERT-style (HuggingFace transformers): mean-pooled token embeddings
    - CLIP-style (open_clip, e.g. BiomedCLIP): pre-aligned text tower
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer


def _is_clip_model(model_name: str) -> bool:
    """CLIP-style models loaded via open_clip live under HF hub paths like
    'microsoft/BiomedCLIP-...'."""
    return "clip" in model_name.lower() or model_name.startswith("hf-hub:")


class YEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        embed_dim: int = 768,
        shared_embed_dim: int = 768,
        max_length: int = 512,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.is_clip = _is_clip_model(model_name)

        if self.is_clip:
            self._init_clip(model_name)
        else:
            self._init_bert(model_name)

        # projection to shared embedding space
        actual_dim = self.embed_dim
        if actual_dim != shared_embed_dim:
            self.projection = nn.Sequential(
                nn.LayerNorm(actual_dim),
                nn.Linear(actual_dim, shared_embed_dim),
            )
        else:
            self.projection = nn.Identity()

        if freeze_base:
            for param in (self._base_modules()):
                param.requires_grad = False

    # ── BERT path ─────────────────────────────────────────────────────────
    def _init_bert(self, model_name: str) -> None:
        # BioBERT-large-cased lacks tokenizer.json + model_type. Use Bert* directly.
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.encoder = BertModel.from_pretrained(model_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.encoder = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.encoder.config.hidden_size

    # ── CLIP path (open_clip) ─────────────────────────────────────────────
    def _init_clip(self, model_name: str) -> None:
        import open_clip
        # strip 'hf-hub:' prefix if present, open_clip handles it natively
        clip_name = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_name)
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        # we only need the text tower
        self.encoder = clip_model
        # discover output dim by running a probe forward
        with torch.no_grad():
            probe = self.tokenizer(["probe"])
            feat = clip_model.encode_text(probe)
        self.embed_dim = feat.shape[-1]

    def _base_modules(self):
        return self.encoder.parameters()

    # ── Shared API ────────────────────────────────────────────────────────
    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode report texts into [B, shared_embed_dim] L2-normalized embeddings."""
        device = next(self.encoder.parameters()).device

        if self.is_clip:
            tokens = self.tokenizer(texts).to(device)
            # open_clip's encode_text returns [B, embed_dim] already pooled
            pooled = self.encoder.encode_text(tokens)
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            outputs = self.encoder(**encoded)
            pooled = self._mean_pooling(outputs, encoded["attention_mask"])

        embeddings = self.projection(pooled.float())
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings
