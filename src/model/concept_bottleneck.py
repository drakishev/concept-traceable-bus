"""Concept bottleneck for the report model.

Instead of conditioning the decoder on the raw predicted embedding, we route
generation through the predicted clinical concepts only. All image information
that reaches the decoder must pass through interpretable concept predictions
(pathology, BI-RADS risk, BI-RADS category, ...). This makes every generated
report traceable to a small set of human-readable concepts and enables
concept-level intervention (force a concept, see the report change).

For each concept head h with C_h classes:
  - probs_h = softmax(logits_h)              [B, C_h]   (or one-hot if hard/intervened)
  - concept_vec_h = probs_h @ E_h            [B, concept_dim]   (soft embedding lookup)
Concatenate over heads and project to the decoder conditioning dim.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptBottleneck(nn.Module):
    def __init__(
        self,
        head_specs: dict[str, int],
        out_dim: int,
        concept_dim: int = 64,
        residual_dim: int = 0,
        residual_in_dim: int = 1024,
    ):
        super().__init__()
        self.head_names = list(head_specs.keys())
        self.embeds = nn.ModuleDict(
            {name: nn.Embedding(n_classes, concept_dim) for name, n_classes in head_specs.items()}
        )
        # B3: optional residual path - a small projection of the raw predicted
        # embedding concatenated with the concept embedding. residual_dim=0 is the
        # pure (fully interpretable) concept bottleneck.
        self.residual_dim = residual_dim
        if residual_dim > 0:
            self.residual_proj = nn.Sequential(
                nn.Linear(residual_in_dim, residual_dim), nn.GELU()
            )
        in_dim = concept_dim * len(head_specs) + residual_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        aux_logits: dict[str, torch.Tensor],
        hard: bool = False,
        overrides: dict[str, torch.Tensor] | None = None,
        temperature: float = 1.0,
        residual_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the decoder conditioning vector from concept predictions.

        Args:
            aux_logits: head_name -> [B, C_h] logits from the Predictor
            hard: if True, use argmax one-hot (discrete bottleneck) instead of soft probs
            overrides: head_name -> [B] class indices to FORCE (concept intervention)
            temperature: softmax temperature for the soft path

        Returns:
            [B, out_dim] conditioning vector
        """
        overrides = overrides or {}
        parts = []
        for name in self.head_names:
            emb = self.embeds[name]
            n_classes = emb.num_embeddings
            if name in overrides:
                probs = F.one_hot(overrides[name].long(), num_classes=n_classes)
                probs = probs.to(emb.weight.dtype)
            elif hard:
                idx = aux_logits[name].argmax(dim=-1)
                probs = F.one_hot(idx, num_classes=n_classes).to(emb.weight.dtype)
            else:
                probs = F.softmax(aux_logits[name] / temperature, dim=-1).to(emb.weight.dtype)
            parts.append(probs @ emb.weight)  # [B, concept_dim]
        if self.residual_dim > 0 and residual_embedding is not None:
            parts.append(self.residual_proj(residual_embedding))
        concat = torch.cat(parts, dim=-1)
        return self.proj(concat)  # [B, out_dim]
