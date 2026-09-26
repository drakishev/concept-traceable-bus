"""Predictor: The main trainable component of VL-JEPA.

A bidirectional (non-causal) transformer that takes vision patch embeddings
and predicts the target text embedding in the shared latent space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PredictorBlock(nn.Module):
    """Transformer block with bidirectional (non-causal) attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attention with pre-norm (no causal mask — fully bidirectional)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x

        # feedforward with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Predictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        max_seq_len: int = 512,
        aux_head_specs: dict[str, int] | None = None,
        num_query_tokens: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token
        self.num_query_tokens = num_query_tokens

        # input projection (from x-encoder dim to predictor hidden dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # learnable [PRED] / query tokens for aggregating prediction (K=num_query_tokens)
        if use_cls_token:
            self.pred_token = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim) * 0.02)

        # positional embeddings (+K for the query tokens)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len + num_query_tokens, hidden_dim) * 0.02
        )

        # transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # output projection to shared embedding space
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # auxiliary classification heads on the pre-projection [PRED] embedding
        # aux_head_specs maps head_name -> num_classes, e.g. {"pathology": 2}
        self.aux_heads = nn.ModuleDict()
        if aux_head_specs:
            for name, n_classes in aux_head_specs.items():
                self.aux_heads[name] = nn.Linear(hidden_dim, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        vision_embeddings: torch.Tensor,
        return_aux: bool = False,
        return_tokens: bool = False,
    ):
        """Predict target text embedding(s) from vision features.

        Args:
            vision_embeddings: [B, N_patches, input_dim] from X-Encoder
            return_aux: if True, also return auxiliary classification logits
            return_tokens: if True, return all K query embeddings [B, K, output_dim]
                (multi-query bottleneck); otherwise the mean-pooled [B, output_dim]

        Returns:
            [B, output_dim] (or [B, K, output_dim] if return_tokens),
            optionally with a dict of head_name -> [B, n_classes] aux logits.
        """
        B, N, _ = vision_embeddings.shape
        K = self.num_query_tokens

        x = self.input_proj(vision_embeddings)  # [B, N, hidden_dim]

        if self.use_cls_token:
            pred_tokens = self.pred_token.expand(B, -1, -1)  # [B, K, hidden_dim]
            x = torch.cat([pred_tokens, x], dim=1)  # [B, K+N, hidden_dim]

        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        if self.use_cls_token:
            queries = x[:, :K, :]            # [B, K, hidden_dim]
        else:
            queries = x.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]

        pooled_hidden = queries.mean(dim=1)  # [B, hidden_dim] for aux heads
        tokens = self.output_proj(queries)   # [B, K, output_dim]
        pooled_out = tokens.mean(dim=1)      # [B, output_dim] for InfoNCE/alignment

        out = tokens if return_tokens else pooled_out
        if return_aux:
            aux_logits = {name: head(pooled_hidden) for name, head in self.aux_heads.items()}
            return out, aux_logits
        return out
