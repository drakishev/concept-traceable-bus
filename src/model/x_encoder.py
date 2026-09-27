"""X-Encoder: Frozen vision backbone for image feature extraction.

Wraps pretrained pathology/medical ViTs (UNI2-h, or generic timm models).
All backbone parameters are frozen - this is a pure feature extractor.

UNI2-h loading note:
    timm.create_model('hf-hub:MahmoodLab/UNI2-h') fails because the hub
    config.json only has metadata and does not specify the architecture
    overrides (embed_dim=1536, SwiGLU MLP, 8 register tokens, layer scale).
    We detect this model and build it from scratch with the correct params
    before loading the cached weights.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Exact architecture parameters inferred from the UNI2-h checkpoint.
_UNI2H_CFG = dict(
    img_size=224,
    patch_size=14,
    num_classes=0,
    embed_dim=1536,
    depth=24,
    num_heads=24,
    mlp_ratio=8192 / 1536,   # GluMlp: fc1→8192 (=2×4096), fc2: 4096→1536
    no_embed_class=True,      # pos_embed has spatial-only tokens (256), no CLS
    reg_tokens=8,             # 8 register tokens
    init_values=1e-5,         # enables layer scale (ls1, ls2)
    dynamic_img_size=True,
)


# HF-transformers ViTs that are NOT timm models (no 'architecture' key in their
# hub config, so timm.create_model('hf-hub:...') raises KeyError). We load them
# with transformers.AutoModel and adapt them to the timm forward_features API.
_HF_VIT_PREFIXES = (
    "microsoft/rad-dino",
    "google/siglip2",
    "google/medsiglip",
)


def _is_hf_vit(model_name: str) -> bool:
    n = model_name.lower()
    return n.startswith("hf-vit:") or any(n.startswith(p) for p in _HF_VIT_PREFIXES)


class _HFViTWrapper(nn.Module):
    """Adapts a transformers vision model to the timm `.forward_features` API.

    Returns patch tokens [B, N, D] from `last_hidden_state`. Handles models that
    nest the vision tower under `.vision_model` (e.g. SigLIP2).
    """

    def __init__(self, model_name: str, freeze: bool):
        super().__init__()
        from transformers import AutoModel

        name = model_name.split("hf-vit:", 1)[-1]
        model = AutoModel.from_pretrained(name)
        # SigLIP/SigLIP2 expose the vision tower under .vision_model
        if hasattr(model, "vision_model") and not hasattr(model, "embeddings"):
            model = model.vision_model
        self.model = model
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        return out.last_hidden_state  # [B, N, D]


#: USFM (Jiao et al., MedIA 2024): the only public ultrasound-pretrained foundation
#: encoder. The released checkpoint is a BEiT-style ViT-B/16 at 224 px with a
#: shared relative-position bias, layer scale and q/v biases, which is exactly
#: timm's `beit_base_patch16_224` with the shared-bias options.
_USFM_PREFIX = "usfm:"


def _load_usfm(weights_path: str, freeze: bool) -> nn.Module:
    """Build timm BEiT-B/16 and load the USFM state dict from a local file."""
    # global_pool="" keeps the checkpoint's final `norm` (timm swaps it for
    # `fc_norm` under average pooling), so forward_features matches training.
    backbone = timm.create_model(
        "beit_base_patch16_224", pretrained=False, num_classes=0, global_pool="",
        use_shared_rel_pos_bias=True, use_rel_pos_bias=False, use_abs_pos_emb=False,
    )
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd.get("model", sd))
    # `mask_token` belongs to the MIM pretraining head; the bias index is a buffer
    # timm rebuilds from the grid size.
    sd = {k: v for k, v in sd.items()
          if k not in ("mask_token", "rel_pos_bias.relative_position_index")}
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if not m.startswith("head")], (
        f"USFM load failed - missing={missing}, unexpected={unexpected}"
    )
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
    return backbone


def _load_uni2h(freeze: bool) -> nn.Module:
    """Build VisionTransformer with UNI2-h exact architecture and load weights."""
    import torch.nn as tnn
    from huggingface_hub import hf_hub_download
    from timm.layers import GluMlp
    from timm.models.vision_transformer import VisionTransformer

    backbone = VisionTransformer(mlp_layer=GluMlp, act_layer=tnn.SiLU, **_UNI2H_CFG)

    # Download (or use cached) weights
    ckpt_path = hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = backbone.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (
        f"UNI2-h load failed - missing={missing}, unexpected={unexpected}"
    )

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

    return backbone


class XEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "hf-hub:MahmoodLab/UNI2-h",
        embed_dim: int = 1536,
        shared_embed_dim: int = 768,
        image_size: int = 224,
        freeze: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_embed_dim = shared_embed_dim

        if "MahmoodLab/UNI2-h" in model_name:
            self.backbone = _load_uni2h(freeze=freeze)
        elif model_name.startswith(_USFM_PREFIX):
            self.backbone = _load_usfm(model_name[len(_USFM_PREFIX):], freeze=freeze)
        elif _is_hf_vit(model_name):
            self.backbone = _HFViTWrapper(model_name, freeze=freeze)
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
            )
            if freeze:
                self._freeze_backbone()

        # projection to shared embedding space (if dimensions differ)
        if embed_dim != shared_embed_dim:
            self.projection = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, shared_embed_dim),
            )
        else:
            self.projection = nn.Identity()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters. Projection stays trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_last_n(self, n: int) -> list[torch.nn.Parameter]:
        """Re-enable gradients on the last `n` transformer blocks (+ final norm).

        Used by the partial-fine-tuning ablation, which asks whether the encoder
        ranking reflects pretraining quality or merely frozen-feature mismatch:
        if adapting one block closes the gap, the finding is about adaptation.

        Returns the newly trainable parameters so the caller can give them their
        own (typically reduced-LR) optimizer group. Returns [] when n <= 0.
        """
        if n <= 0:
            return []
        # timm ViT exposes .blocks; the HF wrapper nests under .model.encoder.layer
        inner = getattr(self.backbone, "model", self.backbone)
        blocks = getattr(inner, "blocks", None)
        if blocks is None:
            enc = getattr(inner, "encoder", None)
            blocks = getattr(enc, "layer", None) if enc is not None else None
        if blocks is None:
            raise ValueError(
                f"Cannot locate transformer blocks on {type(inner).__name__} to unfreeze"
            )

        params: list[torch.nn.Parameter] = []
        for block in list(blocks)[-n:]:
            for p in block.parameters():
                p.requires_grad = True
                params.append(p)
        for norm_attr in ("norm", "layernorm", "ln_post"):
            norm = getattr(inner, norm_attr, None)
            if isinstance(norm, nn.Module):
                for p in norm.parameters():
                    p.requires_grad = True
                    params.append(p)
                break
        self._unfrozen_params = params
        self.backbone.train()  # the unfrozen blocks need train-mode dropout
        return params

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features.

        Gradients are suppressed only when the backbone is fully frozen. When
        `unfreeze_last_n` has been called the graph must be kept so the unfrozen
        blocks receive gradients, so the no_grad context is applied conditionally
        rather than as a decorator.

        Args:
            x: [B, 3, H, W] image tensor

        Returns:
            [B, N_patches, embed_dim] patch embeddings
        """
        if getattr(self, "_unfrozen_params", None):
            features = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                features = self.backbone.forward_features(x)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project vision features.

        Args:
            x: [B, 3, H, W] image tensor

        Returns:
            [B, N_patches, shared_embed_dim] projected patch embeddings
        """
        features = self.forward_features(x)       # [B, N, embed_dim]
        projected = self.projection(features)      # [B, N, shared_embed_dim]
        return projected

    def get_num_patches(self, image_size: int = 224) -> int:
        """Calculate number of patches for a given image size."""
        dummy = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            features = self.backbone.forward_features(dummy)
        return features.shape[1] if features.ndim == 3 else 1
