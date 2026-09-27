"""ConceptReportModel: Full the report model model for histopathology report generation.

Combines X-Encoder, Predictor, Y-Encoder, and Y-Decoder into a single
LightningModule with two training stages:
  Stage 1 (pretrain): InfoNCE alignment in embedding space
  Stage 2 (finetune): Decoder fine-tuning for text generation
"""
from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
import torch.nn as nn

from src.losses.info_nce import BidirectionalInfoNCE
from src.model.concept_bottleneck import ConceptBottleneck
from src.model.predictor import Predictor
from src.model.x_encoder import XEncoder
from src.model.y_decoder import YDecoder
from src.model.y_encoder import YEncoder

log = logging.getLogger(__name__)


class ConceptReportModel(L.LightningModule):
    def __init__(
        self,
        # model config
        shared_embed_dim: int = 768,
        # x-encoder
        x_encoder_name: str = "hf-hub:MahmoodLab/UNI2-h",
        x_encoder_embed_dim: int = 1536,
        x_encoder_image_size: int = 224,
        x_encoder_freeze: bool = True,
        x_encoder_unfreeze_last_n: int = 0,
        # predictor
        predictor_num_layers: int = 8,
        predictor_hidden_dim: int = 1024,
        predictor_num_heads: int = 16,
        predictor_mlp_ratio: float = 4.0,
        predictor_dropout: float = 0.1,
        predictor_use_cls_token: bool = True,
        predictor_num_query_tokens: int = 1,
        # y-encoder
        y_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        y_encoder_embed_dim: int = 768,
        y_encoder_max_length: int = 512,
        y_encoder_freeze_base: bool = False,
        y_encoder_lr_multiplier: float = 0.05,
        # y-decoder
        y_decoder_name: str = "gpt2",
        y_decoder_max_length: int = 512,
        y_decoder_num_soft_prompt_tokens: int = 8,
        y_decoder_lora_rank: int = 16,
        y_decoder_lora_alpha: int = 32,
        y_decoder_lora_dropout: float = 0.05,
        y_decoder_force_lora: bool = False,
        # training
        stage: str = "pretrain",
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        max_steps: int = -1,
        min_lr_ratio: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        # multi-task auxiliary heads on the Predictor
        # aux_heads: list of slot names to add classification heads for, e.g.
        # ["pathology", "risk", "birads"]. Empty/None = no aux heads.
        aux_heads: list[str] | None = None,
        aux_loss_weight: float = 0.1,
        # concept bottleneck: route decoder conditioning through predicted concepts only
        concept_bottleneck: bool = False,
        concept_dim: int = 64,
        concept_hard: bool = False,
        concept_residual_dim: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.stage = stage

        # QLoRA (4-bit) decoders regenerate bitsandbytes quant buffers
        # (absmax, quant_map, quant_state.*) deterministically when the frozen
        # base is re-loaded. These appear as extra keys in the checkpoint, so
        # we must load non-strict to avoid a state_dict mismatch. Lightning's
        # load_from_checkpoint honours this attribute.
        self.strict_loading = False

        # --- build model components ---

        self.x_encoder = XEncoder(
            model_name=x_encoder_name,
            embed_dim=x_encoder_embed_dim,
            shared_embed_dim=shared_embed_dim,
            image_size=x_encoder_image_size,
            freeze=x_encoder_freeze,
        )

        # build aux head specs from configured aux_heads list
        from src.data.slot_labels import NUM_CLASSES as SLOT_NUM_CLASSES
        aux_head_specs: dict[str, int] = {}
        if aux_heads:
            for name in aux_heads:
                if name not in SLOT_NUM_CLASSES:
                    raise ValueError(
                        f"Unknown aux head '{name}'. "
                        f"Valid: {list(SLOT_NUM_CLASSES.keys())}"
                    )
                aux_head_specs[name] = SLOT_NUM_CLASSES[name]
        self.aux_heads = list(aux_head_specs.keys())
        self.aux_loss_weight = aux_loss_weight

        self.predictor = Predictor(
            input_dim=shared_embed_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=shared_embed_dim,
            num_layers=predictor_num_layers,
            num_heads=predictor_num_heads,
            mlp_ratio=predictor_mlp_ratio,
            dropout=predictor_dropout,
            use_cls_token=predictor_use_cls_token,
            aux_head_specs=aux_head_specs,
            num_query_tokens=predictor_num_query_tokens,
        )
        self.predictor_num_query_tokens = predictor_num_query_tokens

        # concept bottleneck: decoder conditioning passes ONLY through concepts
        self.concept_bottleneck_enabled = concept_bottleneck
        self.concept_hard = concept_hard
        if concept_bottleneck:
            if not aux_head_specs:
                raise ValueError("concept_bottleneck=True requires aux_heads to be set")
            self.concept_bottleneck = ConceptBottleneck(
                head_specs=aux_head_specs,
                out_dim=shared_embed_dim,
                concept_dim=concept_dim,
                residual_dim=concept_residual_dim,
                residual_in_dim=shared_embed_dim,
            )
            self.concept_residual_dim = concept_residual_dim

        self.y_encoder = YEncoder(
            model_name=y_encoder_name,
            embed_dim=y_encoder_embed_dim,
            shared_embed_dim=shared_embed_dim,
            max_length=y_encoder_max_length,
            freeze_base=y_encoder_freeze_base,
        )

        self.y_decoder = YDecoder(
            model_name=y_decoder_name,
            shared_embed_dim=shared_embed_dim,
            num_soft_prompt_tokens=y_decoder_num_soft_prompt_tokens,
            max_length=y_decoder_max_length,
            lora_rank=y_decoder_lora_rank,
            lora_alpha=y_decoder_lora_alpha,
            lora_dropout=y_decoder_lora_dropout,
            force_lora=y_decoder_force_lora,
        )

        # loss for stage 1
        self.info_nce = BidirectionalInfoNCE(
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

        # configure for the appropriate training stage
        self._configure_stage(stage)

    def _configure_stage(self, stage: str) -> None:
        """Freeze/unfreeze components based on training stage."""
        if stage == "pretrain":
            # Stage 1: train predictor + y_encoder, freeze x_encoder + y_decoder
            self.x_encoder.eval()
            for p in self.x_encoder.parameters():
                p.requires_grad = False
            for p in self.y_decoder.parameters():
                p.requires_grad = False
            # Partial encoder fine-tuning (ablation): re-enable the last N blocks
            # AFTER the blanket freeze above, so they survive it.
            n_unfreeze = int(getattr(self.hparams, "x_encoder_unfreeze_last_n", 0) or 0)
            if n_unfreeze > 0:
                unfrozen = self.x_encoder.unfreeze_last_n(n_unfreeze)
                log.info("Unfroze last %d x-encoder block(s): %d params",
                         n_unfreeze, sum(p.numel() for p in unfrozen))
            # predictor and y_encoder are trainable (y_encoder at reduced LR)
        elif stage == "finetune":
            # Stage 2: train y_decoder, freeze everything else
            # (optionally keep predictor at very low LR)
            self.x_encoder.eval()
            for p in self.x_encoder.parameters():
                p.requires_grad = False
            for p in self.predictor.parameters():
                p.requires_grad = False
            for p in self.y_encoder.parameters():
                p.requires_grad = False
            # unfreeze decoder - respect LoRA freezing (only LoRA params trainable)
            # calling requires_grad=True on all would break LoRA's selective freezing,
            # so only unfreeze params that were trainable before stage 1 froze them.
            for name, p in self.y_decoder.named_parameters():
                # LoRA adapter params have 'lora_' in their name; always trainable
                # embed_to_prompt bridge is always trainable
                # base model params stay frozen (no 'lora_' prefix)
                if "lora_" in name or "embed_to_prompt" in name:
                    p.requires_grad = True
            # enable gradient checkpointing to reduce activation memory for large LMs
            if hasattr(self.y_decoder.lm, "gradient_checkpointing_enable"):
                self.y_decoder.lm.gradient_checkpointing_enable()

    def forward_embedding(
        self, images: torch.Tensor, texts: list[str],
        return_aux: bool = False,
    ):
        """Forward pass for embedding alignment (Stage 1).

        Returns:
            (predicted_embedding, target_embedding) both [B, shared_embed_dim],
            (optionally) dict of head_name -> logits [B, n_classes]
        """
        # vision path: image → x_encoder → predictor → s_y_hat
        vision_features = self.x_encoder(images)  # [B, N, D]

        if return_aux:
            predicted_embedding, aux_logits = self.predictor(vision_features, return_aux=True)
        else:
            predicted_embedding = self.predictor(vision_features)
            aux_logits = None

        # L2 normalize
        predicted_embedding = nn.functional.normalize(predicted_embedding, p=2, dim=-1)

        # text path: text → y_encoder → s_y
        target_embedding = self.y_encoder(texts)  # [B, D] (already normalized)

        if return_aux:
            return predicted_embedding, target_embedding, aux_logits
        return predicted_embedding, target_embedding

    def _compute_aux_loss(
        self, aux_logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute summed cross-entropy across all configured aux heads.

        Labels with value -100 are skipped by cross_entropy automatically.
        Returns (total_aux_loss, per_head_loss_dict).
        """
        per_head = {}
        total = torch.zeros((), device=next(self.parameters()).device)
        n_heads_with_signal = 0
        for name, logits in aux_logits.items():
            target = labels[name].to(logits.device)
            head_loss = nn.functional.cross_entropy(logits, target, ignore_index=-100)
            # cross_entropy returns NaN when all targets are -100 in a batch; skip safely
            if torch.isfinite(head_loss):
                per_head[name] = head_loss.detach()
                total = total + head_loss
                n_heads_with_signal += 1
        if n_heads_with_signal > 0:
            total = total / n_heads_with_signal
        return total, per_head

    def _decoder_conditioning(
        self,
        vision_features: torch.Tensor,
        overrides: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Vector fed to the Y-Decoder.

        With concept bottleneck: route through predicted concepts only
        (the predictor / aux logits are computed, then the concept embedding
        is built - this is the ONLY information the decoder sees).
        Without: the L2-normalised predicted embedding.
        """
        if self.concept_bottleneck_enabled:
            predicted, aux_logits = self.predictor(vision_features, return_aux=True)
            residual = nn.functional.normalize(predicted, p=2, dim=-1) \
                if getattr(self, "concept_residual_dim", 0) > 0 else None
            return self.concept_bottleneck(
                aux_logits, hard=self.concept_hard, overrides=overrides,
                residual_embedding=residual,
            )
        if getattr(self, "predictor_num_query_tokens", 1) > 1:
            # multi-query bottleneck: feed K token embeddings to the decoder
            tokens = self.predictor(vision_features, return_tokens=True)  # [B, K, D]
            return nn.functional.normalize(tokens, p=2, dim=-1)
        predicted = self.predictor(vision_features)
        return nn.functional.normalize(predicted, p=2, dim=-1)

    def forward_generate(
        self, images: torch.Tensor,
        overrides: dict[str, torch.Tensor] | None = None,
        **generate_kwargs: Any,
    ) -> list[str]:
        """Forward pass for text generation (inference).

        Args:
            images: [B, 3, H, W]
            overrides: optional concept intervention (head_name -> [B] class idx),
                only used when concept_bottleneck is enabled.

        Returns:
            list of generated report strings
        """
        vision_features = self.x_encoder(images)
        cond = self._decoder_conditioning(vision_features, overrides=overrides)
        return self.y_decoder.generate(cond, **generate_kwargs)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        texts = batch["report_text"]

        if self.stage == "pretrain":
            if self.aux_heads:
                predicted_emb, target_emb, aux_logits = self.forward_embedding(
                    images, texts, return_aux=True
                )
                info_nce_loss = self.info_nce(predicted_emb, target_emb)
                aux_loss, per_head = self._compute_aux_loss(aux_logits, batch["labels"])
                loss = info_nce_loss + self.aux_loss_weight * aux_loss
                self.log("train/loss", loss, prog_bar=True, sync_dist=True)
                self.log("train/info_nce", info_nce_loss, prog_bar=False, sync_dist=True)
                self.log("train/aux", aux_loss, prog_bar=False, sync_dist=True)
                for name, head_loss in per_head.items():
                    self.log(f"train/aux_{name}", head_loss, prog_bar=False, sync_dist=True)
            else:
                predicted_emb, target_emb = self.forward_embedding(images, texts)
                loss = self.info_nce(predicted_emb, target_emb)
                self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/temperature", self.info_nce.temperature_value, prog_bar=False)
        elif self.stage == "finetune":
            # frozen encoder + predictor (no grad); the concept bottleneck (if
            # enabled) IS trainable, so compute conditioning outside no_grad.
            with torch.no_grad():
                vision_features = self.x_encoder(images)
                if self.concept_bottleneck_enabled:
                    pred_raw, aux_logits = self.predictor(vision_features, return_aux=True)
                    residual = nn.functional.normalize(pred_raw, p=2, dim=-1) \
                        if getattr(self, "concept_residual_dim", 0) > 0 else None
                elif getattr(self, "predictor_num_query_tokens", 1) > 1:
                    predicted_emb = nn.functional.normalize(
                        self.predictor(vision_features, return_tokens=True), p=2, dim=-1
                    )  # [B, K, D]
                else:
                    predicted_emb = nn.functional.normalize(
                        self.predictor(vision_features), p=2, dim=-1
                    )
            if self.concept_bottleneck_enabled:
                cond = self.concept_bottleneck(
                    aux_logits, hard=self.concept_hard, residual_embedding=residual
                )
            else:
                cond = predicted_emb
            # train decoder (+ concept bottleneck)
            decoder_out = self.y_decoder(cond, target_text=texts)
            loss = decoder_out["loss"]
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        texts = batch["report_text"]

        if self.stage == "pretrain":
            if self.aux_heads:
                predicted_emb, target_emb, aux_logits = self.forward_embedding(
                    images, texts, return_aux=True
                )
                info_nce_loss = self.info_nce(predicted_emb, target_emb)
                aux_loss, per_head = self._compute_aux_loss(aux_logits, batch["labels"])
                loss = info_nce_loss + self.aux_loss_weight * aux_loss
                self.log("val/loss", loss, prog_bar=True, sync_dist=True)
                self.log("val/info_nce", info_nce_loss, prog_bar=False, sync_dist=True)
                self.log("val/aux", aux_loss, prog_bar=False, sync_dist=True)
                # report per-head accuracy on labelled subset of val
                for name, logits in aux_logits.items():
                    target = batch["labels"][name].to(logits.device)
                    mask = target != -100
                    if mask.any():
                        preds = logits.argmax(-1)
                        acc = (preds[mask] == target[mask]).float().mean()
                        self.log(f"val/aux_acc_{name}", acc, prog_bar=False, sync_dist=True)
            else:
                predicted_emb, target_emb = self.forward_embedding(images, texts)
                loss = self.info_nce(predicted_emb, target_emb)
                self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        elif self.stage == "finetune":
            vision_features = self.x_encoder(images)
            cond = self._decoder_conditioning(vision_features)
            decoder_out = self.y_decoder(cond, target_text=texts)
            loss = decoder_out["loss"]
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer with per-component learning rates."""
        hp = self.hparams

        if self.stage == "pretrain":
            param_groups = [
                {
                    "params": [p for p in self.predictor.parameters() if p.requires_grad],
                    "lr": hp.lr,
                    "name": "predictor",
                },
                {
                    "params": [p for p in self.y_encoder.parameters() if p.requires_grad],
                    "lr": hp.lr * hp.y_encoder_lr_multiplier,
                    "name": "y_encoder",
                },
            ]
            # partial encoder fine-tuning: its own reduced-LR group, mirroring
            # the y_encoder multiplier so the pretrained backbone is nudged, not
            # overwritten.
            x_enc_trainable = [p for p in self.x_encoder.parameters() if p.requires_grad]
            if x_enc_trainable:
                param_groups.append({
                    "params": x_enc_trainable,
                    "lr": hp.lr * hp.y_encoder_lr_multiplier,
                    "name": "x_encoder_partial",
                })
            # include info_nce learnable temperature if applicable
            if hp.learnable_temperature:
                param_groups.append({
                    "params": [self.info_nce.log_temperature],
                    "lr": hp.lr,
                    "name": "temperature",
                })
        elif self.stage == "finetune":
            param_groups = [
                {
                    "params": [p for p in self.y_decoder.parameters() if p.requires_grad],
                    "lr": hp.lr,
                    "name": "y_decoder",
                },
            ]
            # concept bottleneck (concept embeddings + projection) trains in stage 2
            if self.concept_bottleneck_enabled:
                param_groups.append({
                    "params": [p for p in self.concept_bottleneck.parameters() if p.requires_grad],
                    "lr": hp.lr,
                    "name": "concept_bottleneck",
                })
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        # filter out empty groups
        param_groups = [g for g in param_groups if len(list(g["params"])) > 0]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            betas=tuple(hp.betas) if isinstance(hp.betas, list) else hp.betas,
        )

        # cosine schedule with linear warmup
        from src.training.scheduler import get_cosine_schedule_with_warmup

        total_steps = hp.max_steps if hp.max_steps > 0 else self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=hp.warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=hp.min_lr_ratio,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
