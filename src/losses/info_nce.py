"""Bidirectional InfoNCE loss for embedding alignment.

Computes symmetric contrastive loss between predicted and target embeddings:
    L = 0.5 * (InfoNCE(pred→target) + InfoNCE(target→pred))

This encourages:
  - Alignment: matching image-text pairs have similar embeddings
  - Uniformity: embeddings spread across the hypersphere (prevents collapse)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalInfoNCE(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        import math

        log_temp_init = -math.log(temperature)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(log_temp_init))
        else:
            self.register_buffer("log_temperature", torch.tensor(log_temp_init))

    @property
    def temperature_value(self) -> float:
        """Current effective temperature."""
        return torch.exp(-self.log_temperature).item()

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bidirectional InfoNCE loss.

        Args:
            predicted: [B, D] L2-normalized predicted embeddings (from Predictor)
            target: [B, D] L2-normalized target embeddings (from Y-Encoder)

        Returns:
            scalar loss
        """
        # temperature-scaled similarity matrix
        logit_scale = self.log_temperature.exp()  # = 1/τ
        logits = logit_scale * (predicted @ target.T)  # [B, B]

        # labels: diagonal entries are positives
        labels = torch.arange(logits.shape[0], device=logits.device)

        # bidirectional cross-entropy
        loss_pred_to_target = F.cross_entropy(logits, labels)
        loss_target_to_pred = F.cross_entropy(logits.T, labels)

        loss = 0.5 * (loss_pred_to_target + loss_target_to_pred)

        return loss


class InfoNCEWithGather(BidirectionalInfoNCE):
    """InfoNCE with all-gather for distributed training.

    Gathers embeddings from all GPUs to compute the loss over
    a larger effective batch size - critical for contrastive learning.
    """

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.distributed.is_initialized():
            return super().forward(predicted, target)

        # all-gather embeddings from all ranks
        predicted_all = self._all_gather(predicted)
        target_all = self._all_gather(target)

        logit_scale = self.log_temperature.exp()
        logits = logit_scale * (predicted @ target_all.T)  # [B_local, B_global]

        # offset labels by rank * local_batch_size
        rank = torch.distributed.get_rank()
        local_bs = predicted.shape[0]
        labels = torch.arange(local_bs, device=logits.device) + rank * local_bs

        loss_p2t = F.cross_entropy(logits, labels)

        logits_t2p = logit_scale * (target @ predicted_all.T)
        loss_t2p = F.cross_entropy(logits_t2p, labels)

        return 0.5 * (loss_p2t + loss_t2p)

    @staticmethod
    def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all ranks with gradient support."""
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return tensor

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)

        # replace current rank's tensor with the original (keeps gradients)
        rank = torch.distributed.get_rank()
        gathered[rank] = tensor

        return torch.cat(gathered, dim=0)
