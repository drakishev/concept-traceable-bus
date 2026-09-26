"""Learning rate schedulers for VL-JEPA training."""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine decay with linear warmup.

    LR schedule:
        - Steps [0, num_warmup_steps): linear warmup from 0 to base_lr
        - Steps [num_warmup_steps, num_training_steps]: cosine decay to min_lr
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # linear warmup
            return current_step / max(1, num_warmup_steps)
        # cosine decay
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
