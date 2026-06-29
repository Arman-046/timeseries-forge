"""
Learning rate scheduling utilities.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Linear warmup followed by cosine decay to `min_lr_ratio` * base_lr.

    Built from scratch with LambdaLR rather than pulling in a
    third-party scheduler library, so the exact schedule is visible
    and easy to plot/debug. This is the same shape of schedule used to
    train most modern transformer models and tends to be far more
    stable early in training than a constant or step-decay LR,
    especially for attention-heavy architectures like ForgeNet where
    early large updates can destabilize the softmax attention logits.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
