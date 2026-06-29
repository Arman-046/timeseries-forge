"""
Reproducibility utilities.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Seeds Python, NumPy, and PyTorch (CPU + all CUDA devices).

    `deterministic_cudnn=True` trades some throughput for exact
    reproducibility of convolution algorithms; left off by default
    since this architecture has no convolutions and the cost isn't
    worth paying, but exposed for anyone adapting the model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
