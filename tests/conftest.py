import sys
from pathlib import Path

import pytest

# allow running tests without installing the package (pip install -e .)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))  # so `scripts.*` is importable in tests

import torch

from timeseries_forge.models.forge_net import ForgeNet, ForgeNetConfig


@pytest.fixture
def small_config() -> ForgeNetConfig:
    return ForgeNetConfig(
        num_features=5,
        num_targets=2,
        seq_len=32,
        horizon=8,
        d_model=16,
        n_heads=2,
        n_layers=2,
        ffn_hidden=32,
        quantiles=(0.1, 0.5, 0.9),
    )


@pytest.fixture
def small_model(small_config) -> ForgeNet:
    torch.manual_seed(0)
    return ForgeNet(small_config)
