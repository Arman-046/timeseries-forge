"""
Sliding-window Dataset for multivariate time series and a simple
per-channel standard scaler with proper train/test fit discipline
(fit on train only, applied to val/test -- a common leakage bug in
naive time-series pipelines).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ChannelScaler:
    """Per-channel (feature-wise) standardization, fit on training data only."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-8

    def fit(self, x: np.ndarray) -> "ChannelScaler":
        """x: (time, channels)"""
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "call fit() before transform()"
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "call fit() before transform()"
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ChannelScaler":
        return cls(mean=np.array(state["mean"]), std=np.array(state["std"]))


class SlidingWindowDataset(Dataset):
    """Produces (input_window, forecast_target, reconstruction_target) triples.

    Given a (time, num_features) array, generates all valid windows of
    length `seq_len` with a forecast target of the next `horizon` steps
    on the configured target channel indices. The reconstruction
    target is the (optionally noised) input window itself, used for
    the anomaly-detection auxiliary task.

    Windows are precomputed as index offsets (not materialized copies)
    to keep memory usage proportional to the raw series length rather
    than O(num_windows * seq_len), which matters once series get into
    the millions of rows (e.g. high-frequency sensor data).
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        horizon: int,
        target_indices: list[int],
        stride: int = 1,
        noise_std: float = 0.0,
    ):
        """
        Args:
            data: (time, num_features) array, already scaled
            seq_len: length of input window
            horizon: number of future steps to forecast
            target_indices: which feature columns to forecast
            stride: step between consecutive windows (1 = maximum overlap)
            noise_std: if > 0, Gaussian noise added to reconstruction *input*
                (not target) to make the reconstruction task a true denoising
                objective, improving robustness of the anomaly score.
        """
        assert data.ndim == 2, "data must be (time, num_features)"
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_indices = target_indices
        self.noise_std = noise_std

        max_start = len(data) - seq_len - horizon
        if max_start < 0:
            raise ValueError(
                f"Series length {len(data)} too short for seq_len={seq_len} "
                f"+ horizon={horizon}"
            )
        self.starts = list(range(0, max_start + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        window = self.data[start : start + self.seq_len]
        future = self.data[
            start + self.seq_len : start + self.seq_len + self.horizon
        ]
        future_targets = future[:, self.target_indices]

        recon_input = window
        if self.noise_std > 0:
            recon_input = window + torch.randn_like(window) * self.noise_std

        return {
            "input": recon_input,
            "forecast_target": future_targets,
            "reconstruction_target": window,
        }


def train_val_test_split_indices(
    n: int, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[slice, slice, slice]:
    """Chronological (non-shuffled) split -- the only valid split for time series.

    Returns contiguous slices in time order: train comes first, then
    val, then test, with no shuffling, which would otherwise leak
    future information into training (a frequent and serious bug in
    naive time-series pipelines that use sklearn's default random
    train_test_split).
    """
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)
