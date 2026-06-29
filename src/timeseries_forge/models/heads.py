"""
Positional encoding and output heads for ForgeNet.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned (not sinusoidal) positional embeddings.

    For fixed-length input windows, a learned embedding table
    typically outperforms sinusoidal encodings slightly and is simpler
    to reason about. Falls back gracefully if a longer sequence than
    `max_len` is ever passed, by re-interpolating the embedding table.
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if t <= self.max_len:
            pos = self.embedding[:, :t, :]
        else:
            # interpolate to support sequences longer than max_len at inference
            pos = nn.functional.interpolate(
                self.embedding.transpose(1, 2), size=t, mode="linear", align_corners=False
            ).transpose(1, 2)
        return x + pos


class QuantileForecastHead(nn.Module):
    """Probabilistic forecasting head producing multiple quantiles.

    Rather than a single point estimate, this head predicts a fixed
    set of quantiles (e.g. 0.1, 0.5, 0.9) for each future time step and
    each target variable, trained with the pinball (quantile) loss.
    This gives calibrated uncertainty bands "for free" -- crucial for
    any real decision-making system (e.g. capacity planning, alerting
    thresholds) where a single point forecast hides risk.
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        num_targets: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Project the final encoder representation to horizon * targets * quantiles.
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon * num_targets * self.num_quantiles),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            summary: (batch, d_model) pooled encoder representation
        Returns:
            (batch, horizon, num_targets, num_quantiles)
        """
        b = summary.shape[0]
        out = self.proj(summary)
        out = out.view(b, self.horizon, self.num_targets, self.num_quantiles)
        # enforce monotonicity across quantiles (q10 <= q50 <= q90) via cumulative sum
        # of non-negative increments -- prevents quantile crossing, a common failure
        # mode of naively-trained multi-quantile heads.
        base = out[..., :1]
        increments = torch.nn.functional.softplus(out[..., 1:])
        sorted_out = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        return sorted_out


class ReconstructionAnomalyHead(nn.Module):
    """Anomaly detection head based on masked sequence reconstruction.

    Reconstructs the (denoised) input window from the encoder's
    per-timestep representations. Anomaly scores are derived at
    inference time from reconstruction error -- points the model
    cannot reconstruct well are flagged as anomalous. This is trained
    jointly with forecasting via a shared encoder (see ForgeNet),
    which acts as a form of multi-task regularization: a
    representation that is good for forecasting the near future is
    also encouraged to be good for explaining the recent past.
    """

    def __init__(self, d_model: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_features),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: (batch, time, d_model) per-timestep encoder output
        Returns:
            (batch, time, num_features) reconstruction
        """
        return self.decoder(encoded)
