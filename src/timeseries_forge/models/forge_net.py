"""
ForgeNet: a shared-encoder multi-task model for joint probabilistic
forecasting and anomaly detection on multivariate time series.

Architecture summary
---------------------
    raw features (batch, time, num_features)
        -> VariableSelectionNetwork           (learns per-step feature importance)
        -> LearnedPositionalEncoding
        -> stack of TransformerEncoderBlocks   (InterpretableMultiHeadAttention + GRN feedforward)
        -> shared encoded representation (batch, time, d_model)
            -> QuantileForecastHead   (pooled summary -> future quantiles)
            -> ReconstructionAnomalyHead (per-step -> reconstruction -> anomaly score)

The two task losses are combined using learned homoscedastic
uncertainty weighting (Kendall et al., 2018) rather than a fixed
hand-tuned weight, so the relative scale of the forecasting loss vs.
the reconstruction loss is balanced automatically during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from timeseries_forge.models.heads import (
    LearnedPositionalEncoding,
    QuantileForecastHead,
    ReconstructionAnomalyHead,
)
from timeseries_forge.models.layers import (
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
    causal_mask,
)


@dataclass
class ForgeNetConfig:
    num_features: int                      # total number of input channels
    num_targets: int                       # number of channels to forecast
    seq_len: int = 168                     # input window length (e.g. 168 hourly steps = 1 week)
    horizon: int = 24                      # forecast horizon
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ffn_hidden: int = 256
    dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    causal: bool = True                    # whether self-attention is causally masked
    static_dim: int | None = None          # optional dimensionality of static covariates


class EncoderBlock(nn.Module):
    """Transformer encoder block: interpretable attention + GRN feedforward."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = GatedResidualNetwork(d_model, ffn_hidden, d_model, dropout=dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(x, attn_mask)
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        x = self.ffn(x)
        return x, attn_weights


class ForgeNet(nn.Module):
    """Shared-encoder multi-task forecasting + anomaly detection model."""

    def __init__(self, config: ForgeNetConfig):
        super().__init__()
        self.config = config
        c = config

        self.vsn = VariableSelectionNetwork(
            num_vars=c.num_features,
            hidden_dim=c.d_model,
            context_dim=c.static_dim,
            dropout=c.dropout,
        )
        self.pos_encoding = LearnedPositionalEncoding(c.d_model, max_len=max(c.seq_len, 4096))

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(c.d_model, c.n_heads, c.ffn_hidden, c.dropout)
                for _ in range(c.n_layers)
            ]
        )

        # Attention pooling over time to get a single summary vector for forecasting,
        # rather than naively using only the last time step (which discards information).
        self.pool_query = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            c.d_model, num_heads=c.n_heads, dropout=c.dropout, batch_first=True
        )

        self.forecast_head = QuantileForecastHead(
            c.d_model, c.horizon, c.num_targets, c.quantiles
        )
        self.anomaly_head = ReconstructionAnomalyHead(c.d_model, c.num_features, c.dropout)

        # Learned log-variance per task for uncertainty-weighted multi-task loss.
        # Loss_total = sum_i [ exp(-log_var_i) * Loss_i + log_var_i ]
        # This lets the optimizer discover the right relative weighting between
        # the forecasting loss and the reconstruction loss instead of us guessing
        # a fixed lambda, and it adapts automatically as training progresses.
        self.log_vars = nn.Parameter(torch.zeros(2))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self, x: torch.Tensor, static_context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, num_features)
            static_context: optional (batch, static_dim)
        Returns:
            encoded: (batch, seq_len, d_model)
            var_weights: (batch, seq_len, num_features) variable selection weights
            attn_weights_per_layer: list of (batch, n_heads, seq_len, seq_len)
        """
        selected, var_weights = self.vsn(x, static_context)
        h = self.pos_encoding(selected)

        mask = None
        if self.config.causal:
            mask = causal_mask(x.shape[1], x.device)

        attn_weights_per_layer = []
        for block in self.encoder_blocks:
            h, attn_w = block(h, mask)
            attn_weights_per_layer.append(attn_w)

        return h, var_weights, attn_weights_per_layer

    def forward(
        self, x: torch.Tensor, static_context: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, num_features)
            static_context: optional (batch, static_dim)
        Returns:
            dict with keys:
                forecast: (batch, horizon, num_targets, num_quantiles)
                reconstruction: (batch, seq_len, num_features)
                var_weights: (batch, seq_len, num_features)
                attn_weights: list of per-layer attention tensors
        """
        encoded, var_weights, attn_weights = self.encode(x, static_context)

        b = encoded.shape[0]
        query = self.pool_query.expand(b, -1, -1)
        summary, pool_weights = self.pool_attn(query, encoded, encoded, need_weights=True)
        summary = summary.squeeze(1)  # (batch, d_model)

        forecast = self.forecast_head(summary)
        reconstruction = self.anomaly_head(encoded)

        return {
            "forecast": forecast,
            "reconstruction": reconstruction,
            "var_weights": var_weights,
            "attn_weights": attn_weights,
            "pool_weights": pool_weights,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        y_forecast: torch.Tensor,
        y_reconstruction_target: torch.Tensor,
        quantiles: tuple[float, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: dict returned by forward()
            y_forecast: (batch, horizon, num_targets) ground-truth future values
            y_reconstruction_target: (batch, seq_len, num_features) clean input
                (i.e. what the model should reconstruct; can be the input itself
                when training without injected noise/anomalies)
        Returns:
            dict with 'total', 'forecast_loss', 'reconstruction_loss'
        """
        quantiles = quantiles or self.config.quantiles
        pred = outputs["forecast"]  # (batch, horizon, num_targets, num_quantiles)

        # Pinball / quantile loss, averaged over quantiles, targets, horizon, batch.
        y_expanded = y_forecast.unsqueeze(-1)  # (batch, horizon, num_targets, 1)
        errors = y_expanded - pred
        q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
        pinball = torch.maximum(q * errors, (q - 1) * errors)
        forecast_loss = pinball.mean()

        recon_loss = torch.nn.functional.mse_loss(
            outputs["reconstruction"], y_reconstruction_target
        )

        precision = torch.exp(-self.log_vars)
        total = (
            precision[0] * forecast_loss
            + self.log_vars[0]
            + precision[1] * recon_loss
            + self.log_vars[1]
        )

        return {
            "total": total,
            "forecast_loss": forecast_loss.detach(),
            "reconstruction_loss": recon_loss.detach(),
            "forecast_weight": precision[0].detach(),
            "reconstruction_weight": precision[1].detach(),
        }

    @torch.no_grad()
    def anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep anomaly score = mean squared reconstruction error.

        Returns:
            (batch, seq_len) anomaly scores, higher = more anomalous
        """
        self.eval()
        outputs = self.forward(x)
        error = (outputs["reconstruction"] - x).pow(2).mean(dim=-1)
        return error

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
