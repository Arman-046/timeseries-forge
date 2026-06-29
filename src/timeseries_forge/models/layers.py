"""
Core building blocks shared across ForgeNet components.

These implement the gating and variable-selection machinery popularized
by the Temporal Fusion Transformer (Lim et al., 2021), written from
scratch against raw nn.Module primitives so every shape and gradient
path is explicit and inspectable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """GLU(x) = sigmoid(W1 x + b1) * (W2 x + b2)

    A learned gate that lets the network suppress (zero out) a branch
    of computation entirely when it isn't useful, which is what gives
    the Gated Residual Network below its "skip irrelevant inputs"
    behaviour.
    """

    def __init__(self, input_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN).

    GRN(x, c) = LayerNorm(x_proj + GLU(ELU(W2(ELU(W1 x + W3 c) ))))

    The optional context vector `c` (e.g. a static embedding) is
    injected additively before the nonlinearity, letting static
    covariates modulate how every time step is processed without
    forcing a fixed fusion point earlier in the network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)

        self.skip_proj = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip_proj(x)

        h = self.fc1(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)

        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """Learns, per time step, how much weight to give each input variable.

    Each scalar feature is first embedded independently to `hidden_dim`
    via its own GRN ("flatten" GRNs), then a softmax over a GRN applied
    to the concatenation of all features produces per-variable
    attention weights. This gives a built-in, inspectable feature
    importance signal: `weights` returned alongside the output can be
    plotted directly to see which sensors/features the model is
    relying on at each time step, which is invaluable for production
    debugging and stakeholder trust.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # One independent GRN per scalar input variable, each mapping
        # a single scalar (or pre-embedded vector) to hidden_dim.
        self.single_var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(num_vars)
            ]
        )

        self.weight_grn = GatedResidualNetwork(
            num_vars * hidden_dim,
            hidden_dim,
            num_vars,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, num_vars)
            context: optional (batch, time, context_dim) or (batch, context_dim)
        Returns:
            selected: (batch, time, hidden_dim) -- weighted combination
            weights:  (batch, time, num_vars)   -- softmax selection weights
        """
        b, t, n = x.shape
        assert n == self.num_vars, f"expected {self.num_vars} variables, got {n}"

        embedded = torch.stack(
            [self.single_var_grns[i](x[..., i : i + 1]) for i in range(n)], dim=-2
        )  # (batch, time, num_vars, hidden_dim)

        flat = embedded.reshape(b, t, n * self.hidden_dim)
        if context is not None and context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, t, -1)

        weight_logits = self.weight_grn(flat, context)  # (batch, time, num_vars)
        weights = F.softmax(weight_logits, dim=-1)

        selected = (embedded * weights.unsqueeze(-1)).sum(dim=-2)  # (batch, time, hidden_dim)
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head self-attention with shared value projections across heads.

    Standard multi-head attention gives each head its own value
    projection, which makes the per-head attention pattern hard to
    interpret in aggregate. Following the TFT design, this variant
    shares a single value projection across all heads and only varies
    the query/key projections per head, then averages head outputs.
    This keeps multi-head expressiveness for *what to attend to* while
    making *what is being attended to* consistent and interpretable
    across heads -- the averaged attention matrix is a meaningful,
    single interpretability artifact rather than H separate ones.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_head)  # shared value projection
        self.out_proj = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, d_model)
            attn_mask: optional (time, time) boolean mask, True = masked out
        Returns:
            output: (batch, time, d_model)
            attn_weights: (batch, n_heads, time, time)
        """
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x)  # (batch, time, d_head) -- shared across heads

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.d_head**0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # broadcast the shared value across heads, then average head outputs
        out = torch.einsum("bhqk,bkd->bhqd", attn_weights, v)
        out = out.mean(dim=1)  # average over heads -> (batch, time, d_head)

        return self.out_proj(out), attn_weights


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular boolean mask (True = position to be masked out)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
