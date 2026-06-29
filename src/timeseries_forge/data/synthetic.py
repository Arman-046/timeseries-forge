"""
Synthetic multivariate time series generator.

Produces realistic-looking sensor/metric-style data: multiple
correlated channels with trend, multi-period seasonality, noise, and
optionally injected point/contextual anomalies with ground-truth
labels -- so the anomaly detection head can be evaluated against a
known answer key (impossible with most real-world datasets, which
rarely have labeled anomalies).
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_series(
    n_steps: int = 8760,  # default: one year of hourly data
    n_channels: int = 6,
    seed: int = 42,
    anomaly_rate: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        data: (n_steps, n_channels) float array
        anomaly_labels: (n_steps,) int array, 1 where any channel was
            perturbed by an injected anomaly at that timestep, else 0
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    data = np.zeros((n_steps, n_channels))
    # shared latent factor so channels are correlated, like sensors on
    # the same physical system (e.g. correlated server CPU/memory/IO)
    latent = (
        0.5 * np.sin(2 * np.pi * t / 24)            # daily cycle
        + 0.3 * np.sin(2 * np.pi * t / (24 * 7))    # weekly cycle
        + 0.0008 * t                                 # slow upward trend
    )

    for c in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        freq_jitter = rng.uniform(0.9, 1.1)
        channel_signal = (
            (rng.uniform(0.5, 1.5)) * latent * freq_jitter
            + rng.uniform(0.3, 0.8) * np.sin(2 * np.pi * t / 24 * freq_jitter + phase)
            + rng.normal(0, 0.15, size=n_steps)
            + rng.uniform(-1, 1)  # per-channel offset
        )
        data[:, c] = channel_signal

    anomaly_labels = np.zeros(n_steps, dtype=int)
    n_anomalies = int(n_steps * anomaly_rate)
    margin = 5
    anomaly_starts = rng.choice(
        np.arange(margin, n_steps - margin), size=n_anomalies, replace=False
    )

    for start in anomaly_starts:
        kind = rng.choice(["spike", "level_shift", "dropout"])
        duration = rng.integers(1, 6)
        end = min(start + duration, n_steps)
        affected_channels = rng.choice(
            n_channels, size=rng.integers(1, n_channels + 1), replace=False
        )

        if kind == "spike":
            magnitude = rng.uniform(3, 6) * rng.choice([-1, 1])
            data[start:end, affected_channels] += magnitude
        elif kind == "level_shift":
            magnitude = rng.uniform(2, 4) * rng.choice([-1, 1])
            data[start:end, affected_channels] += magnitude
        elif kind == "dropout":
            data[start:end, affected_channels] = 0.0

        anomaly_labels[start:end] = 1

    return data, anomaly_labels


def channel_names(n_channels: int, prefix: str = "sensor") -> list[str]:
    return [f"{prefix}_{i}" for i in range(n_channels)]
