#!/usr/bin/env python3
"""
Evaluation entrypoint.

Loads a trained checkpoint, runs it over a held-out test set, and
reports both forecast metrics (point + calibration) and anomaly
detection metrics (against synthetic ground-truth labels, if using
synthetic data).

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from timeseries_forge.data.datasets import ChannelScaler, SlidingWindowDataset, train_val_test_split_indices
from timeseries_forge.data.synthetic import generate_synthetic_series
from timeseries_forge.evaluation.anomaly_metrics import evaluate_anomaly_detection
from timeseries_forge.evaluation.forecast_metrics import evaluate_forecast
from timeseries_forge.models.forge_net import ForgeNet, ForgeNetConfig
from timeseries_forge.utils.seed import configure_logging, set_seed

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained ForgeNet checkpoint")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--scaler-path", type=str, default="artifacts/scaler.json")
    p.add_argument("--n-channels", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=8760)
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--target-indices", type=int, nargs="+", default=[0, 1])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-path", type=str, default="artifacts/eval_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, anomaly_labels = generate_synthetic_series(n_steps=args.n_steps, n_channels=args.n_channels)
    _, _, test_idx = train_val_test_split_indices(len(data))
    test_raw = data[test_idx]
    test_labels = anomaly_labels[test_idx]

    with open(args.scaler_path) as f:
        scaler = ChannelScaler.from_state_dict(json.load(f))
    test_scaled = scaler.transform(test_raw)

    test_ds = SlidingWindowDataset(test_scaled, args.seq_len, args.horizon, args.target_indices)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ForgeNetConfig(**ckpt["config"]) if ckpt.get("config") else ForgeNetConfig(
        num_features=args.n_channels, num_targets=len(args.target_indices),
        seq_len=args.seq_len, horizon=args.horizon,
    )
    model = ForgeNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_forecasts, all_targets, all_anomaly_scores = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device)
            outputs = model(x)
            all_forecasts.append(outputs["forecast"].cpu())
            all_targets.append(batch["forecast_target"])
            err = (outputs["reconstruction"] - x).pow(2).mean(dim=-1)
            all_anomaly_scores.append(err.cpu())

    forecasts = torch.cat(all_forecasts, dim=0)
    targets = torch.cat(all_targets, dim=0)
    forecast_report = evaluate_forecast(targets, forecasts, config.quantiles)

    anomaly_scores = torch.cat(all_anomaly_scores, dim=0).numpy()
    # align per-window anomaly labels: use the label at the last timestep of each window
    window_labels = np.array(
        [test_labels[s + args.seq_len - 1] for s in test_ds.starts]
    )
    # anomaly_scores is (n_windows, seq_len); take last-timestep score per window
    last_step_scores = anomaly_scores[:, -1]
    anomaly_report = evaluate_anomaly_detection(window_labels, last_step_scores)

    full_report = {"forecast": forecast_report, "anomaly_detection": anomaly_report}
    logger.info("Forecast metrics:\n%s", json.dumps(forecast_report, indent=2))
    logger.info("Anomaly detection metrics:\n%s", json.dumps(anomaly_report, indent=2))

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info("full report written to %s", args.report_path)


if __name__ == "__main__":
    main()
