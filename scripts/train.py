#!/usr/bin/env python3
"""
End-to-end training entrypoint.

Usage:
    python scripts/train.py --epochs 30 --d-model 128 --n-layers 3

Generates synthetic multivariate sensor data (swap in your own CSV via
--data-path), builds chronological train/val/test splits, fits a
scaler on train only, constructs ForgeNet, and runs the full Trainer
loop with checkpointing + TensorBoard logging.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from timeseries_forge.data.datasets import (
    ChannelScaler,
    SlidingWindowDataset,
    train_val_test_split_indices,
)
from timeseries_forge.data.synthetic import generate_synthetic_series
from timeseries_forge.models.forge_net import ForgeNet, ForgeNetConfig
from timeseries_forge.training.trainer import Trainer, TrainerConfig
from timeseries_forge.utils.seed import configure_logging, set_seed

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ForgeNet on multivariate time series")
    p.add_argument("--data-path", type=str, default=None, help="optional .npy/.csv file (time, channels)")
    p.add_argument("--n-channels", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=8760, help="synthetic data length if no --data-path")
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--target-indices", type=int, nargs="+", default=[0, 1])
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--output-dir", type=str, default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(args: argparse.Namespace) -> np.ndarray:
    if args.data_path is None:
        logger.info("no --data-path given; generating synthetic data (%d steps, %d channels)",
                    args.n_steps, args.n_channels)
        data, _ = generate_synthetic_series(n_steps=args.n_steps, n_channels=args.n_channels)
        return data

    path = Path(args.data_path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        return numeric.values
    else:
        raise ValueError(f"unsupported data file type: {path.suffix}")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    data = load_data(args)
    n_channels = data.shape[1]
    logger.info("data shape: %s", data.shape)

    train_idx, val_idx, test_idx = train_val_test_split_indices(len(data))
    train_raw, val_raw, test_raw = data[train_idx], data[val_idx], data[test_idx]

    scaler = ChannelScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "scaler.json", "w") as f:
        json.dump(scaler.state_dict(), f, indent=2)

    train_ds = SlidingWindowDataset(
        train_scaled, args.seq_len, args.horizon, args.target_indices, noise_std=0.05
    )
    val_ds = SlidingWindowDataset(val_scaled, args.seq_len, args.horizon, args.target_indices)
    test_ds = SlidingWindowDataset(test_scaled, args.seq_len, args.horizon, args.target_indices)

    logger.info("train/val/test windows: %d / %d / %d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = ForgeNetConfig(
        num_features=n_channels,
        num_targets=len(args.target_indices),
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = ForgeNet(config)
    logger.info("model parameters: %s", f"{model.num_parameters():,}")

    trainer_config = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    trainer = Trainer(model, trainer_config)
    history = trainer.fit(train_loader, val_loader)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("training complete. best checkpoint at %s/best.pt", args.checkpoint_dir)


if __name__ == "__main__":
    main()
