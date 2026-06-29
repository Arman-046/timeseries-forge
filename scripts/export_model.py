#!/usr/bin/env python3
"""
Export a trained checkpoint to TorchScript and ONNX for deployment.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from timeseries_forge.deployment.export import export_onnx, export_torchscript
from timeseries_forge.models.forge_net import ForgeNet, ForgeNetConfig
from timeseries_forge.utils.seed import configure_logging

logger = logging.getLogger("export")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ForgeNet checkpoint for deployment")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--output-dir", type=str, default="artifacts")
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--skip-onnx", action="store_true", help="skip ONNX export (requires `onnx` package)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not ckpt.get("config"):
        raise ValueError(
            "checkpoint has no stored config; re-save with Trainer (which stores "
            "model.config automatically) or pass --num-features/etc manually"
        )
    config = ForgeNetConfig(**ckpt["config"])
    model = ForgeNet(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    example_input = torch.randn(args.batch_size, args.seq_len, config.num_features)

    output_dir = Path(args.output_dir)
    ts_path = export_torchscript(model, example_input, output_dir / "model.pt")
    logger.info("TorchScript artifact written to %s", ts_path)

    if not args.skip_onnx:
        try:
            onnx_path = export_onnx(model, example_input, output_dir / "model.onnx")
            logger.info("ONNX artifact written to %s", onnx_path)
        except ImportError:
            logger.warning("onnx package not installed; skipping ONNX export. "
                            "Install with `pip install onnx` to enable it.")


if __name__ == "__main__":
    main()
