"""
Early stopping and checkpoint management utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Tracks the best value seen and a patience counter; `should_stop`
    becomes True once `patience` consecutive epochs have passed
    without an improvement greater than `min_delta`.
    """

    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "min"  # 'min' for loss, 'max' for accuracy-like metrics

    def __post_init__(self):
        self.best: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Returns True if `value` is a new best."""
        if self.best is None:
            self.best = value
            return True

        improved = (
            (value < self.best - self.min_delta)
            if self.mode == "min"
            else (value > self.best + self.min_delta)
        )

        if improved:
            self.best = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class CheckpointManager:
    """Saves/loads model + optimizer + scheduler + metadata to disk.

    Always keeps the single best checkpoint (by monitored metric) plus
    the most recent checkpoint, so training can be resumed after a
    crash without re-running from scratch, and the best model is never
    accidentally overwritten by a later, worse epoch.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_path = self.directory / "best.pt"
        self.last_path = self.directory / "last.pt"
        self.meta_path = self.directory / "metadata.json"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: dict,
        is_best: bool,
        config: dict | None = None,
    ) -> None:
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        }
        torch.save(state, self.last_path)
        if is_best:
            torch.save(state, self.best_path)

        meta = {"epoch": epoch, "metrics": metrics, "is_best_so_far": is_best}
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str | Path | None = None, map_location: str = "cpu") -> dict:
        path = Path(path) if path else self.best_path
        return torch.load(path, map_location=map_location, weights_only=False)
