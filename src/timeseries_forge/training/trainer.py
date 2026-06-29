"""
Trainer: orchestrates the full training loop for ForgeNet.

Demonstrates production training engineering:
  - Automatic mixed precision (AMP) via torch.amp.autocast + GradScaler
  - Gradient accumulation (effective batch size > what fits in memory)
  - Gradient clipping (stabilizes attention-heavy architectures)
  - Cosine LR schedule with warmup
  - Early stopping on validation loss
  - Best + last checkpointing
  - TensorBoard scalar logging
  - Device-agnostic (CPU / single-GPU here; see README for DDP notes)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from timeseries_forge.training.checkpoint import CheckpointManager, EarlyStopping
from timeseries_forge.training.scheduler import cosine_warmup_scheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = True
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_n_steps: int = 50


class Trainer:
    def __init__(self, model: torch.nn.Module, config: TrainerConfig):
        self.model = model.to(config.device)
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # GradScaler becomes a no-op when enabled=False, so it's safe to
        # construct unconditionally; we just make sure its device string
        # matches the actual training device rather than hardcoding "cuda",
        # since GradScaler is also valid (and a no-op without fp16 gradient
        # scaling benefit) on CPU.
        self.scaler = torch.amp.GradScaler(
            device=config.device, enabled=(config.use_amp and config.device == "cuda")
        )

        self.checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min")

        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=config.log_dir)
        except ImportError:
            logger.warning("tensorboard not installed; scalar logging to TensorBoard disabled")

        self.global_step = 0
        self.scheduler = None  # built lazily once we know len(train_loader)

    def _autocast_ctx(self):
        return torch.amp.autocast(
            device_type="cuda" if self.config.device == "cuda" else "cpu",
            enabled=self.config.use_amp,
        )

    def _move_batch(self, batch: dict) -> dict:
        return {k: v.to(self.config.device, non_blocking=True) for k, v in batch.items()}

    def _run_batch(self, batch: dict, training: bool) -> dict:
        batch = self._move_batch(batch)
        with self._autocast_ctx():
            outputs = self.model(batch["input"])
            losses = self.model.compute_loss(
                outputs,
                y_forecast=batch["forecast_target"],
                y_reconstruction_target=batch["reconstruction_target"],
            )
        return losses

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        cfg = self.config
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * cfg.epochs // max(1, cfg.grad_accum_steps)
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
        self.scheduler = cosine_warmup_scheduler(self.optimizer, warmup_steps, total_steps)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._train_one_epoch(train_loader, epoch)
            val_metrics = self._validate(val_loader)

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])

            is_best = self.early_stopping.step(val_metrics["total"])
            self.checkpoint_mgr.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                {"train": train_metrics, "val": val_metrics},
                is_best=is_best,
                config=vars(self.model.config) if hasattr(self.model, "config") else None,
            )

            logger.info(
                "epoch %d | train_loss=%.4f val_loss=%.4f forecast=%.4f recon=%.4f %s",
                epoch,
                train_metrics["total"],
                val_metrics["total"],
                val_metrics["forecast_loss"],
                val_metrics["reconstruction_loss"],
                "(best)" if is_best else "",
            )

            if self._writer:
                for k, v in train_metrics.items():
                    self._writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    self._writer.add_scalar(f"val/{k}", v, epoch)

            if self.early_stopping.should_stop:
                logger.info("early stopping triggered at epoch %d", epoch)
                break

        if self._writer:
            self._writer.close()
        return history

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        cfg = self.config
        agg = {"total": 0.0, "forecast_loss": 0.0, "reconstruction_loss": 0.0}
        n_batches = 0
        start = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            losses = self._run_batch(batch, training=True)
            loss = losses["total"] / cfg.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % cfg.grad_accum_steps == 0 or (i + 1) == len(loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()
                self.global_step += 1

            for k in agg:
                value = losses[k]
                value = float(value.detach()) if torch.is_tensor(value) else float(value)
            n_batches += 1

            if (i + 1) % cfg.log_every_n_steps == 0:
                elapsed = time.time() - start
                logger.debug(
                    "epoch %d step %d/%d | loss=%.4f (%.1fs elapsed)",
                    epoch,
                    i + 1,
                    len(loader),
                    float(losses["total"].detach()),
                    elapsed,
                )

        return {k: v / n_batches for k, v in agg.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        agg = {"total": 0.0, "forecast_loss": 0.0, "reconstruction_loss": 0.0}
        n_batches = 0
        for batch in loader:
            losses = self._run_batch(batch, training=False)
            for k in agg:
                value = losses[k]
                agg[k] += float(value.detach()) if torch.is_tensor(value) else float(value)
            n_batches += 1
        return {k: v / max(1, n_batches) for k, v in agg.items()}
