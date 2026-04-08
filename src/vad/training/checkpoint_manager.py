from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from vad.training.callbacks import MetricTracker


class CheckpointManager:
    """
    Save best and/or latest training checkpoints for one experiment run.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_f1",
        mode: str = "max",
        min_delta: float = 0.0,
        save_last: bool = True,
        best_filename: str = "best.pt",
        last_filename: str = "last.pt",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.save_last = save_last
        self.best_path = self.checkpoint_dir / best_filename
        self.last_path = self.checkpoint_dir / last_filename
        self.metric_tracker = MetricTracker(mode=mode, min_delta=min_delta)

    @property
    def best_value(self) -> float | None:
        return self.metric_tracker.best_value

    def _build_state(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        metrics: dict[str, float],
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "epoch": epoch,
            "monitor": self.monitor,
            "best_value": self.best_value,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if extra_state is not None:
            state["extra_state"] = extra_state
        return state

    def step(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        metrics: dict[str, float],
        extra_state: dict[str, Any] | None = None,
    ) -> bool:
        """
        Save checkpoints for the current epoch.

        Returns:
            bool: True if a new best checkpoint was saved.
        """
        if self.monitor not in metrics:
            raise KeyError(
                f"Monitored metric '{self.monitor}' not found in metrics: {sorted(metrics)}"
            )

        current_value = metrics[self.monitor]
        improved = self.metric_tracker.update(current_value)

        if improved:
            self.best_epoch = epoch

        state = self._build_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            extra_state=extra_state,
        )

        if self.save_last:
            torch.save(state, self.last_path)

        if improved:
            torch.save(state, self.best_path)

        return improved
