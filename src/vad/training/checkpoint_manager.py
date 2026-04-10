from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from vad.training.callbacks import MetricTracker


def _serialize_extra_state(extra_state: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Convert dataclass values in extra_state to plain dictionaries so they
    can be safely saved and reloaded.
    """
    if extra_state is None:
        return None

    serialized: dict[str, Any] = {}
    for key, value in extra_state.items():
        if dataclasses.is_dataclass(value):
            serialized[key] = dataclasses.asdict(value)
        else:
            serialized[key] = value
    return serialized


class CheckpointManager:
    """
    Manage checkpoint saving during training.

    This class can save:
    - the most recent checkpoint (``last.pt`` by default)
    - the best checkpoint according to a monitored validation metric (``best.pt`` by default)

    The saved checkpoint includes model state, optimizer state, metrics,
    and optional extra training state.
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
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory where checkpoint files will be saved.
            monitor: Name of the metric used to determine whether a checkpoint is the new best one.
            mode: Optimization direction for the monitored metric:
                - ``"min"`` means lower is better
                - ``"max"`` means higher is better
            min_delta: Minimum improvement required to replace the current best checkpoint.
            save_last: Whether to always save the latest checkpoint.
            best_filename: Filename for the best checkpoint.
            last_filename: Filename for the latest checkpoint.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.save_last = save_last
        self.best_path = self.checkpoint_dir / best_filename
        self.last_path = self.checkpoint_dir / last_filename
        self.metric_tracker = MetricTracker(mode=mode, min_delta=min_delta)
        self.best_epoch: int | None = None

    @property
    def best_value(self) -> float | None:
        """
        Return the best monitored metric value seen so far.

        Returns:
            The best metric value, or None if no checkpoint has been evaluated yet.
        """
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
        """
        Build the checkpoint payload for the current training state.

        Args:
            epoch: Current epoch number.
            model: Model whose parameters should be saved.
            optimizer: Optimizer whose internal state should be saved.
            metrics: Dictionary of metrics computed for the current epoch.
            extra_state: Optional additional state to include, such as
                scheduler state, scaler state, random seeds, or configuration.

        Returns:
            A dictionary ready to be serialized with ``torch.save()``.
        """
        state: dict[str, Any] = {
            "epoch": epoch,
            "monitor": self.monitor,
            "best_value": self.best_value,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        serialized_extra_state = _serialize_extra_state(extra_state)
        if serialized_extra_state is not None:
            state["extra_state"] = serialized_extra_state

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

        The method checks whether the monitored metric improved. If so,
        it updates the best checkpoint. Independently, it can also save
        the latest checkpoint if ``save_last=True``.

        Args:
            epoch: Current epoch number.
            model: Model whose parameters should be saved.
            optimizer: Optimizer whose internal state should be saved.
            metrics: Dictionary of current epoch metrics. It must contain
                the monitored metric specified by ``self.monitor``.
            extra_state: Optional additional state to include in the checkpoint.

        Returns:
            True if a new best checkpoint was saved, otherwise False.

        Raises:
            KeyError: If the monitored metric is missing from ``metrics``.
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
