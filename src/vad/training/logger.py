from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.vad.training.metrics import BinaryClassificationMetrics


class TensorBoardLogger:
    """
    Thin wrapper around PyTorch's SummaryWriter for structured training logs.
    """

    def __init__(
        self,
        log_dir: str | Path = "runs",
    ) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_epoch(
        self,
        epoch: int,
        train: BinaryClassificationMetrics,
        val: BinaryClassificationMetrics,
    ) -> None:
        """
        Log aggregated train/validation metrics for one epoch.

        Args:
            epoch: Epoch number.
            train: Training metrics.
            val: Validation metrics.
        """
        # --- Core metrics ---
        self._log_pair("loss", train.loss, val.loss, epoch)
        self._log_pair("f1", train.f1, val.f1, epoch)
        self._log_pair("precision", train.precision, val.precision, epoch)
        self._log_pair("recall", train.recall, val.recall, epoch)
        self._log_pair("accuracy", train.accuracy, val.accuracy, epoch)

        # --- VAD-specific metrics ---
        self._log_pair(
            "false_positive_rate", train.false_positive_rate, val.false_positive_rate, epoch
        )
        self._log_pair(
            "false_negative_rate", train.false_negative_rate, val.false_negative_rate, epoch
        )

        # --- Confusion matrix counts (validation only, usually more relevant) ---
        self.writer.add_scalar("confusion/val/tp", val.tp, epoch)
        self.writer.add_scalar("confusion/val/fp", val.fp, epoch)
        self.writer.add_scalar("confusion/val/tn", val.tn, epoch)
        self.writer.add_scalar("confusion/val/fn", val.fn, epoch)

        self.writer.flush()

    def _log_pair(self, name: str, train_value: float, val_value: float, step: int) -> None:
        """
        Log a train/val scalar pair under a shared namespace.

        Example:
            name="loss" → loss/train, loss/val
        """
        self.writer.add_scalar(f"{name}/train", train_value, step)
        self.writer.add_scalar(f"{name}/val", val_value, step)

    def log_hparams(
        self,
        hparams: dict[str, float | int | str],
        final_metrics: dict[str, float],
    ) -> None:
        """
        Log hyperparameters and final summary metrics.

        Args:
            hparams: Hyperparameter dictionary.
            final_metrics: Final metric dictionary.
        """
        self.writer.add_hparams(hparams, final_metrics)
        self.writer.flush()

    def close(self) -> None:
        """Flush and close the writer."""
        self.writer.flush()
        self.writer.close()
