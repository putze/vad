from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.vad.training.metrics import BinaryClassificationMetrics


class TensorBoardLogger:
    """
    Thin wrapper around PyTorch's SummaryWriter for training logs.
    """

    def __init__(self, log_dir: str | Path = "runs") -> None:
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
        self.writer.add_scalar("loss/train", train.loss, epoch)
        self.writer.add_scalar("loss/val", val.loss, epoch)

        self.writer.add_scalar("accuracy/train", train.accuracy, epoch)
        self.writer.add_scalar("accuracy/val", val.accuracy, epoch)

        self.writer.add_scalar("precision/train", train.precision, epoch)
        self.writer.add_scalar("precision/val", val.precision, epoch)

        self.writer.add_scalar("recall/train", train.recall, epoch)
        self.writer.add_scalar("recall/val", val.recall, epoch)

        self.writer.add_scalar("f1/train", train.f1, epoch)
        self.writer.add_scalar("f1/val", val.f1, epoch)

        self.writer.flush()

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
