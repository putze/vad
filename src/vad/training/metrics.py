from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """
    Container for aggregated binary classification metrics.

    Attributes:
        loss: Mean loss over all valid frames.
        accuracy: Fraction of correct predictions.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
        num_frames: Number of valid evaluated frames.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_frames: int


class VADMetricsTracker:
    """
    Hybrid metrics tracker for frame-level binary VAD.

    TorchMetrics is used for the actual metric implementations, while this
    wrapper handles:
        - shape normalization
        - masking padded frames
        - epoch-level loss accumulation

    Expected inputs:
        logits:  [B, T] or [B, 1, T]
        targets: [B, T]
        mask:    [B, T] with True for valid frames
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Args:
            threshold: Probability threshold used to convert sigmoid outputs
                into binary predictions.
        """
        self.threshold = threshold

        self.accuracy = BinaryAccuracy(threshold=threshold)
        self.precision = BinaryPrecision(threshold=threshold)
        self.recall = BinaryRecall(threshold=threshold)
        self.f1 = BinaryF1Score(threshold=threshold)

        self.reset()

    def reset(self) -> None:
        """Reset all metric states and accumulated loss."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

        self.total_loss = 0.0
        self.total_frames = 0

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
        loss: Tensor,
        mask: Tensor | None = None,
    ) -> None:
        """
        Update tracker state from one batch.

        Args:
            logits: Model outputs before sigmoid. Shape [B, T] or [B, 1, T].
            targets: Binary frame labels. Shape [B, T].
            loss: Scalar batch loss.
            mask: Optional boolean mask with shape [B, T]. True means valid.

        Raises:
            ValueError: If tensor shapes are not compatible.
        """
        logits_bt = self._flatten_logits(logits)
        targets_bt = self._flatten_targets(targets)

        if logits_bt.shape != targets_bt.shape:
            raise ValueError(
                f"logits and targets must have the same shape, got "
                f"{tuple(logits_bt.shape)} and {tuple(targets_bt.shape)}"
            )

        if mask is None:
            valid_mask = torch.ones_like(targets_bt, dtype=torch.bool)
        else:
            if mask.shape != targets_bt.shape:
                raise ValueError(
                    f"mask and targets must have the same shape, got "
                    f"{tuple(mask.shape)} and {tuple(targets_bt.shape)}"
                )
            valid_mask = mask.to(dtype=torch.bool)

        logits_valid = logits_bt[valid_mask]
        targets_valid = targets_bt[valid_mask]

        if logits_valid.numel() == 0:
            return

        probs_valid = torch.sigmoid(logits_valid)
        targets_valid = targets_valid.to(dtype=torch.long)

        self.accuracy.update(probs_valid, targets_valid)
        self.precision.update(probs_valid, targets_valid)
        self.recall.update(probs_valid, targets_valid)
        self.f1.update(probs_valid, targets_valid)

        num_valid = int(targets_valid.numel())
        self.total_loss += float(loss.detach().item()) * num_valid
        self.total_frames += num_valid

    def compute(self) -> BinaryClassificationMetrics:
        """
        Compute final epoch metrics.

        Returns:
            BinaryClassificationMetrics: Aggregated metrics.
        """
        if self.total_frames == 0:
            return BinaryClassificationMetrics(
                loss=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                num_frames=0,
            )

        return BinaryClassificationMetrics(
            loss=self.total_loss / self.total_frames,
            accuracy=float(self.accuracy.compute().item()),
            precision=float(self.precision.compute().item()),
            recall=float(self.recall.compute().item()),
            f1=float(self.f1.compute().item()),
            num_frames=self.total_frames,
        )

    @staticmethod
    def _flatten_logits(logits: Tensor) -> Tensor:
        """
        Normalize logits to shape [B, T].

        Args:
            logits: Tensor of shape [B, T] or [B, 1, T].

        Returns:
            Tensor: Logits with shape [B, T].

        Raises:
            ValueError: If logits have an unsupported shape.
        """
        if logits.ndim == 2:
            return logits

        if logits.ndim == 3 and logits.shape[1] == 1:
            return logits[:, 0, :]

        raise ValueError(
            f"Expected logits with shape [B, T] or [B, 1, T], got {tuple(logits.shape)}"
        )

    @staticmethod
    def _flatten_targets(targets: Tensor) -> Tensor:
        """
        Validate targets shape.

        Args:
            targets: Tensor expected to have shape [B, T].

        Returns:
            Tensor: Unchanged targets.

        Raises:
            ValueError: If targets do not have shape [B, T].
        """
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape [B, T], got {tuple(targets.shape)}")
        return targets
