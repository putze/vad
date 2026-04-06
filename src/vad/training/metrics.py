from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    """
    Aggregated frame-level binary classification metrics.

    Attributes:
        loss: Mean loss over all valid frames.
        accuracy: Fraction of correct predictions.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
        false_positive_rate: FP / (FP + TN).
        false_negative_rate: FN / (FN + TP).
        miss_rate: Alias for false_negative_rate.
        tp: Number of true positives.
        fp: Number of false positives.
        tn: Number of true negatives.
        fn: Number of false negatives.
        num_frames: Number of valid evaluated frames.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    miss_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    num_frames: int

    @classmethod
    def empty(cls) -> BinaryClassificationMetrics:
        """Return a zero-initialized metrics object."""
        return cls(
            loss=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            miss_rate=0.0,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            num_frames=0,
        )


class VADMetricsTracker:
    """
    Hybrid metrics tracker for frame-level binary VAD.

    TorchMetrics provides the metric implementations, while this wrapper handles:
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
        Initialize the tracker.

        Args:
            threshold: Probability threshold used to convert sigmoid outputs
                into binary predictions.
        """
        self.threshold = threshold

        self.accuracy = BinaryAccuracy(threshold=threshold)
        self.precision = BinaryPrecision(threshold=threshold)
        self.recall = BinaryRecall(threshold=threshold)
        self.f1 = BinaryF1Score(threshold=threshold)
        self.confusion = BinaryConfusionMatrix(threshold=threshold)

        self.reset()

    def reset(self) -> None:
        """Reset all metric states and accumulated loss."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confusion.reset()

        self.total_loss = 0.0
        self.total_frames = 0

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
        loss: float | Tensor,
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
                "logits and targets must have the same shape, got "
                f"{tuple(logits_bt.shape)} and {tuple(targets_bt.shape)}"
            )

        valid_mask = self._build_valid_mask(targets_bt, mask)

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
        self.confusion.update(probs_valid, targets_valid)

        num_valid = int(targets_valid.numel())
        loss_value = float(loss.detach().item()) if isinstance(loss, Tensor) else float(loss)

        self.total_loss += loss_value * num_valid
        self.total_frames += num_valid

    def compute(self) -> BinaryClassificationMetrics:
        """
        Compute final epoch metrics.

        Returns:
            Aggregated frame-level binary classification metrics.
        """
        if self.total_frames == 0:
            return BinaryClassificationMetrics.empty()

        confusion = self.confusion.compute()
        tn = int(confusion[0, 0].item())
        fp = int(confusion[0, 1].item())
        fn = int(confusion[1, 0].item())
        tp = int(confusion[1, 1].item())

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return BinaryClassificationMetrics(
            loss=self.total_loss / self.total_frames,
            accuracy=float(self.accuracy.compute().item()),
            precision=float(self.precision.compute().item()),
            recall=float(self.recall.compute().item()),
            f1=float(self.f1.compute().item()),
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            miss_rate=false_negative_rate,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            num_frames=self.total_frames,
        )

    @staticmethod
    def _flatten_logits(logits: Tensor) -> Tensor:
        """
        Normalize logits to shape [B, T].

        Args:
            logits: Tensor of shape [B, T] or [B, 1, T].

        Returns:
            Logits with shape [B, T].

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
            Unchanged targets tensor.

        Raises:
            ValueError: If targets do not have shape [B, T].
        """
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape [B, T], got {tuple(targets.shape)}")
        return targets

    @staticmethod
    def _build_valid_mask(targets: Tensor, mask: Tensor | None) -> Tensor:
        """
        Build a boolean validity mask.

        Args:
            targets: Target tensor with shape [B, T].
            mask: Optional boolean mask with shape [B, T].

        Returns:
            Boolean mask with shape [B, T], where True marks valid frames.

        Raises:
            ValueError: If mask shape does not match targets.
        """
        if mask is None:
            return torch.ones_like(targets, dtype=torch.bool)

        if mask.shape != targets.shape:
            raise ValueError(
                f"mask and targets must have the same shape, got "
                f"{tuple(mask.shape)} and {tuple(targets.shape)}"
            )

        return mask.to(dtype=torch.bool)
