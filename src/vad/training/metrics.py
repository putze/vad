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
        accuracy: Fraction of correctly classified frames.
        precision: Positive predictive value, defined as TP / (TP + FP).
        recall: True positive rate, defined as TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
        false_positive_rate: Fraction of non-speech frames predicted as speech,
            defined as FP / (FP + TN).
        false_negative_rate: Fraction of speech frames predicted as non-speech,
            defined as FN / (FN + TP).
        tp: Number of true positive frames.
        fp: Number of false positive frames.
        tn: Number of true negative frames.
        fn: Number of false negative frames.
        num_frames: Number of valid frames included in the computation.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    num_frames: int

    @classmethod
    def empty(cls) -> BinaryClassificationMetrics:
        """
        Return a zero-initialized metrics object.

        Returns:
            A metrics object with all scalar values set to zero.
        """
        return cls(
            loss=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            num_frames=0,
        )


class VADMetricsTracker:
    """
    Accumulate frame-level binary classification metrics for VAD.

    This tracker supports two update modes:
        - ``update_from_logits`` for training/validation with raw model logits
        - ``update_from_predictions`` for evaluation with hard binary predictions

    Expected shapes:
        logits:       [B, T] or [B, 1, T]
        predictions:  [B, T]
        targets:      [B, T]
        mask:         [B, T], where True indicates a valid frame

    Notes:
        - ``update_from_logits`` applies sigmoid internally and accumulates loss.
        - ``update_from_predictions`` assumes predictions are already binary and
          does not accumulate loss.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize the metrics tracker.

        Args:
            threshold: Probability threshold used to convert sigmoid outputs
                into binary predictions.

        Raises:
            ValueError: If ``threshold`` is outside the interval [0, 1].
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.threshold = threshold

        self.accuracy = BinaryAccuracy(threshold=threshold)
        self.precision = BinaryPrecision(threshold=threshold)
        self.recall = BinaryRecall(threshold=threshold)
        self.f1 = BinaryF1Score(threshold=threshold)
        self.confusion = BinaryConfusionMatrix(threshold=threshold)

        self.reset()

    def reset(self) -> None:
        """
        Reset all metric states and accumulated loss statistics.
        """
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confusion.reset()

        self.total_loss = 0.0
        self.total_frames = 0
        self.loss_frames = 0

    def update_from_logits(
        self,
        logits: Tensor,
        targets: Tensor,
        loss: float | Tensor,
        mask: Tensor | None = None,
    ) -> None:
        """
        Update tracker state from raw logits.

        Args:
            logits: Model outputs before sigmoid, with shape [B, T] or [B, 1, T].
            targets: Binary frame labels with shape [B, T].
            loss: Scalar batch loss. It is assumed to represent the mean loss
                over valid frames in the batch.
            mask: Optional boolean mask of shape [B, T], where True marks valid
                frames.

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        logits_bt = self._flatten_logits(logits)
        targets_bt = self._flatten_targets(targets)
        valid_mask = self._build_valid_mask(targets_bt, mask)

        if logits_bt.shape != targets_bt.shape:
            raise ValueError(
                "logits and targets must have the same shape, got "
                f"{tuple(logits_bt.shape)} and {tuple(targets_bt.shape)}"
            )

        logits_valid = logits_bt[valid_mask]
        targets_valid = targets_bt[valid_mask]

        if logits_valid.numel() == 0:
            return

        probs_valid = torch.sigmoid(logits_valid)
        targets_valid = targets_valid.to(dtype=torch.long)

        self._update_metric_states(probs_valid, targets_valid)

        num_valid = int(targets_valid.numel())
        loss_value = float(loss.detach().item()) if isinstance(loss, Tensor) else float(loss)

        self.total_loss += loss_value * num_valid
        self.loss_frames += num_valid
        self.total_frames += num_valid

    def update_from_predictions(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> None:
        """
        Update tracker state from hard binary predictions.

        Args:
            predictions: Binary frame predictions with shape [B, T].
            targets: Binary frame labels with shape [B, T].
            mask: Optional boolean mask of shape [B, T], where True marks valid
                frames.

        Raises:
            ValueError: If tensor shapes are incompatible or if predictions /
                targets contain values other than 0 and 1.
        """
        predictions_bt = self._flatten_targets(predictions)
        targets_bt = self._flatten_targets(targets)
        valid_mask = self._build_valid_mask(targets_bt, mask)

        if predictions_bt.shape != targets_bt.shape:
            raise ValueError(
                "predictions and targets must have the same shape, got "
                f"{tuple(predictions_bt.shape)} and {tuple(targets_bt.shape)}"
            )

        predictions_valid = predictions_bt[valid_mask].to(dtype=torch.long)
        targets_valid = targets_bt[valid_mask].to(dtype=torch.long)

        if predictions_valid.numel() == 0:
            return

        self._validate_binary_tensor(predictions_valid, name="predictions")
        self._validate_binary_tensor(targets_valid, name="targets")

        # TorchMetrics accepts integer predictions directly for binary metrics.
        self._update_metric_states(predictions_valid, targets_valid)

        self.total_frames += int(targets_valid.numel())

    def compute(self) -> BinaryClassificationMetrics:
        """
        Compute aggregated metrics over all accumulated updates.

        Returns:
            A dataclass containing loss, standard binary classification metrics,
            confusion counts, and derived error rates.
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
        mean_loss = self.total_loss / self.loss_frames if self.loss_frames > 0 else 0.0

        return BinaryClassificationMetrics(
            loss=mean_loss,
            accuracy=float(self.accuracy.compute().item()),
            precision=float(self.precision.compute().item()),
            recall=float(self.recall.compute().item()),
            f1=float(self.f1.compute().item()),
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            num_frames=self.total_frames,
        )

    def _update_metric_states(self, preds_or_probs: Tensor, targets: Tensor) -> None:
        """
        Update all TorchMetrics states.

        Args:
            preds_or_probs: Either binary predictions or probabilities.
            targets: Binary targets as integer tensor.
        """
        self.accuracy.update(preds_or_probs, targets)
        self.precision.update(preds_or_probs, targets)
        self.recall.update(preds_or_probs, targets)
        self.f1.update(preds_or_probs, targets)
        self.confusion.update(preds_or_probs, targets)

    @staticmethod
    def _flatten_logits(logits: Tensor) -> Tensor:
        """
        Normalize logits to shape [B, T].

        Args:
            logits: Tensor of shape [B, T] or [B, 1, T].

        Returns:
            Logits reshaped to [B, T].

        Raises:
            ValueError: If ``logits`` has an unsupported shape.
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
        Validate that a tensor has shape [B, T].

        Args:
            targets: Tensor expected to have shape [B, T].

        Returns:
            The unchanged tensor.

        Raises:
            ValueError: If the tensor does not have shape [B, T].
        """
        if targets.ndim != 2:
            raise ValueError(f"Expected tensor with shape [B, T], got {tuple(targets.shape)}")
        return targets

    @staticmethod
    def _build_valid_mask(targets: Tensor, mask: Tensor | None) -> Tensor:
        """
        Build a boolean mask indicating which frames are valid.

        Args:
            targets: Target tensor with shape [B, T].
            mask: Optional mask tensor with shape [B, T].

        Returns:
            A boolean tensor of shape [B, T], where True marks valid frames.

        Raises:
            ValueError: If ``mask`` shape does not match ``targets``.
        """
        if mask is None:
            return torch.ones_like(targets, dtype=torch.bool)

        if mask.shape != targets.shape:
            raise ValueError(
                f"mask and targets must have the same shape, got "
                f"{tuple(mask.shape)} and {tuple(targets.shape)}"
            )

        return mask.to(dtype=torch.bool)

    @staticmethod
    def _validate_binary_tensor(values: Tensor, name: str) -> None:
        """
        Validate that a tensor contains only binary values 0 and 1.

        Args:
            values: Tensor to validate.
            name: Name used in the error message.

        Raises:
            ValueError: If non-binary values are found.
        """
        if not torch.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} must contain only binary values {{0, 1}}")
