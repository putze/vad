from __future__ import annotations

from torch import Tensor

from vad.training.metrics import BinaryClassificationMetrics, VADMetricsTracker


def binary_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    """
    Compute binary metrics from hard predictions and targets.

    This is a lightweight compatibility wrapper around ``VADMetricsTracker``.
    Both inputs must be binary tensors with identical shape.

    Args:
        pred: Binary predictions.
        target: Binary ground-truth labels.

    Returns:
        Dictionary containing accuracy, precision, recall, F1 score,
        false positive rate, and false negative rate.
    """
    tracker = VADMetricsTracker()
    tracker.update_from_predictions(
        predictions=pred.unsqueeze(0),
        targets=target.unsqueeze(0),
    )
    metrics = tracker.compute()
    return metrics_to_dict(metrics)


def metrics_to_dict(metrics: BinaryClassificationMetrics) -> dict[str, float]:
    """
    Convert a ``BinaryClassificationMetrics`` object into a plain dictionary.

    Args:
        metrics: Aggregated metrics object.

    Returns:
        Dictionary representation of the main scalar metrics.
    """
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "false_negative_rate": metrics.false_negative_rate,
    }
