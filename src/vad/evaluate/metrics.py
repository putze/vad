from __future__ import annotations

import torch
from torch import Tensor


def binary_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    pred = pred.to(torch.int64)
    target = target.to(torch.int64)

    tp = int(((pred == 1) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
    }
