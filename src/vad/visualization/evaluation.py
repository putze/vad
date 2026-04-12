from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


@dataclass(slots=True)
class RocCurveResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


def compute_roc_curve(
    targets: np.ndarray,
    scores: np.ndarray,
) -> RocCurveResult:
    """
    Compute ROC curve and AUC from binary targets and continuous scores.
    """
    fpr, tpr, thresholds = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    return RocCurveResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=float(roc_auc),
    )


def plot_roc_curve(
    roc_result: RocCurveResult,
    operating_points: dict[str, tuple[float, float]] | None = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """
    Plot ROC curve with optional operating points.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        roc_result.fpr,
        roc_result.tpr,
        label=f"Model ROC (AUC = {roc_result.auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    if operating_points is not None:
        for label, (fpr, tpr) in operating_points.items():
            ax.scatter([fpr], [tpr], s=60, label=label)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig
