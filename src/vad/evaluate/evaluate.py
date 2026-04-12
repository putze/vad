from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from vad.data.preprocessing.labels import LabelAligner
from vad.training.metrics import BinaryClassificationMetrics, VADMetricsTracker


@dataclass(slots=True)
class ScoreEvaluationResult:
    """
    Aggregated evaluation outputs for binary VAD models.

    Attributes:
        metrics: Global frame-level thresholded metrics.
        targets: Concatenated frame-level ground-truth labels.
        scores: Concatenated continuous speech scores if available, else None.
        predictions: Concatenated hard binary predictions.
    """

    metrics: BinaryClassificationMetrics
    targets: np.ndarray
    scores: np.ndarray | None
    predictions: np.ndarray


def evaluate_model(dataset: Iterable, model) -> ScoreEvaluationResult:
    """
    Evaluate a VAD model over a dataset using global frame-level aggregation.

    Ground-truth sample-level labels are aligned to frame-level targets before
    metric computation. Metrics are aggregated globally across all files, so
    longer sequences contribute proportionally more than shorter ones.

    This evaluator supports:
    - score-based models exposing ``.probabilities`` and ``.predictions``
    - hard-decision models exposing only ``.predictions``

    Args:
        dataset: Iterable yielding ``(waveform, target, sample_rate)``.
        model: Object exposing ``predict_waveform(waveform, sample_rate)``.

    Returns:
        ScoreEvaluationResult containing global threshold metrics and raw
        concatenated arrays for further analysis such as ROC/AUC.

    Raises:
        ValueError: If a large prediction/target misalignment is detected.
        AttributeError: If the model output does not expose ``.predictions``.
    """
    # FIXME: hard coded; should ideally come from the model / feature config
    aligner = LabelAligner(hop_length=160, frame_length=400, center=False)
    tracker = VADMetricsTracker()

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    has_scores = True

    for waveform, target, sample_rate in dataset:
        output = model.predict_waveform(waveform, sample_rate)

        if not hasattr(output, "predictions"):
            raise AttributeError("Model prediction output must expose `.predictions`.")

        pred = output.predictions.detach().cpu().reshape(-1)
        target_frames = aligner(target, num_frames=len(pred)).cpu().reshape(-1)

        scores = None
        if hasattr(output, "probabilities") and output.probabilities is not None:
            scores = output.probabilities.detach().cpu().reshape(-1)
        else:
            has_scores = False

        if abs(len(pred) - len(target_frames)) > 2:
            raise ValueError(
                "Large misalignment detected: "
                f"pred={len(pred)} frames, target={len(target_frames)} frames"
            )

        if len(pred) != len(target_frames):
            n = min(len(pred), len(target_frames))
            pred = pred[:n]
            target_frames = target_frames[:n]
            if scores is not None:
                scores = scores[:n]

        tracker.update_from_predictions(
            predictions=pred.unsqueeze(0),
            targets=target_frames.unsqueeze(0),
        )

        all_targets.append(target_frames.numpy().astype(np.int64))
        all_predictions.append(pred.numpy().astype(np.int64))

        if has_scores and scores is not None:
            all_scores.append(scores.numpy().astype(np.float32))

    return ScoreEvaluationResult(
        metrics=tracker.compute(),
        targets=np.concatenate(all_targets),
        scores=np.concatenate(all_scores) if has_scores and all_scores else None,
        predictions=np.concatenate(all_predictions),
    )
