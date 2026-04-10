from __future__ import annotations

from collections.abc import Iterable

from vad.data.preprocessing.labels import LabelAligner
from vad.training.metrics import BinaryClassificationMetrics, VADMetricsTracker


def evaluate_model(dataset: Iterable, model) -> BinaryClassificationMetrics:
    """
    Evaluate a VAD model over a dataset using global frame-level aggregation.

    Ground-truth sample-level labels are aligned to frame-level targets before
    metric computation. Metrics are aggregated globally across all files, so
    longer sequences contribute proportionally more than shorter ones.

    Args:
        dataset: Iterable yielding ``(waveform, target, sample_rate)``.
        model: Object exposing ``predict_waveform(waveform, sample_rate)``.

    Returns:
        Aggregated binary classification metrics over the full dataset.

    Raises:
        ValueError: If a large prediction/target misalignment is detected.
    """
    # FIXME: hard coded; should ideally come from the model / feature config
    aligner = LabelAligner(hop_length=160, frame_length=400, center=False)
    tracker = VADMetricsTracker()

    for waveform, target, sample_rate in dataset:
        pred = model.predict_waveform(waveform, sample_rate).predictions.cpu()
        target_frames = aligner(target, num_frames=len(pred)).cpu()

        if abs(len(pred) - len(target_frames)) > 2:
            raise ValueError(
                "Large misalignment detected: "
                f"pred={len(pred)} frames, target={len(target_frames)} frames"
            )

        if len(pred) != len(target_frames):
            n = min(len(pred), len(target_frames))
            pred = pred[:n]
            target_frames = target_frames[:n]

        tracker.update_from_predictions(
            predictions=pred.unsqueeze(0),
            targets=target_frames.unsqueeze(0),
        )

    return tracker.compute()
