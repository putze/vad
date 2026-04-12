from __future__ import annotations

from collections.abc import Iterable

from vad.data.preprocessing.labels import LabelAligner
from vad.training.metrics import VADMetricsTracker


def evaluate_binary_model(dataset: Iterable, model):
    """
    Evaluate a hard-decision VAD model over a dataset.
    Returns BinaryClassificationMetrics.
    """
    aligner = LabelAligner(hop_length=160, frame_length=400, center=False)
    tracker = VADMetricsTracker()

    for waveform, target, sample_rate in dataset:
        pred = model.predict_waveform(waveform, sample_rate).predictions.cpu()
        target_frames = aligner(target, num_frames=len(pred)).cpu()

        if len(pred) != len(target_frames):
            n = min(len(pred), len(target_frames))
            pred = pred[:n]
            target_frames = target_frames[:n]

        tracker.update_from_predictions(
            predictions=pred.unsqueeze(0),
            targets=target_frames.unsqueeze(0),
        )

    return tracker.compute()


def evaluate_webrtc_operating_points(
    dataset_factory,
    make_webrtc_model,
    aggressiveness_values: list[int] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Evaluate WebRTC VAD aggressiveness levels as ROC operating points.

    Args:
        dataset_factory: Callable producing a fresh iterable dataset each time.
            Use a factory because some iterables may be exhausted after one pass.
        make_webrtc_model: Callable taking aggressiveness -> model instance.
        aggressiveness_values: Levels to evaluate.

    Returns:
        Dict mapping label to (fpr, tpr).
    """
    if aggressiveness_values is None:
        aggressiveness_values = [0, 1, 2, 3]

    points: dict[str, tuple[float, float]] = {}

    for level in aggressiveness_values:
        dataset = dataset_factory()
        model = make_webrtc_model(level)
        metrics = evaluate_binary_model(dataset, model)
        points[f"WebRTC {level}"] = (
            metrics.false_positive_rate,
            metrics.recall,
        )

    return points
