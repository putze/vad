from __future__ import annotations

from collections.abc import Iterable

from src.vad.data.preprocessing import LabelAligner
from src.vad.evaluate.metrics import binary_metrics


def evaluate_model(dataset: Iterable, model) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0

    # FIXME: hard coded
    aligner = LabelAligner(hop_length=160, frame_length=400, center=False)

    for waveform, target, sample_rate in dataset:
        pred = model.predict_waveform(waveform, sample_rate).predictions
        target_frames = aligner(target, num_frames=len(pred))

        if abs(len(pred) - len(target_frames)) > 2:
            raise ValueError("Large misalignment detected")

        # Trim small differences
        if len(pred) != len(target_frames):
            n = min(len(pred), len(target_frames))
            pred = pred[:n]
            target_frames = target_frames[:n]

        metrics = binary_metrics(pred, target_frames)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        count += 1

    return {k: v / max(count, 1) for k, v in totals.items()}
