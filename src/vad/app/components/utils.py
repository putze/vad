from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from vad.app.state import StreamingState
from vad.data.audio_utils import ensure_mono_waveform
from vad.inference.offline import OfflineVADPrediction
from vad.inference.streaming import StreamingPrediction


def chunk_waveform(
    waveform: Tensor,
    sample_rate: int,
    chunk_seconds: float,
) -> list[Tensor]:
    """
    Split a mono waveform into fixed-size chunks.

    Args:
        waveform: Audio tensor of shape [N] or [C, N].
        sample_rate: Sampling rate in Hz.
        chunk_seconds: Duration of each chunk in seconds.

    Returns:
        List of waveform chunks, each of shape [chunk_size] (last may be shorter).
    """
    # Ensure mono
    waveform = ensure_mono_waveform(waveform)

    chunk_size = max(1, int(round(chunk_seconds * sample_rate)))

    chunks = [waveform[i : i + chunk_size] for i in range(0, waveform.numel(), chunk_size)]

    return chunks


def append_chunk_to_state(
    state: StreamingState,
    chunk_waveform: torch.Tensor,
    sample_rate: int,
    chunk_prediction: OfflineVADPrediction | StreamingPrediction,
    max_window_seconds: float,
) -> None:
    """Append a chunk result and keep only the last max_window_seconds."""
    mono = ensure_mono_waveform(chunk_waveform).detach().cpu().numpy()
    chunk_duration = mono.shape[0] / float(sample_rate)

    start_t = state.waveform_times[-1] + (1.0 / sample_rate) if state.waveform_times else 0.0
    waveform_times = start_t + np.arange(mono.shape[0], dtype=np.float32) / float(sample_rate)

    pred_times = chunk_prediction.frame_times.detach().cpu().numpy()
    if len(state.times) > 0 and len(pred_times) > 0:
        offset = state.times[-1] + (
            pred_times[1] - pred_times[0] if len(pred_times) > 1 else chunk_duration
        )
        pred_times = pred_times + offset

    state.waveform_times.extend(waveform_times.tolist())
    state.waveform_values.extend(mono.tolist())
    state.times.extend(pred_times.tolist())
    state.probabilities.extend(chunk_prediction.probabilities.detach().cpu().numpy().tolist())
    state.predictions.extend(
        chunk_prediction.predictions.detach().cpu().numpy().astype(int).tolist()
    )

    min_t = max(
        0.0, (state.waveform_times[-1] if state.waveform_times else 0.0) - max_window_seconds
    )

    waveform_mask = [t >= min_t for t in state.waveform_times]
    state.waveform_times = [t for t, keep in zip(state.waveform_times, waveform_mask) if keep]
    state.waveform_values = [v for v, keep in zip(state.waveform_values, waveform_mask) if keep]

    pred_mask = [t >= min_t for t in state.times]
    state.times = [t for t, keep in zip(state.times, pred_mask) if keep]
    state.probabilities = [v for v, keep in zip(state.probabilities, pred_mask) if keep]
    state.predictions = [v for v, keep in zip(state.predictions, pred_mask) if keep]
