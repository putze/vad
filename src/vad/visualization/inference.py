from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from vad.data.utils import ensure_mono_waveform
from vad.visualization.style import set_plot_style


def plot_offline_vad_prediction(
    waveform: Tensor,
    sample_rate: int,
    probabilities: Tensor,
    predictions: Tensor,
    frame_hop_s: float,
    threshold: float = 0.5,
    title: str = "Offline VAD prediction",
    figsize: tuple[int, int] = (14, 8),
    show: bool = True,
    use_seaborn: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot waveform, speech probabilities, and binary VAD predictions.

    Args:
        waveform: Input waveform.
        sample_rate: Waveform sample rate in Hz.
        probabilities: Frame-level speech probabilities.
        predictions: Frame-level binary predictions.
        frame_hop_s: Frame hop duration in seconds.
        threshold: Decision threshold for speech detection.
        title: Plot title.
        figsize: Figure size.
        show: Whether to display the plot.
        use_seaborn: Whether to apply seaborn styling.

    Returns:
        Tuple containing the matplotlib figure and list of axes.
    """
    set_plot_style(use_seaborn=use_seaborn)

    waveform = ensure_mono_waveform(waveform)

    if probabilities.ndim != 1:
        probabilities = probabilities.reshape(-1)
    if predictions.ndim != 1:
        predictions = predictions.reshape(-1)

    waveform_np = waveform.detach().cpu().numpy().astype(np.float32)
    prob_np = probabilities.detach().cpu().numpy().astype(np.float32)
    pred_np = (predictions.detach().cpu().numpy() > 0).astype(np.float32)

    waveform_time = np.arange(len(waveform_np), dtype=np.float32) / float(sample_rate)
    frame_time = np.arange(len(prob_np), dtype=np.float32) * float(frame_hop_s)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ax_wave, ax_prob, ax_pred = axes

    ax_wave.plot(waveform_time, waveform_np, linewidth=0.8)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(title)
    ax_wave.grid(True, alpha=0.3)

    ax_prob.plot(frame_time, prob_np, linewidth=1.2, label="speech_probability")
    ax_prob.axhline(threshold, linestyle="--", label="threshold")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend()
    ax_prob.grid(True, alpha=0.3)

    ax_pred.step(frame_time, pred_np, where="post")
    ax_pred.fill_between(frame_time, pred_np, step="post", alpha=0.3)
    ax_pred.set_ylabel("Prediction")
    ax_pred.set_xlabel("Time (s)")
    ax_pred.set_ylim(-0.1, 1.1)
    ax_pred.grid(True, alpha=0.3)

    waveform_duration = (
        float(len(waveform_np)) / float(sample_rate) if len(waveform_np) > 0 else 0.0
    )
    frame_duration = float(frame_time[-1] + frame_hop_s) if len(frame_time) > 0 else 0.0
    total_duration = max(waveform_duration, frame_duration, 1.0)
    axes[-1].set_xlim(0.0, total_duration)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, list(axes)
