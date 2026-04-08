from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from vad.visualization.helpers import (
    binarize_labels,
    extract_time_slice,
    shade_positive_regions,
    to_numpy_1d,
    validate_1d_tensor,
)
from vad.visualization.style import set_plot_style


def plot_waveform_with_labels(
    audio: Tensor,
    labels: Tensor,
    sr: int,
    title: str = "Waveform with Sample-Level Labels",
    figsize: tuple[int, int] = (14, 4),
    start_s: float | None = None,
    end_s: float | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a waveform with sample-level labels highlighted.

    Args:
        audio (Tensor): 1D waveform [T].
        labels (Tensor): 1D sample-level labels [T].
        sr (int): Sampling rate in Hz.
        title (str): Plot title.
        figsize (tuple[int, int]): Figure size.
        start_s (float | None): Start time in seconds.
        end_s (float | None): End time in seconds.
        show (bool): Whether to display the plot.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
    """
    validate_1d_tensor(audio, "audio")
    validate_1d_tensor(labels, "labels")

    if len(audio) != len(labels):
        raise ValueError(
            f"`audio` and `labels` must have the same length, got {len(audio)} and {len(labels)}"
        )

    audio_np = to_numpy_1d(audio).astype("float32")
    labels_np = binarize_labels(to_numpy_1d(labels))

    audio_np, labels_np, time = extract_time_slice(
        audio=audio_np,
        labels=labels_np,
        sr=sr,
        start_s=start_s,
        end_s=end_s,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, audio_np, linewidth=0.8, label="Waveform")
    shade_positive_regions(ax=ax, labels=labels_np, time=time, sr=sr)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    if len(time) > 0:
        ax.set_xlim(time[0], time[-1] if len(time) > 1 else time[0] + 1.0)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def print_sample_debug_info(
    audio: Tensor,
    labels: Tensor,
    sr: int,
    sample_name: str | None = None,
) -> None:
    """
    Print summary information for an audio sample and labels.

    Args:
        audio (Tensor): Waveform [T].
        labels (Tensor): Sample-level labels [T].
        sr (int): Sampling rate in Hz.
        sample_name (str | None): Optional sample identifier.
    """
    validate_1d_tensor(audio, "audio")
    validate_1d_tensor(labels, "labels")

    if len(audio) != len(labels):
        raise ValueError(
            f"`audio` and `labels` must have the same length, got {len(audio)} and {len(labels)}"
        )

    header = "Sample debug info"
    if sample_name is not None:
        header += f" - {sample_name}"

    unique_labels = torch.unique(labels)

    print(header)
    print(f"audio shape: {tuple(audio.shape)}")
    print(f"labels shape: {tuple(labels.shape)}")
    print(f"sample rate: {sr}")
    print(f"duration (s): {len(audio) / sr:.3f}")
    print(f"audio min/max: {audio.min().item():.5f} / {audio.max().item():.5f}")
    print(f"label unique values: {unique_labels}")
    print(f"positive samples: {int((labels > 0).sum().item())}")
    print(f"negative samples: {int((labels <= 0).sum().item())}")


def debug_plot_waveform_with_labels(
    audio: Tensor,
    labels: Tensor,
    sr: int,
    sample_name: str | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    use_seaborn: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Print debug info and plot waveform with labels.

    Args:
        audio (Tensor): Waveform [T].
        labels (Tensor): Sample-level labels [T].
        sr (int): Sampling rate in Hz.
        sample_name (str | None): Optional sample identifier.
        start_s (float | None): Start time in seconds.
        end_s (float | None): End time in seconds.
        use_seaborn (bool): Whether to apply Seaborn styling.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
    """
    set_plot_style(use_seaborn=use_seaborn)
    print_sample_debug_info(audio=audio, labels=labels, sr=sr, sample_name=sample_name)

    title = "Waveform with Sample-Level Labels"
    if sample_name is not None:
        title += f" - {sample_name}"

    return plot_waveform_with_labels(
        audio=audio,
        labels=labels,
        sr=sr,
        title=title,
        start_s=start_s,
        end_s=end_s,
        show=True,
    )
