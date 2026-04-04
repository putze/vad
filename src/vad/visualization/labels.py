from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from src.vad.visualization.helpers import (
    binarize_labels,
    extract_time_slice,
    to_numpy_1d,
    validate_1d_tensor,
)


def plot_label_timeline(
    labels: Tensor,
    sr: int,
    title: str = "Sample-Level Label Timeline",
    figsize: tuple[int, int] = (14, 2.5),
    start_s: float | None = None,
    end_s: float | None = None,
    mode: Literal["step", "fill"] = "step",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot sample-level labels as a time series.

    Args:
        labels (Tensor): 1D tensor of sample-level labels.
        sr (int): Sampling rate in Hz.
        title (str): Plot title.
        figsize (tuple[int, int]): Figure size.
        start_s (float | None): Start time in seconds.
        end_s (float | None): End time in seconds.
        mode (Literal["step", "fill"]): Plot style.
        show (bool): Whether to display the plot.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
    """
    validate_1d_tensor(labels, "labels")

    labels_np = binarize_labels(to_numpy_1d(labels))
    dummy_audio = np.zeros_like(labels_np, dtype=np.float32)

    _, labels_np, time = extract_time_slice(
        audio=dummy_audio,
        labels=labels_np,
        sr=sr,
        start_s=start_s,
        end_s=end_s,
    )

    fig, ax = plt.subplots(figsize=figsize)

    if mode == "step":
        ax.step(time, labels_np, where="post")
    elif mode == "fill":
        ax.fill_between(time, labels_np, step="post", alpha=0.4)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Label")
    ax.set_ylim(-0.1, 1.1)

    if len(time) > 0:
        ax.set_xlim(time[0], time[-1] if len(time) > 1 else time[0] + 1.0)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax
