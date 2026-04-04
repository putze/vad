from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def validate_1d_tensor(x: Tensor, name: str) -> None:
    """
    Validate that a tensor is 1D.

    Args:
        x (Tensor): Tensor to validate.
        name (str): Name used in error messages.
    """
    if x.ndim != 1:
        raise ValueError(f"`{name}` must be 1D, got shape {tuple(x.shape)}")


def validate_2d_tensor(x: Tensor, name: str) -> None:
    """
    Validate that a tensor is 2D.

    Args:
        x (Tensor): Tensor to validate.
        name (str): Name used in error messages.
    """
    if x.ndim != 2:
        raise ValueError(f"`{name}` must be 2D, got shape {tuple(x.shape)}")


def to_numpy_1d(x: Tensor) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.

    Args:
        x (Tensor): Input tensor.

    Returns:
        np.ndarray: Detached CPU NumPy array.
    """
    return x.detach().cpu().numpy()


def to_numpy_2d(x: Tensor) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.

    Args:
        x (Tensor): Input tensor.

    Returns:
        np.ndarray: Detached CPU NumPy array.
    """
    return x.detach().cpu().numpy()


def binarize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert labels to binary values.

    Args:
        labels (np.ndarray): Input label array.

    Returns:
        np.ndarray: Binary label array.
    """
    return (labels > 0).astype(np.int32)


def extract_time_slice(
    audio: np.ndarray,
    labels: np.ndarray,
    sr: int,
    start_s: float | None,
    end_s: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a time slice from audio and labels.

    Args:
        audio (np.ndarray): Audio waveform.
        labels (np.ndarray): Label array.
        sr (int): Sampling rate in Hz.
        start_s (float | None): Start time in seconds.
        end_s (float | None): End time in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            Sliced audio, sliced labels, and time axis.
    """
    num_samples = len(audio)

    start_idx = 0 if start_s is None else max(0, int(start_s * sr))
    end_idx = num_samples if end_s is None else min(num_samples, int(end_s * sr))

    if end_idx <= start_idx:
        raise ValueError(f"Invalid time slice: start_s={start_s}, end_s={end_s}")

    audio_slice = audio[start_idx:end_idx]
    label_slice = labels[start_idx:end_idx]
    time = np.arange(start_idx, end_idx) / sr

    return audio_slice, label_slice, time


def validate_frame_alignment(
    sequence: Tensor,
    frame_labels: Tensor,
    time_dim: int = 1,
    sequence_name: str = "sequence",
) -> None:
    """
    Validate that a 2D sequence matches frame-level labels.

    Args:
        sequence (Tensor): Input sequence with a time dimension.
        frame_labels (Tensor): Frame-level labels.
        time_dim (int): Dimension corresponding to frames.
        sequence_name (str): Name used in error messages.
    """
    validate_2d_tensor(sequence, sequence_name)
    validate_1d_tensor(frame_labels, "frame_labels")

    num_frames = sequence.shape[time_dim]
    if len(frame_labels) != num_frames:
        raise ValueError(
            f"{sequence_name} has {num_frames} frames along dim {time_dim}, "
            f"but `frame_labels` has length {len(frame_labels)}"
        )


def shade_positive_regions(
    ax: plt.Axes,
    labels: np.ndarray,
    time: np.ndarray,
    sr: int,
    alpha: float = 0.25,
) -> None:
    """
    Shade time regions where labels are positive.

    Args:
        ax (plt.Axes): Target axes.
        labels (np.ndarray): Binary label array.
        time (np.ndarray): Time axis in seconds.
        sr (int): Sampling rate in Hz.
        alpha (float): Region transparency.
    """
    if len(labels) == 0:
        return

    in_region = False
    start_idx = 0
    base_sample_idx = int(round(time[0] * sr)) if len(time) > 0 else 0

    for i, value in enumerate(labels):
        if value == 1 and not in_region:
            in_region = True
            start_idx = i
        elif value == 0 and in_region:
            start_t = (base_sample_idx + start_idx) / sr
            end_t = (base_sample_idx + i) / sr
            ax.axvspan(start_t, end_t, alpha=alpha)
            in_region = False

    if in_region:
        start_t = (base_sample_idx + start_idx) / sr
        end_t = (base_sample_idx + len(labels)) / sr
        ax.axvspan(start_t, end_t, alpha=alpha)


def shade_positive_frames(
    ax: plt.Axes,
    labels: np.ndarray,
    alpha: float = 0.25,
) -> None:
    """
    Shade frame regions where labels are positive.

    Args:
        ax (plt.Axes): Target axes.
        labels (np.ndarray): Binary frame-level label array.
        alpha (float): Region transparency.
    """
    if len(labels) == 0:
        return

    in_region = False
    start_idx = 0

    for i, value in enumerate(labels):
        if value == 1 and not in_region:
            in_region = True
            start_idx = i
        elif value == 0 and in_region:
            ax.axvspan(start_idx - 0.5, i - 0.5, alpha=alpha)
            in_region = False

    if in_region:
        ax.axvspan(start_idx - 0.5, len(labels) - 0.5, alpha=alpha)
