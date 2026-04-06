from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def validate_1d_tensor(x: Tensor, name: str) -> None:
    """
    Validate that a tensor is one-dimensional.

    Args:
        x: Tensor to validate.
        name: Tensor name used in error messages.

    Raises:
        ValueError: If the tensor is not 1D.
    """
    if x.ndim != 1:
        raise ValueError(f"`{name}` must be 1D, got shape {tuple(x.shape)}")


def validate_2d_tensor(x: Tensor, name: str) -> None:
    """
    Validate that a tensor is two-dimensional.

    Args:
        x: Tensor to validate.
        name: Tensor name used in error messages.

    Raises:
        ValueError: If the tensor is not 2D.
    """
    if x.ndim != 2:
        raise ValueError(f"`{name}` must be 2D, got shape {tuple(x.shape)}")


def validate_equal_length_1d(x: Tensor, y: Tensor, x_name: str, y_name: str) -> None:
    """
    Validate that two 1D tensors have the same length.

    Args:
        x: First tensor.
        y: Second tensor.
        x_name: Name of the first tensor.
        y_name: Name of the second tensor.

    Raises:
        ValueError: If either tensor is not 1D or their lengths differ.
    """
    validate_1d_tensor(x, x_name)
    validate_1d_tensor(y, y_name)
    if len(x) != len(y):
        raise ValueError(
            f"`{x_name}` and `{y_name}` must have the same length, got {len(x)} and {len(y)}"
        )


def to_numpy_1d(x: Tensor) -> NDArray[np.float32]:
    """
    Convert a tensor to a NumPy array.

    Args:
        x: Input tensor.

    Returns:
        Detached CPU NumPy array.
    """
    return cast(NDArray[np.float32], x.detach().cpu().numpy())


def to_numpy_2d(x: Tensor) -> NDArray[np.float32]:
    """
    Convert a tensor to a NumPy array.

    Args:
        x: Input tensor.

    Returns:
        Detached CPU NumPy array.
    """
    return cast(NDArray[np.float32], x.detach().cpu().numpy())


def binarize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert labels to binary values.

    Args:
        labels: Input label array.

    Returns:
        Binary label array.
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
        audio: Audio waveform.
        labels: Label array aligned with the waveform.
        sr: Sample rate in Hz.
        start_s: Start time in seconds.
        end_s: End time in seconds.

    Returns:
        Tuple containing the sliced audio, sliced labels, and time axis.

    Raises:
        ValueError: If the requested time slice is invalid.
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
    Validate that a sequence matches frame-level labels.

    Args:
        sequence: Input sequence with a frame dimension.
        frame_labels: Frame-level labels.
        time_dim: Dimension corresponding to frames.
        sequence_name: Sequence name used in error messages.

    Raises:
        ValueError: If the sequence shape is invalid or frame counts differ.
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
    Shade waveform regions where labels are positive.

    Args:
        ax: Target axes.
        labels: Binary label array.
        time: Time axis in seconds.
        sr: Sample rate in Hz.
        alpha: Shading transparency.
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
    Shade frame indices where labels are positive.

    Args:
        ax: Target axes.
        labels: Binary frame-level label array.
        alpha: Shading transparency.
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


def shade_positive_frame_regions_seconds(
    ax: plt.Axes,
    labels: np.ndarray,
    frame_hop_s: float,
    alpha: float = 0.25,
) -> None:
    """
    Shade positive frame regions on a time axis.

    Args:
        ax: Target axes.
        labels: Binary frame-level label array.
        frame_hop_s: Frame hop duration in seconds.
        alpha: Shading transparency.
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
            ax.axvspan(start_idx * frame_hop_s, i * frame_hop_s, alpha=alpha)
            in_region = False

    if in_region:
        ax.axvspan(start_idx * frame_hop_s, len(labels) * frame_hop_s, alpha=alpha)


def build_waveform_time_axis(num_samples: int, sr: int) -> np.ndarray:
    """
    Build a waveform time axis in seconds.

    Args:
        num_samples: Number of waveform samples.
        sr: Sample rate in Hz.

    Returns:
        Time axis in seconds.
    """
    return np.arange(num_samples, dtype=np.float32) / float(sr)


def build_frame_time_axis(num_frames: int, frame_hop_s: float) -> np.ndarray:
    """
    Build a frame time axis in seconds.

    Args:
        num_frames: Number of frames.
        frame_hop_s: Frame hop duration in seconds.

    Returns:
        Time axis in seconds.
    """
    return np.arange(num_frames, dtype=np.float32) * float(frame_hop_s)
