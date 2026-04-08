from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from vad.visualization.helpers import (
    binarize_labels,
    to_numpy_1d,
    validate_1d_tensor,
)
from vad.visualization.style import set_plot_style


def compute_frame_boundaries(
    num_samples: int,
    num_frames: int,
    hop_length: int,
    frame_length: int,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sample index boundaries for each frame window.

    Args:
        num_samples (int): Number of samples in the original sequence.
        num_frames (int): Number of frames.
        hop_length (int): Number of samples between consecutive frames.
        frame_length (int): Number of samples per frame.
        center (bool): Whether frames are centered.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frame start and end indices, each of shape [num_frames].
    """
    frame_starts = np.zeros(num_frames, dtype=np.int64)
    frame_ends = np.zeros(num_frames, dtype=np.int64)

    if center:
        pad = frame_length // 2
        for i in range(num_frames):
            start = i * hop_length - pad
            end = start + frame_length
            frame_starts[i] = max(0, start)
            frame_ends[i] = min(num_samples, end)
    else:
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame_starts[i] = max(0, start)
            frame_ends[i] = min(num_samples, end)

    return frame_starts, frame_ends


def compute_frame_labels_from_samples(
    sample_labels: Tensor,
    num_frames: int,
    hop_length: int,
    frame_length: int,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recompute frame-level labels from sample-level labels.

    Args:
        sample_labels (Tensor): Sample-level labels [num_samples].
        num_frames (int): Number of output frames.
        hop_length (int): Number of samples between consecutive frames.
        frame_length (int): Number of samples per frame.
        center (bool): Whether frames are centered.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            Frame labels, frame starts, and frame ends.
    """
    validate_1d_tensor(sample_labels, "sample_labels")
    labels_np = binarize_labels(to_numpy_1d(sample_labels))

    frame_starts, frame_ends = compute_frame_boundaries(
        num_samples=len(labels_np),
        num_frames=num_frames,
        hop_length=hop_length,
        frame_length=frame_length,
        center=center,
    )

    frame_labels = np.zeros(num_frames, dtype=np.int32)

    for i, (start, end) in enumerate(zip(frame_starts, frame_ends)):
        if end > start:
            frame_labels[i] = int(labels_np[start:end].max())

    return frame_labels, frame_starts, frame_ends


def _validate_aligner_output(
    aligner_output: Tensor | None,
    num_frames: int,
) -> np.ndarray | None:
    """
    Validate and convert optional frame-level aligner output.

    Args:
        aligner_output (Tensor | None): Optional frame-level labels.
        num_frames (int): Expected number of frames.

    Returns:
        np.ndarray | None: Binary NumPy array of shape [num_frames], or None.
    """
    if aligner_output is None:
        return None

    validate_1d_tensor(aligner_output, "aligner_output")

    if len(aligner_output) != num_frames:
        raise ValueError(
            f"`aligner_output` must have length {num_frames}, got {len(aligner_output)}"
        )

    return (to_numpy_1d(aligner_output) > 0).astype(np.int32)


def _frame_range(
    num_frames: int,
    start_frame: int,
    num_display_frames: int,
) -> tuple[int, int]:
    """
    Compute a valid frame display range.

    Args:
        num_frames (int): Total number of frames.
        start_frame (int): First frame to display.
        num_display_frames (int): Number of frames to display.

    Returns:
        tuple[int, int]: Start and end frame indices.
    """
    start_frame = max(0, start_frame)
    end_frame = min(num_frames, start_frame + num_display_frames)

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid frame range: start_frame={start_frame}, "
            f"num_display_frames={num_display_frames}"
        )

    return start_frame, end_frame


def _plot_sample_labels(
    ax: plt.Axes,
    labels_slice: np.ndarray,
    sample_start: int,
    sample_end: int,
) -> None:
    """
    Plot sample-level labels over a sample index range.

    Args:
        ax (plt.Axes): Target axes.
        labels_slice (np.ndarray): Sample labels to plot.
        sample_start (int): Start sample index.
        sample_end (int): End sample index.
    """
    x = np.arange(sample_start, sample_end)
    ax.step(x, labels_slice, where="post")
    ax.fill_between(x, labels_slice, step="post", alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Sample\nlabel")
    ax.set_xlabel("Sample index")
    ax.grid(True, alpha=0.3)


def _plot_frame_windows(
    ax: plt.Axes,
    frame_starts: np.ndarray,
    frame_ends: np.ndarray,
    frame_labels: np.ndarray,
    start_frame: int,
    end_frame: int,
    show_frame_text: bool = True,
) -> None:
    """ "
    Plot frame windows in sample coordinates.

    Args:
        ax (plt.Axes): Target axes.
        frame_starts (np.ndarray): Frame start indices.
        frame_ends (np.ndarray): Frame end indices.
        frame_labels (np.ndarray): Frame-level labels.
        start_frame (int): First frame to display.
        end_frame (int): End frame index.
        show_frame_text (bool): Whether to annotate each frame.
    """
    for row_idx, frame_idx in enumerate(range(start_frame, end_frame)):
        start = frame_starts[frame_idx]
        end = frame_ends[frame_idx]
        label = frame_labels[frame_idx]

        color = "tab:red" if label == 1 else "tab:blue"

        ax.hlines(y=row_idx, xmin=start, xmax=end, linewidth=3, color=color)
        ax.plot([start, end], [row_idx, row_idx], marker="|", linestyle="None", color=color)

        if show_frame_text:
            ax.text(
                end,
                row_idx,
                f"  f{frame_idx}:{label}",
                va="center",
                fontsize=8,
            )

    ax.set_ylabel("Frame")
    ax.set_xlabel("Sample index")
    ax.set_yticks(np.arange(end_frame - start_frame))
    ax.set_yticklabels([str(i) for i in range(start_frame, end_frame)])
    ax.grid(True, alpha=0.3)


def _plot_frame_label_overlay_panel(
    ax: plt.Axes,
    recomputed_labels: np.ndarray,
    aligner_labels: np.ndarray | None,
    start_frame: int,
    end_frame: int,
) -> None:
    """
    Plot frame-level labels for visual comparison.

    Args:
        ax (plt.Axes): Target axes.
        recomputed_labels (np.ndarray): Recomputed frame labels.
        aligner_labels (np.ndarray | None): Optional aligner output labels.
        start_frame (int): First frame to display.
        end_frame (int): End frame index.
    """
    rec = recomputed_labels[start_frame:end_frame]

    if aligner_labels is None:
        image = rec[np.newaxis, :]
        ax.imshow(
            image,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )
        ax.set_yticks([0])
        ax.set_yticklabels(["Recomputed"])
    else:
        ali = aligner_labels[start_frame:end_frame]
        image = np.vstack([ali, rec])
        ax.imshow(
            image,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Aligner", "Recomputed"])

        mismatches = np.where(ali != rec)[0]
        for idx in mismatches:
            ax.axvline(idx - 0.5, linestyle="--", linewidth=1)
            ax.text(
                idx,
                1.5,
                "x",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame\nlabels")
    ax.set_xticks(np.arange(end_frame - start_frame))
    ax.set_xticklabels([str(i) for i in range(start_frame, end_frame)], rotation=90)
    ax.grid(False)


def _print_alignment_summary(
    sample_labels: Tensor,
    num_frames: int,
    hop_length: int,
    frame_length: int,
    center: bool,
    start_frame: int,
    num_display_frames: int,
    recomputed_labels: np.ndarray,
    aligner_labels: np.ndarray | None,
) -> None:
    """
    Print summary statistics for alignment debugging.

    Args:
        sample_labels (Tensor): Sample-level labels.
        num_frames (int): Total number of frames.
        hop_length (int): Hop length in samples.
        frame_length (int): Frame length in samples.
        center (bool): Whether frames are centered.
        start_frame (int): First displayed frame.
        num_display_frames (int): Number of displayed frames.
        recomputed_labels (np.ndarray): Recomputed frame labels.
        aligner_labels (np.ndarray | None): Optional aligner output labels.
    """
    labels_np = binarize_labels(to_numpy_1d(sample_labels))

    print("Alignment debug info")
    print(f"sample labels shape: {tuple(sample_labels.shape)}")
    print(f"num_frames: {num_frames}")
    print(f"hop_length: {hop_length}")
    print(f"frame_length: {frame_length}")
    print(f"center: {center}")
    print(f"start_frame: {start_frame}")
    print(f"num_display_frames: {num_display_frames}")
    print(f"sample-level positive ratio: {labels_np.mean():.4f}")
    print(f"recomputed frame-level positive ratio: {recomputed_labels.mean():.4f}")

    if aligner_labels is not None:
        mismatch_count = int((aligner_labels != recomputed_labels).sum())
        print(f"aligner frame-level positive ratio: {aligner_labels.mean():.4f}")
        print(f"mismatched frames: {mismatch_count} / {num_frames}")


def plot_alignment_debug(
    sample_labels: Tensor,
    num_frames: int,
    hop_length: int,
    frame_length: int,
    center: bool = True,
    start_frame: int = 0,
    num_display_frames: int = 40,
    aligner_output: Tensor | None = None,
    title: str = "Label Alignment Debug View",
    figsize: tuple[int, int] = (14, 8),
    show: bool = True,
    show_frame_text: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Visualize the mapping from sample-level to frame-level labels.

    Args:
        sample_labels (Tensor): Sample-level labels [num_samples].
        num_frames (int): Total number of frames.
        hop_length (int): Hop length in samples.
        frame_length (int): Frame length in samples.
        center (bool): Whether frames are centered.
        start_frame (int): First frame to display.
        num_display_frames (int): Number of frames to display.
        aligner_output (Tensor | None): Optional aligner output for comparison.
        title (str): Figure title.
        figsize (tuple[int, int]): Figure size.
        show (bool): Whether to display the plot.
        show_frame_text (bool): Whether to annotate frame windows.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
            Figure and axes for samples, windows, and frame labels.
    """
    validate_1d_tensor(sample_labels, "sample_labels")

    labels_np = binarize_labels(to_numpy_1d(sample_labels))
    aligner_labels = _validate_aligner_output(aligner_output, num_frames)

    recomputed_labels, frame_starts, frame_ends = compute_frame_labels_from_samples(
        sample_labels=sample_labels,
        num_frames=num_frames,
        hop_length=hop_length,
        frame_length=frame_length,
        center=center,
    )

    start_frame, end_frame = _frame_range(
        num_frames=num_frames,
        start_frame=start_frame,
        num_display_frames=num_display_frames,
    )

    sample_start = int(frame_starts[start_frame:end_frame].min())
    sample_end = int(frame_ends[start_frame:end_frame].max())
    labels_slice = labels_np[sample_start:sample_end]

    fig, (ax_samples, ax_windows, ax_frames) = plt.subplots(
        3,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.5, 3.0, 1.2]},
        sharex=False,
    )

    _plot_sample_labels(
        ax=ax_samples,
        labels_slice=labels_slice,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    ax_samples.set_title(title)

    _plot_frame_windows(
        ax=ax_windows,
        frame_starts=frame_starts,
        frame_ends=frame_ends,
        frame_labels=recomputed_labels,
        start_frame=start_frame,
        end_frame=end_frame,
        show_frame_text=show_frame_text,
    )
    ax_windows.set_xlim(sample_start, sample_end)

    _plot_frame_label_overlay_panel(
        ax=ax_frames,
        recomputed_labels=recomputed_labels,
        aligner_labels=aligner_labels,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, (ax_samples, ax_windows, ax_frames)


def debug_plot_alignment(
    sample_labels: Tensor,
    num_frames: int,
    hop_length: int,
    frame_length: int,
    center: bool = True,
    start_frame: int = 0,
    num_display_frames: int = 40,
    aligner_output: Tensor | None = None,
    use_seaborn: bool = True,
    title: str = "Label Alignment Debug View",
    figsize: tuple[int, int] = (14, 8),
    show_frame_text: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Print alignment diagnostics and display the debug plot.

    Args:
        sample_labels (Tensor): Sample-level labels.
        num_frames (int): Total number of frames.
        hop_length (int): Hop length in samples.
        frame_length (int): Frame length in samples.
        center (bool): Whether frames are centered.
        start_frame (int): First frame to display.
        num_display_frames (int): Number of frames to display.
        aligner_output (Tensor | None): Optional aligner output for comparison.
        use_seaborn (bool): Whether to apply Seaborn-style plotting.
        title (str): Figure title.
        figsize (tuple[int, int]): Figure size.
        show_frame_text (bool): Whether to annotate frame windows.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
            Figure and axes for the debug visualization.
    """
    set_plot_style(use_seaborn=use_seaborn)

    recomputed_labels, _, _ = compute_frame_labels_from_samples(
        sample_labels=sample_labels,
        num_frames=num_frames,
        hop_length=hop_length,
        frame_length=frame_length,
        center=center,
    )
    aligner_labels = _validate_aligner_output(aligner_output, num_frames)

    _print_alignment_summary(
        sample_labels=sample_labels,
        num_frames=num_frames,
        hop_length=hop_length,
        frame_length=frame_length,
        center=center,
        start_frame=start_frame,
        num_display_frames=num_display_frames,
        recomputed_labels=recomputed_labels,
        aligner_labels=aligner_labels,
    )

    return plot_alignment_debug(
        sample_labels=sample_labels,
        num_frames=num_frames,
        hop_length=hop_length,
        frame_length=frame_length,
        center=center,
        start_frame=start_frame,
        num_display_frames=num_display_frames,
        aligner_output=aligner_output,
        title=title,
        figsize=figsize,
        show=True,
        show_frame_text=show_frame_text,
    )
