from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from src.vad.visualization.helpers import (
    shade_positive_frames,
    to_numpy_1d,
    to_numpy_2d,
    validate_frame_alignment,
)
from src.vad.visualization.style import set_plot_style


def print_feature_debug_info(
    features: Tensor,
    frame_labels: Tensor,
    sample_name: str | None = None,
) -> None:
    """
    Print summary information for features and aligned frame labels.

    Args:
        features (Tensor): Feature tensor [num_features, num_frames].
        frame_labels (Tensor): Frame-level labels [num_frames].
        sample_name (str | None): Optional sample identifier.
    """
    validate_frame_alignment(
        sequence=features,
        frame_labels=frame_labels,
        time_dim=1,
        sequence_name="features",
    )

    header = "Feature debug info"
    if sample_name is not None:
        header += f" - {sample_name}"

    unique_labels = np.unique(to_numpy_1d(frame_labels))

    print(header)
    print(f"features shape: {tuple(features.shape)}")
    print(f"frame labels shape: {tuple(frame_labels.shape)}")
    print(f"num feature bins: {features.shape[0]}")
    print(f"num frames: {features.shape[1]}")
    print(f"feature min/max: {features.min().item():.5f} / {features.max().item():.5f}")
    print(f"label unique values: {unique_labels}")
    print(f"positive frames: {int((frame_labels > 0).sum().item())}")
    print(f"negative frames: {int((frame_labels <= 0).sum().item())}")


def plot_features_with_labels(
    features: Tensor,
    frame_labels: Tensor,
    title: str = "Features with Frame-Level Labels",
    figsize: tuple[int, int] = (14, 5),
    show: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot features and aligned frame labels in separate panels.

    Args:
        features (Tensor): Feature tensor [num_features, num_frames].
        frame_labels (Tensor): Frame-level labels [num_frames].
        title (str): Figure title.
        figsize (tuple[int, int]): Figure size.
        show (bool): Whether to display the figure.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]: Figure and axes.
    """
    validate_frame_alignment(
        sequence=features,
        frame_labels=frame_labels,
        time_dim=1,
        sequence_name="features",
    )

    feature_np = to_numpy_2d(features).astype(np.float32)
    labels_np = (to_numpy_1d(frame_labels) > 0).astype(np.float32)

    fig, (ax_feat, ax_lab) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    im = ax_feat.imshow(
        feature_np,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax_feat.set_ylabel("Feature bin")
    ax_feat.set_title(title)
    fig.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)

    x = np.arange(len(labels_np))
    ax_lab.step(x, labels_np, where="post")
    ax_lab.fill_between(x, labels_np, step="post", alpha=0.3)
    ax_lab.set_xlabel("Frame")
    ax_lab.set_ylabel("Label")
    ax_lab.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, (ax_feat, ax_lab)


def plot_features_with_label_overlay(
    features: Tensor,
    frame_labels: Tensor,
    title: str = "Features with Frame-Level Label Overlay",
    figsize: tuple[int, int] = (14, 5),
    alpha: float = 0.25,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot features with positive frames highlighted on the same axis.

    Args:
        features (Tensor): Feature tensor [num_features, num_frames].
        frame_labels (Tensor): Frame-level labels [num_frames].
        title (str): Figure title.
        figsize (tuple[int, int]): Figure size.
        alpha (float): Transparency of highlighted regions.
        show (bool): Whether to display the figure.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
    """
    validate_frame_alignment(
        sequence=features,
        frame_labels=frame_labels,
        time_dim=1,
        sequence_name="features",
    )

    feature_np = to_numpy_2d(features).astype(np.float32)
    labels_np = (to_numpy_1d(frame_labels) > 0).astype(np.int32)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        feature_np,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="YlOrRd",
    )

    shade_positive_frames(ax=ax, labels=labels_np, alpha=alpha)

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Feature bin")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def debug_plot_features_with_labels(
    features: Tensor,
    frame_labels: Tensor,
    sample_name: str | None = None,
    use_seaborn: bool = True,
    overlay: bool = False,
) -> tuple[plt.Figure, object]:
    """
    Print debug info and plot features with aligned labels.

    Args:
        features (Tensor): Feature tensor [num_features, num_frames].
        frame_labels (Tensor): Frame-level labels [num_frames].
        sample_name (str | None): Optional sample identifier.
        use_seaborn (bool): Whether to apply Seaborn-style plotting.
        overlay (bool): Whether to use the overlay plot.

    Returns:
        tuple[plt.Figure, object]: Figure and axes.
    """
    set_plot_style(use_seaborn=use_seaborn)
    print_feature_debug_info(
        features=features,
        frame_labels=frame_labels,
        sample_name=sample_name,
    )

    if overlay:
        title = "Features with Frame-Level Label Overlay"
        if sample_name is not None:
            title += f" - {sample_name}"

        return plot_features_with_label_overlay(
            features=features,
            frame_labels=frame_labels,
            title=title,
            show=True,
        )

    title = "Features with Frame-Level Labels"
    if sample_name is not None:
        title += f" - {sample_name}"

    return plot_features_with_labels(
        features=features,
        frame_labels=frame_labels,
        title=title,
        show=True,
    )


def debug_plot_features_with_label_overlay(
    features: Tensor,
    frame_labels: Tensor,
    sample_name: str | None = None,
    use_seaborn: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Print debug info and plot the compact overlay view.

    Args:
        features (Tensor): Feature tensor [num_features, num_frames].
        frame_labels (Tensor): Frame-level labels [num_frames].
        sample_name (str | None): Optional sample identifier.
        use_seaborn (bool): Whether to apply Seaborn-style plotting.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
    """
    set_plot_style(use_seaborn=use_seaborn)
    print_feature_debug_info(
        features=features,
        frame_labels=frame_labels,
        sample_name=sample_name,
    )

    title = "Features with Frame-Level Label Overlay"
    if sample_name is not None:
        title += f" - {sample_name}"

    return plot_features_with_label_overlay(
        features=features,
        frame_labels=frame_labels,
        title=title,
        show=True,
    )
