from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from vad.baseline.webrtc import WebRTCVADBaseline
from vad.cli.utils import resolve_device
from vad.config import InferenceConfig
from vad.data import DatasetConfig, build_raw_dataset
from vad.evaluate.evaluate import evaluate_model
from vad.inference.offline import OfflineVADInferencer
from vad.visualization.evaluation import compute_roc_curve, plot_roc_curve


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare trained VAD model against WebRTC VAD.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/antje/Blynt/LibriVAD/Results"),
        help="Root directory containing evaluation audio files.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("/Users/antje/Blynt/LibriVAD/Files/Labels"),
        help="Root directory containing evaluation labels.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test-clean",
        help="Dataset split used for evaluation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Speech detection threshold for the trained model.",
    )
    parser.add_argument(
        "--webrtc-mode",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="WebRTC aggressiveness mode.",
    )
    parser.add_argument(
        "--all-webrtc",
        action="store_true",
        help="Plot operating points for all WebRTC aggressiveness modes (0-3).",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=None,
        help="Optional path to save ROC curve figure, e.g. outputs/roc.png",
    )
    parser.add_argument(
        "--show-roc",
        action="store_true",
        help="Display ROC curve interactively.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cpu', 'cuda', or 'mps'. Defaults to auto-detect.",
    )
    return parser.parse_args()


def metric_value(metrics: Any, name: str) -> float:
    """Read a metric from either a dict or a metrics object."""
    if isinstance(metrics, dict):
        return float(metrics.get(name, float("nan")))
    return float(getattr(metrics, name, float("nan")))


def print_comparison(
    model_metrics: Any,
    baseline_metrics: Any,
    baseline_name: str,
) -> None:
    """Print a side-by-side metric table."""
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
    ]

    left_width = 22
    col_width = 14

    header = f"{'metric':<{left_width}}{'my model':>{col_width}}{baseline_name:>{col_width}}"
    print(header)
    print("-" * len(header))

    for name in metric_names:
        model_value = metric_value(model_metrics, name)
        baseline_value = metric_value(baseline_metrics, name)
        print(f"{name:<{left_width}}{model_value:>{col_width}.4f}{baseline_value:>{col_width}.4f}")


def print_webrtc_points_table(points: dict[str, tuple[float, float]]) -> None:
    """Print WebRTC operating points as FPR/TPR table."""
    left_width = 16
    col_width = 14

    header = f"{'baseline':<{left_width}}{'fpr':>{col_width}}{'tpr':>{col_width}}"
    print(header)
    print("-" * len(header))

    for name, (fpr, tpr) in points.items():
        print(f"{name:<{left_width}}{fpr:>{col_width}.4f}{tpr:>{col_width}.4f}")


def make_dataset(
    results_root: Path,
    labels_root: Path,
    split: str,
):
    """
    Build a fresh evaluation dataset.

    A fresh dataset is created each time because some dataset/iterator
    implementations may be exhausted after one full pass.
    """
    return build_raw_dataset(
        "librivad",
        DatasetConfig(
            results_root=results_root,
            labels_root=labels_root,
            datasets=("LibriSpeech",),
            splits=(split,),
        ),
    )


def evaluate_webrtc_operating_points(
    results_root: Path,
    labels_root: Path,
    split: str,
    all_webrtc: bool,
    selected_mode: int,
) -> tuple[dict[str, tuple[float, float]], Any]:
    """
    Evaluate WebRTC as ROC operating points.

    Returns:
        points: mapping name -> (fpr, tpr)
        selected_metrics: metrics object for the selected mode
    """
    modes = [0, 1, 2, 3] if all_webrtc else [selected_mode]
    points: dict[str, tuple[float, float]] = {}
    selected_metrics = None

    for mode in modes:
        dataset = make_dataset(
            results_root=results_root,
            labels_root=labels_root,
            split=split,
        )
        baseline = WebRTCVADBaseline(aggressiveness=mode)
        result = evaluate_model(dataset, baseline)
        metrics = result.metrics

        points[f"webrtc ({mode})"] = (
            float(metrics.false_positive_rate),
            float(metrics.recall),
        )

        if mode == selected_mode:
            selected_metrics = metrics

    if selected_metrics is None:
        # should never happen, but keep it safe
        dataset = make_dataset(
            results_root=results_root,
            labels_root=labels_root,
            split=split,
        )
        baseline = WebRTCVADBaseline(aggressiveness=selected_mode)
        selected_metrics = evaluate_model(dataset, baseline).metrics

    return points, selected_metrics


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model_dataset = make_dataset(
        results_root=args.results_root,
        labels_root=args.labels_root,
        split=args.split,
    )

    inferencer = OfflineVADInferencer(
        checkpoint_path=args.checkpoint,
        device=device,
        inference_config=InferenceConfig(threshold=args.threshold),
    )

    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Split        : {args.split}")
    print(f"Threshold    : {args.threshold}")
    print(f"WebRTC mode  : {args.webrtc_mode}")
    print(f"All WebRTC   : {args.all_webrtc}")
    print()

    # Evaluate trained model once and keep scores for ROC/AUC
    model_result = evaluate_model(model_dataset, inferencer)
    model_metrics = model_result.metrics

    # Evaluate WebRTC operating points
    webrtc_points, baseline_metrics = evaluate_webrtc_operating_points(
        results_root=args.results_root,
        labels_root=args.labels_root,
        split=args.split,
        all_webrtc=args.all_webrtc,
        selected_mode=args.webrtc_mode,
    )

    # Print fixed-threshold comparison against selected WebRTC mode
    print_comparison(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        baseline_name=f"webrtc ({args.webrtc_mode})",
    )
    print()

    if model_result.scores is None:
        raise ValueError(
            "The evaluated model did not return probabilities/scores, "
            "so ROC/AUC cannot be computed."
        )

    # Compute ROC/AUC for trained model
    roc_result = compute_roc_curve(
        targets=model_result.targets,
        scores=model_result.scores,
    )

    print(f"Model ROC AUC: {roc_result.auc:.4f}")
    print()

    print_webrtc_points_table(webrtc_points)

    # Plot ROC curve + WebRTC points
    fig = plot_roc_curve(
        roc_result=roc_result,
        operating_points=webrtc_points,
        title=f"ROC Curve - {args.split}",
    )

    if args.roc_output is not None:
        args.roc_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.roc_output, dpi=200, bbox_inches="tight")
        print()
        print(f"Saved ROC plot to: {args.roc_output}")

    if args.show_roc:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
