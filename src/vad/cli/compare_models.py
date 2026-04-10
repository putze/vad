from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from vad.baseline.webrtc import WebRTCVADBaseline
from vad.config import InferenceConfig
from vad.data import DatasetConfig, build_raw_dataset
from vad.evaluate.evaluate import evaluate_model
from vad.inference import OfflineVADInferencer


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
        default="dev-clean",
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
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cpu', 'cuda', or 'mps'. Defaults to auto-detect.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    """Resolve torch device from CLI or auto-detect."""
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = build_raw_dataset(
        "librivad",
        DatasetConfig(
            results_root=args.results_root,
            labels_root=args.labels_root,
            datasets=("LibriSpeech",),
            splits=(args.split,),
        ),
    )

    inferencer = OfflineVADInferencer(
        checkpoint_path=args.checkpoint,
        device=device,
        inference_config=InferenceConfig(threshold=args.threshold),
    )
    baseline = WebRTCVADBaseline(aggressiveness=args.webrtc_mode)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split     : {args.split}")
    print(f"Threshold : {args.threshold}")
    print(f"Baseline  : {baseline.name} (mode={args.webrtc_mode})")
    print()

    model_metrics = evaluate_model(dataset, inferencer)
    baseline_metrics = evaluate_model(dataset, baseline)

    print_comparison(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        baseline_name=f"webrtc ({args.webrtc_mode})",
    )


if __name__ == "__main__":
    main()
