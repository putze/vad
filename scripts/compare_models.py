from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.vad.baseline.webrtc import WebRTCVADBaseline
from src.vad.data import DatasetConfig, build_raw_dataset
from src.vad.evaluate.evaluate import evaluate_model
from src.vad.inference import OfflineVADInferencer


def print_comparison(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
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
        model_value = model_metrics.get(name, float("nan"))
        baseline_value = baseline_metrics.get(name, float("nan"))
        print(f"{name:<{left_width}}{model_value:>{col_width}.4f}{baseline_value:>{col_width}.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained VAD model against WebRTC VAD.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint.",
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
        default="cpu",
        help="Torch device.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    results_root = Path("/Users/antje/Blynt/LibriVAD/Results")
    labels_root = Path("/Users/antje/Blynt/LibriVAD/Files/Labels")
    dataset = build_raw_dataset(
        "librivad",
        DatasetConfig(
            results_root=results_root,
            labels_root=labels_root,
            datasets=("LibriSpeech",),
            splits=("dev-clean",),
        ),
    )

    inferencer = OfflineVADInferencer(args.checkpoint, device=device)
    baseline = WebRTCVADBaseline(aggressiveness=args.webrtc_mode)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Baseline: {baseline.name} (mode={args.webrtc_mode})")
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
