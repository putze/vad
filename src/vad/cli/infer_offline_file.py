from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from vad.cli.utils import resolve_device
from vad.config import InferenceConfig
from vad.inference.offline import OfflineVADInferencer, OfflineVADPrediction
from vad.inference.utils import predictions_to_segments


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run offline VAD inference on an audio file.")
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to input audio file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_causal_vad.pt"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Speech detection threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot interactively.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cpu', 'cuda', or 'mps'. Defaults to auto-detect.",
    )
    return parser.parse_args()


def save_csv(
    output_path: Path,
    frame_times: torch.Tensor,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """
    Save frame-level predictions to a CSV file.

    Args:
        output_path: Destination CSV path.
        frame_times: Frame time stamps in seconds.
        probabilities: Speech probabilities for each frame.
        predictions: Binary speech decisions for each frame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_sec", "probability", "prediction"])
        for t, p, y in zip(
            frame_times.tolist(),
            probabilities.tolist(),
            predictions.tolist(),
        ):
            writer.writerow([t, p, y])


def save_prediction_plot(
    output_path: Path,
    prediction: OfflineVADPrediction,
    sample_rate: int,
    threshold: float,
    audio_name: str,
    show: bool = False,
) -> None:
    """
    Save an offline VAD prediction plot.

    Args:
        output_path: Destination image path.
        prediction: Offline prediction object with waveform and frame outputs.
        sample_rate: Waveform sample rate in Hz.
        threshold: Speech detection threshold.
        audio_name: Audio file name used in the plot title.
        show: Whether to display the plot interactively.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from vad.visualization.inference import plot_offline_vad_prediction

    frame_hop_s = (
        float(prediction.frame_times[1] - prediction.frame_times[0])
        if len(prediction.frame_times) > 1
        else 0.01
    )

    fig, _ = plot_offline_vad_prediction(
        waveform=prediction.waveform,
        sample_rate=sample_rate,
        probabilities=prediction.probabilities,
        predictions=prediction.predictions,
        frame_hop_s=frame_hop_s,
        threshold=threshold,
        title=f"Offline VAD - {audio_name}",
        show=False,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    """Run offline VAD inference from the command line."""
    args = parse_args()
    device = resolve_device(args.device)

    inferencer = OfflineVADInferencer(
        checkpoint_path=args.checkpoint,
        device=device,
        inference_config=InferenceConfig(threshold=args.threshold),
    )
    prediction = inferencer.predict_file(args.audio)

    segments = predictions_to_segments(
        prediction.predictions,
        frame_shift_ms=inferencer.frame_shift_ms,
    )

    stem = args.audio.stem
    csv_path = args.output_dir / f"{stem}.csv"
    plot_path = args.output_dir / f"{stem}.png"

    save_csv(
        csv_path,
        prediction.frame_times,
        prediction.probabilities,
        prediction.predictions,
    )

    save_prediction_plot(
        output_path=plot_path,
        prediction=prediction,
        sample_rate=prediction.sample_rate,
        threshold=inferencer.threshold,
        audio_name=args.audio.name,
        show=args.show_plot,
    )

    speech_ratio = prediction.predictions.float().mean().item()

    print(f"audio          : {args.audio}")
    print(f"checkpoint     : {args.checkpoint}")
    print(f"sample_rate    : {inferencer.audio_config.sample_rate}")
    print(f"frame_shift_ms : {inferencer.frame_shift_ms}")
    print(f"duration       : {prediction.duration_seconds:.2f}s")
    print(f"num_frames     : {len(prediction.predictions)}")
    print(f"speech_ratio   : {speech_ratio:.3f}")
    print(f"csv_saved_to   : {csv_path}")
    print(f"plot_saved_to  : {plot_path}")

    print("\ndetected speech segments:")
    if not segments:
        print("  none")
    else:
        for start, end in segments:
            print(f"  {start:.2f}s -> {end:.2f}s")


if __name__ == "__main__":
    main()
