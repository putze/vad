from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor
from vad.data.file_utils import load_audio
from vad.data.preprocessing import AudioPreprocessor, LogMelFeatureExtractor
from vad.data.utils import ensure_mono_waveform
from vad.inference.streaming import StreamingVADInferencer
from vad.inference.utils import ensure_time_major_features
from vad.models.loading import load_model


class StreamingFeatureExtractorAdapter:
    """Adapt preprocessing and feature extraction for streaming inference."""

    def __init__(
        self,
        audio_preprocessor: AudioPreprocessor,
        feature_extractor: LogMelFeatureExtractor,
        n_mels: int,
    ) -> None:
        """
        Initialize the streaming feature extractor adapter.

        Args:
            audio_preprocessor: Waveform preprocessor.
            feature_extractor: Frame-level feature extractor.
            n_mels: Expected feature dimension.
        """
        self.audio_preprocessor = audio_preprocessor
        self.feature_extractor = feature_extractor
        self.n_mels = n_mels

    def extract(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """
        Extract time-major features from a waveform.

        Args:
            waveform: Input waveform.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            Feature tensor of shape ``[T, F]``.
        """
        waveform = ensure_mono_waveform(waveform)
        waveform, sample_rate = self.audio_preprocessor.process_waveform(
            waveform,
            sample_rate,
        )
        features = self.feature_extractor(waveform)
        return ensure_time_major_features(features, feature_dim=self.n_mels)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run simulated streaming VAD on an audio file.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=100.0,
        help="Chunk size in milliseconds.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to speech probabilities.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=40,
        help="Number of mel bins.",
    )
    parser.add_argument(
        "--frame-length-ms",
        type=float,
        default=25.0,
        help="Frame length in milliseconds.",
    )
    parser.add_argument(
        "--frame-shift-ms",
        type=float,
        default=10.0,
        help="Frame hop in milliseconds.",
    )
    parser.add_argument(
        "--feature-context-frames",
        type=int,
        default=20,
        help="Number of past feature frames retained as left context.",
    )
    parser.add_argument(
        "--min-buffer-samples",
        type=int,
        default=400,
        help="Minimum buffered samples before feature extraction.",
    )
    return parser.parse_args()


def main() -> None:
    """Run simulated streaming VAD on an audio file."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waveform, sample_rate = load_audio(str(args.audio))

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        n_mels=args.n_mels,
    )

    frame_length = round(args.target_sample_rate * args.frame_length_ms / 1000.0)
    hop_length = round(args.target_sample_rate * args.frame_shift_ms / 1000.0)

    audio_preprocessor = AudioPreprocessor(
        target_sample_rate=args.target_sample_rate,
    )
    feature_extractor = LogMelFeatureExtractor(
        sample_rate=args.target_sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=frame_length,
        n_mels=args.n_mels,
    )
    extractor = StreamingFeatureExtractorAdapter(
        audio_preprocessor=audio_preprocessor,
        feature_extractor=feature_extractor,
        n_mels=args.n_mels,
    )

    inferencer = StreamingVADInferencer(
        model=model,
        feature_extractor=extractor,
        sample_rate=args.target_sample_rate,
        hop_length=hop_length,
        threshold=args.threshold,
        min_buffer_samples=args.min_buffer_samples,
        feature_context_frames=args.feature_context_frames,
        device=device,
    )

    chunk_samples = round(args.target_sample_rate * args.chunk_ms / 1000.0)

    all_probabilities: list[Tensor] = []
    all_predictions: list[Tensor] = []

    for start in range(0, waveform.numel(), chunk_samples):
        end = min(start + chunk_samples, waveform.numel())
        chunk = waveform[start:end]

        prediction = inferencer.process_chunk(chunk)
        if prediction is None:
            continue

        all_probabilities.append(prediction.probabilities)
        all_predictions.append(prediction.predictions)

        speech_ratio = float(prediction.predictions.float().mean().item())
        print(
            f"[{prediction.start_time_sec:7.2f}s - "
            f"{prediction.end_time_sec:7.2f}s] "
            f"frames={prediction.end_frame - prediction.start_frame:4d} "
            f"speech_ratio={speech_ratio:.3f}"
        )

    final_prediction = inferencer.flush()
    if final_prediction is not None:
        all_probabilities.append(final_prediction.probabilities)
        all_predictions.append(final_prediction.predictions)

    if not all_probabilities:
        print("No predictions were emitted.")
        return

    probabilities = torch.cat(all_probabilities)
    predictions = torch.cat(all_predictions)

    print()
    print(f"Total emitted frames: {probabilities.numel()}")
    print(f"Mean speech probability: {probabilities.mean().item():.4f}")
    print(f"Speech frame ratio: {predictions.float().mean().item():.4f}")


if __name__ == "__main__":
    main()
