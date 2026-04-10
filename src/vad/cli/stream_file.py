from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from vad.cli.utils import resolve_device
from vad.config import InferenceConfig, StreamingConfig
from vad.data.file_utils import load_audio
from vad.data.preprocessing.features import LogMelFeatureExtractor
from vad.data.preprocessing.waveform import WaveformPreprocessor
from vad.inference.adapters import StreamingFeatureExtractorAdapter
from vad.inference.streaming import StreamingVADInferencer
from vad.models.loading import load_model


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
        "--min-buffer-ms",
        type=float,
        default=25.0,
        help="Minimum buffered audio before inference, in milliseconds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cpu', 'cuda', or 'mps'. Defaults to auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    """Run simulated streaming VAD on an audio file."""
    args = parse_args()
    device = resolve_device(args.device)

    waveform, sample_rate = load_audio(str(args.audio))

    model, audio_config, _ = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    waveform_preprocessor = WaveformPreprocessor(
        target_sample_rate=audio_config.sample_rate,
    )
    feature_extractor = LogMelFeatureExtractor(
        sample_rate=audio_config.sample_rate,
        frame_length=audio_config.frame_length_samples,
        hop_length=audio_config.hop_length_samples,
        n_fft=audio_config.frame_length_samples,
        n_mels=audio_config.n_mels,
        center=False,
    )
    extractor = StreamingFeatureExtractorAdapter(
        waveform_preprocessor=waveform_preprocessor,
        feature_extractor=feature_extractor,
        n_mels=audio_config.n_mels,
    )

    streaming_config = StreamingConfig(
        chunk_seconds=args.chunk_ms / 1000.0,
        min_buffer_seconds=args.min_buffer_ms / 1000.0,
    )

    inferencer = StreamingVADInferencer(
        model=model,
        feature_extractor=extractor,
        audio_config=audio_config,
        inference_config=InferenceConfig(threshold=args.threshold),
        streaming_config=streaming_config,
        device=device,
    )

    chunk_samples = round(audio_config.sample_rate * streaming_config.chunk_seconds)

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
    print(f"Audio file            : {args.audio}")
    print(f"Checkpoint            : {args.checkpoint}")
    print(f"Model sample rate     : {audio_config.sample_rate}")
    print(f"Frame length (ms)     : {audio_config.frame_length_ms}")
    print(f"Frame shift (ms)      : {audio_config.frame_shift_ms}")
    print(f"Chunk size (ms)       : {args.chunk_ms}")
    print(f"Min buffer (ms)       : {args.min_buffer_ms}")
    print(f"Total emitted frames  : {probabilities.numel()}")
    print(f"Mean speech prob      : {probabilities.mean().item():.4f}")
    print(f"Speech frame ratio    : {predictions.float().mean().item():.4f}")


if __name__ == "__main__":
    main()
