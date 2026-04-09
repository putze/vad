from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from vad.data.audio_utils import ensure_mono_waveform
from vad.data.file_utils import load_audio
from vad.data.preprocessing import LogMelFeatureExtractor, WaveformPreprocessor
from vad.inference.utils import (
    ensure_time_major_features,
    logits_to_predictions,
    normalize_binary_logits,
    prepare_conv1d_input,
)
from vad.models.loading import load_model


@dataclass(slots=True)
class OfflineVADPrediction:
    """
    Container for frame-level offline VAD outputs.

    Attributes:
        waveform: Original input waveform provided for inference.
        sample_rate: Sample rate of ``waveform`` in Hz.
        frame_times: Time stamp for each output frame in seconds.
        probabilities: Frame-level speech probabilities.
        predictions: Frame-level binary speech decisions.
        duration_seconds: Total waveform duration in seconds.
    """

    waveform: Tensor
    sample_rate: int
    frame_times: Tensor
    probabilities: Tensor
    predictions: Tensor
    duration_seconds: float


class OfflineVADInferencer:
    """
    Run offline VAD inference on complete audio inputs.

    This class encapsulates the full inference pipeline:
        waveform -> preprocessing -> feature extraction -> model -> predictions
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        threshold: float = 0.5,
        target_sample_rate: int = 16000,
        n_mels: int = 40,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
    ) -> None:
        """
        Initialize the offline inference pipeline.

        Args:
            checkpoint_path: Path to the trained model checkpoint.
            device: Torch device used for inference.
            threshold: Probability threshold for speech detection.
            target_sample_rate: Target sample rate used by the preprocessing pipeline.
            n_mels: Number of mel-frequency bins.
            frame_length_ms: Analysis window length in milliseconds.
            frame_shift_ms: Frame hop in milliseconds.

        Raises:
            ValueError: If any numeric parameter is invalid.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if target_sample_rate <= 0:
            raise ValueError(f"target_sample_rate must be positive, got {target_sample_rate}")
        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")
        if frame_length_ms <= 0:
            raise ValueError(f"frame_length_ms must be positive, got {frame_length_ms}")
        if frame_shift_ms <= 0:
            raise ValueError(f"frame_shift_ms must be positive, got {frame_shift_ms}")

        self.device = device
        self.threshold = threshold
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms

        self.waveform_preprocessor = WaveformPreprocessor(
            target_sample_rate=target_sample_rate,
        )

        frame_length = round(target_sample_rate * frame_length_ms / 1000.0)
        hop_length = round(target_sample_rate * frame_shift_ms / 1000.0)

        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=target_sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            n_fft=frame_length,
            n_mels=n_mels,
        )

        self.model = load_model(checkpoint_path, self.device, self.n_mels)

    def _prepare_features(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """
        Convert a waveform into model-ready Conv1d input.

        The waveform is converted to mono if needed, resampled to the target
        sample rate, transformed into log-mel features, normalized to time-major
        layout, and finally reshaped to Conv1d format.

        Args:
            waveform: Input waveform.
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            Tensor of shape ``[1, n_mels, T]`` ready for the model.
        """
        waveform = ensure_mono_waveform(waveform)

        waveform, sample_rate = self.waveform_preprocessor.process_waveform(
            waveform,
            sample_rate,
        )

        features = self.feature_extractor(waveform)
        features = ensure_time_major_features(features, feature_dim=self.n_mels)

        return prepare_conv1d_input(features, self.device)

    @torch.inference_mode()
    def predict_waveform(self, waveform: Tensor, sample_rate: int) -> OfflineVADPrediction:
        """
        Run frame-level VAD on a waveform.

        Args:
            waveform: Input audio waveform.
            sample_rate: Sample rate of the input waveform in Hz.

        Returns:
            An ``OfflineVADPrediction`` containing frame-level probabilities,
            hard predictions, frame times, and basic waveform metadata.
        """
        duration_seconds = waveform.shape[-1] / sample_rate
        features = self._prepare_features(waveform, sample_rate)

        logits = self.model(features)
        logits = normalize_binary_logits(logits)
        probabilities, predictions = logits_to_predictions(logits, self.threshold)

        num_frames = probabilities.shape[0]
        frame_times = torch.arange(num_frames, dtype=torch.float32) * (self.frame_shift_ms / 1000.0)

        return OfflineVADPrediction(
            waveform=waveform,
            sample_rate=sample_rate,
            probabilities=probabilities,
            predictions=predictions,
            frame_times=frame_times,
            duration_seconds=duration_seconds,
        )

    @torch.inference_mode()
    def predict_file(self, audio_path: str | Path) -> OfflineVADPrediction:
        """
        Run frame-level VAD on an audio file.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            Offline VAD prediction for the file.
        """
        waveform, sample_rate = load_audio(str(audio_path))
        return self.predict_waveform(waveform, sample_rate)
