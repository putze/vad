from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from vad.config import InferenceConfig
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
        inference_config: InferenceConfig | None = None,
    ) -> None:
        """
        Initialize the offline inference pipeline.

        Args:
            checkpoint_path: Path to the trained model checkpoint.
            device: Torch device used for inference.
            inference_config: Optional inference-time configuration.

        Raises:
            ValueError: If the threshold is invalid.
        """
        self.device = device
        self.inference_config = inference_config or InferenceConfig()

        if not 0.0 <= self.inference_config.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.inference_config.threshold}")

        self.model, self.audio_config, _ = load_model(checkpoint_path, self.device)

        self.waveform_preprocessor = WaveformPreprocessor(
            target_sample_rate=self.audio_config.sample_rate,
        )

        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=self.audio_config.sample_rate,
            frame_length=self.audio_config.frame_length_samples,
            hop_length=self.audio_config.hop_length_samples,
            n_fft=self.audio_config.frame_length_samples,
            n_mels=self.audio_config.n_mels,
            center=False,
        )

    @property
    def threshold(self) -> float:
        """Return the speech detection threshold."""
        return self.inference_config.threshold

    @property
    def frame_shift_ms(self) -> float:
        """Return the frame shift in milliseconds."""
        return self.audio_config.frame_shift_ms

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
        features = ensure_time_major_features(
            features,
            feature_dim=self.audio_config.n_mels,
        )

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
        frame_times = torch.arange(num_frames, dtype=torch.float32) * (
            self.audio_config.frame_shift_ms / 1000.0
        )

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
