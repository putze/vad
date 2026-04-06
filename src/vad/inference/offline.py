from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch import Tensor, nn

from src.vad.data.preprocessing import AudioPreprocessor, LogMelFeatureExtractor
from src.vad.inference.utils import (
    ensure_mono_waveform,
    ensure_time_major_features,
    logits_to_predictions,
    normalize_binary_logits,
    prepare_conv1d_input,
)
from src.vad.models import CausalVAD


@dataclass(slots=True)
class OfflineVADPrediction:
    """
    Container for frame-level offline VAD outputs.

    Attributes:
        waveform: Input waveform used for inference.
        sample_rate: Waveform sample rate in Hz.
        frame_times: Time stamp for each output frame in seconds.
        probabilities: Speech probabilities for each frame.
        predictions: Binary speech decisions for each frame.
        duration_seconds: Total waveform duration in seconds.
    """

    waveform: torch.Tensor
    sample_rate: int
    frame_times: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    duration_seconds: float


class OfflineVADInferencer:
    """Run offline inference with a trained model."""

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
            checkpoint_path: Path to the trained checkpoint.
            device: Torch device used for inference.
            threshold: Decision threshold applied to speech probabilities.
            target_sample_rate: Target sample rate in Hz.
            n_mels: Number of mel-frequency bins.
            frame_length_ms: Analysis window length in milliseconds.
            frame_shift_ms: Frame hop in milliseconds.
        """
        self.device = device
        self.threshold = threshold
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms

        self.audio_preprocessor = AudioPreprocessor(
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

        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str | Path) -> nn.Module:
        """
        Load a trained model checkpoint.

        Supports a raw state dict or a checkpoint containing
        ``model_state_dict``.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Loaded model in evaluation mode.
        """
        model = CausalVAD(n_mels=self.n_mels)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _prepare_features(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """
        Convert a waveform into model-ready input features.

        Args:
            waveform: Input waveform.
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            Input tensor of shape ``[1, n_mels, T]``.
        """
        waveform = ensure_mono_waveform(waveform)

        waveform, sample_rate = self.audio_preprocessor.process_waveform(
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
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            Offline VAD prediction for the waveform.
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
        """Run frame-level VAD on an audio file.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            Offline VAD prediction for the file.
        """
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return self.predict_waveform(waveform, sample_rate)
