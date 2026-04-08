from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from vad.inference.utils import (
    FeatureExtractorProtocol,
    logits_to_predictions,
    normalize_binary_logits,
    prepare_conv1d_input,
)


@dataclass(slots=True)
class StreamingPrediction:
    """
    Container for frame-level predictions emitted for one audio chunk.

    Attributes:
        probabilities: Speech probabilities for emitted frames.
        predictions: Binary speech decisions for emitted frames.
        start_frame: Index of the first emitted frame.
        end_frame: Index after the last emitted frame.
        start_time_sec: Start time of the emitted segment in seconds.
        end_time_sec: End time of the emitted segment in seconds.
    """

    probabilities: Tensor
    predictions: Tensor
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float


class StreamingVADInferencer:
    """Run stateful chunk-based VAD inference on streaming audio."""

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: FeatureExtractorProtocol,
        *,
        sample_rate: int = 16000,
        hop_length: int = 160,
        threshold: float = 0.5,
        min_buffer_samples: int = 400,
        feature_context_frames: int = 20,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the streaming inference pipeline.

        Args:
            model: Trained VAD model.
            feature_extractor: Feature extractor used on buffered audio.
            sample_rate: Expected sample rate in Hz.
            hop_length: Feature hop length in samples.
            threshold: Decision threshold applied to speech probabilities.
            min_buffer_samples: Minimum buffered samples before inference.
            feature_context_frames: Number of past feature frames kept as
                left context.
            device: Torch device used for inference.
        """
        self.model = model.eval()
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.threshold = threshold
        self.min_buffer_samples = min_buffer_samples
        self.feature_context_frames = feature_context_frames
        self.device = device or torch.device("cpu")

        self.model.to(self.device)
        self.reset()

    def reset(self) -> None:
        """Reset the internal streaming state."""
        self.sample_buffer = torch.empty(0, dtype=torch.float32)
        self.num_samples_seen = 0
        self.num_frames_emitted = 0
        self.left_context_features = torch.empty((0, 0), dtype=torch.float32)
        self.num_left_context_frames = 0

    @property
    def frame_hop_sec(self) -> float:
        """Return the duration of one frame hop in seconds."""
        return self.hop_length / self.sample_rate

    def _extract_features(self, waveform: Tensor) -> Tensor:
        """
        Extract frame-level features from a waveform.

        Args:
            waveform: Mono waveform of shape [N].

        Returns:
            Feature tensor of shape [T, F].
        """
        features = self.feature_extractor.extract(waveform, self.sample_rate)
        if features.ndim != 2:
            raise ValueError(
                f"Expected feature tensor with shape [T, F], got {tuple(features.shape)}."
            )

        return features

    @torch.inference_mode()
    def process_chunk(self, chunk: Tensor) -> StreamingPrediction | None:
        """
        Process one incoming audio chunk.

        Args:
            chunk: Audio chunk of shape ``[N]`` or ``[1, N]``.

        Returns:
            Prediction for newly emitted frames, or ``None`` if there is not
            enough audio to emit output.

        Raises:
            ValueError: If the chunk shape is invalid or the model output
                length does not match the feature length.
        """
        if chunk.ndim == 2:
            if chunk.shape[0] != 1:
                raise ValueError(
                    f"Expected mono audio chunk with shape [1, N], got {tuple(chunk.shape)}."
                )
            chunk = chunk.squeeze(0)

        if chunk.ndim != 1:
            raise ValueError(
                f"Expected audio chunk with shape [N] or [1, N], got {tuple(chunk.shape)}."
            )

        chunk = chunk.detach().cpu().float()
        self.sample_buffer = torch.cat([self.sample_buffer, chunk], dim=0)
        self.num_samples_seen += int(chunk.numel())

        if self.sample_buffer.numel() < self.min_buffer_samples:
            return None

        current_features = self._extract_features(self.sample_buffer)
        num_current_frames = int(current_features.shape[0])

        if num_current_frames == 0:
            return None

        num_new_frames = num_current_frames - self.num_frames_emitted
        if num_new_frames <= 0:
            return None

        if self.num_left_context_frames > 0:
            model_features = torch.cat(
                [self.left_context_features, current_features],
                dim=0,
            )
        else:
            model_features = current_features

        model_input = prepare_conv1d_input(model_features, self.device)
        logits = self.model(model_input)
        logits = normalize_binary_logits(logits)

        if self.num_left_context_frames > 0:
            logits = logits[self.num_left_context_frames :]

        if logits.shape[0] != current_features.shape[0]:
            raise ValueError(
                "Model output length does not match feature length after "
                "removing left context: "
                f"logits={tuple(logits.shape)}, "
                f"features={tuple(current_features.shape)}."
            )

        new_logits = logits[self.num_frames_emitted :]
        probabilities, predictions = logits_to_predictions(new_logits, self.threshold)

        start_frame = self.num_frames_emitted
        end_frame = start_frame + int(new_logits.shape[0])
        self.num_frames_emitted = end_frame

        num_context_to_keep = min(
            self.feature_context_frames,
            current_features.shape[0],
        )
        self.left_context_features = current_features[-num_context_to_keep:].detach().cpu()
        self.num_left_context_frames = num_context_to_keep

        max_samples_to_keep = (self.feature_context_frames + 4) * self.hop_length
        if self.sample_buffer.numel() > max_samples_to_keep:
            self.sample_buffer = self.sample_buffer[-max_samples_to_keep:].clone()

        return StreamingPrediction(
            probabilities=probabilities,
            predictions=predictions,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_sec=start_frame * self.frame_hop_sec,
            end_time_sec=end_frame * self.frame_hop_sec,
        )

    @torch.inference_mode()
    def flush(self) -> StreamingPrediction | None:
        """
        Process one incoming audio chunk.

        Args:
            chunk: Audio chunk of shape ``[N]`` or ``[1, N]``.

        Returns:
            Prediction for newly emitted frames, or ``None`` if there is not
            enough audio to emit output.

        Raises:
            ValueError: If the chunk shape is invalid or the model output
                length does not match the feature length.
        """
        if self.sample_buffer.numel() == 0:
            return None

        return self.process_chunk(torch.empty(0, dtype=torch.float32))
