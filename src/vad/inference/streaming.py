from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from vad.config import AudioConfig, InferenceConfig, StreamingConfig
from vad.inference.utils import (
    FeatureExtractorProtocol,
    logits_to_predictions,
    normalize_binary_logits,
    prepare_conv1d_input,
)


@dataclass(slots=True)
class StreamingPrediction:
    """
    Container for frame-level predictions emitted for one streaming update.
    """

    probabilities: Tensor
    predictions: Tensor
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float


class StreamingVADInferencer:
    """
    Run stateful chunk-based VAD inference on streaming audio.

    This implementation accumulates all received audio, re-extracts features
    from the full buffered waveform, runs the model on the full feature
    sequence, and emits only frames that have not been returned previously.

    This design prioritizes correctness and simplicity:
        - frame indexing stays globally consistent
        - no fragile bookkeeping is needed when trimming the waveform buffer
        - no assumptions are required about the feature extractor's exact
          frame/window alignment beyond its hop length

    The tradeoff is that inference cost grows with stream duration because the
    full waveform is reprocessed on each update.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: FeatureExtractorProtocol,
        *,
        audio_config: AudioConfig,
        inference_config: InferenceConfig | None = None,
        streaming_config: StreamingConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the streaming inference pipeline.

        Args:
            model: Trained VAD model.
            feature_extractor: Feature extractor applied to the buffered audio.
                It must return features with shape ``[T, F]``.
            audio_config: Shared audio configuration used during training.
            inference_config: Inference-time settings such as threshold.
            streaming_config: Streaming-specific settings.
            device: Torch device used for inference.
        """
        inference_config = inference_config or InferenceConfig()
        streaming_config = streaming_config or StreamingConfig()

        if audio_config.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {audio_config.sample_rate}")
        if audio_config.hop_length_samples <= 0:
            raise ValueError(f"hop_length must be positive, got {audio_config.hop_length_samples}")
        if not 0.0 <= inference_config.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {inference_config.threshold}")
        if streaming_config.chunk_seconds <= 0:
            raise ValueError(
                f"chunk_seconds must be positive, got {streaming_config.chunk_seconds}"
            )
        if streaming_config.min_buffer_seconds < 0:
            raise ValueError(
                "min_buffer_seconds must be non-negative, "
                f"got {streaming_config.min_buffer_seconds}"
            )

        self.model = model.eval()
        self.feature_extractor = feature_extractor
        self.audio_config = audio_config
        self.inference_config = inference_config
        self.streaming_config = streaming_config
        self.min_buffer_samples = max(
            audio_config.frame_length_samples,
            int(round(streaming_config.min_buffer_seconds * audio_config.sample_rate)),
        )
        self.device = device or torch.device("cpu")

        self.model.to(self.device)
        self.reset()

    @property
    def sample_rate(self) -> int:
        """
        Return sampling rate of the audio signal.

        Returns:
            Sampling rate in Hz.
        """
        return self.audio_config.sample_rate

    @property
    def hop_length(self) -> int:
        """
        Return the hop length between consecutive frames in samples.

        Returns:
            Number of samples between two consecutive frames.
        """
        return self.audio_config.hop_length_samples

    @property
    def threshold(self) -> float:
        """
        Return the ecision threshold applied to model outputs.

        Returns:
            Threshold value in the range [0, 1].
        """
        return self.inference_config.threshold

    @property
    def frame_hop_sec(self) -> float:
        """
        Return the duration of one frame hop in seconds.

        Returns:
            Frame hop duration in seconds.
        """
        return self.hop_length / self.sample_rate

    def reset(self) -> None:
        """
        Reset the internal streaming state.

        This clears the buffered waveform and the counter tracking how many
        frame predictions have already been emitted.
        """
        self.sample_buffer = torch.empty(0, dtype=torch.float32)
        self.num_samples_seen = 0
        self.num_frames_emitted = 0

    def _extract_features(self, waveform: Tensor) -> Tensor:
        """
        Extract frame-level features from a waveform.

        Args:
            waveform: Mono waveform of shape ``[N]``.

        Returns:
            Feature tensor of shape ``[T, F]``.

        Raises:
            ValueError: If the feature extractor returns an unexpected shape.
        """
        features = self.feature_extractor.extract(waveform, self.sample_rate)
        if features.ndim != 2:
            raise ValueError(
                f"Expected feature tensor with shape [T, F], got {tuple(features.shape)}."
            )
        return features

    def _normalize_chunk(self, chunk: Tensor) -> Tensor:
        """
        Normalize an incoming chunk to shape ``[N]`` on CPU as float32.

        Args:
            chunk: Audio chunk with shape ``[N]`` or ``[1, N]``.

        Returns:
            Mono chunk of shape ``[N]`` on CPU with dtype ``float32``.

        Raises:
            ValueError: If the chunk shape is unsupported.
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

        return chunk.detach().cpu().float()

    @torch.inference_mode()
    def process_chunk(self, chunk: Tensor) -> StreamingPrediction | None:
        """
        Process one incoming audio chunk.

        The chunk is appended to the internal waveform buffer. Features are
        recomputed on the full buffered waveform, the model is run on the full
        feature sequence, and only the frames that have not already been
        emitted are returned.

        Args:
            chunk: Audio chunk of shape ``[N]`` or ``[1, N]``.

        Returns:
            A ``StreamingPrediction`` for newly available frames, or ``None``
            if there is not enough buffered audio or if no new frames became
            available.

        Raises:
            ValueError: If the chunk shape is invalid or the model output shape
                is unsupported.
        """
        chunk = self._normalize_chunk(chunk)

        self.sample_buffer = torch.cat([self.sample_buffer, chunk], dim=0)
        self.num_samples_seen += int(chunk.numel())

        if self.sample_buffer.numel() < self.min_buffer_samples:
            return None

        current_features = self._extract_features(self.sample_buffer)
        num_current_frames = int(current_features.shape[0])

        if num_current_frames == 0:
            return None

        if num_current_frames <= self.num_frames_emitted:
            return None

        model_input = prepare_conv1d_input(current_features, self.device)
        logits = self.model(model_input)
        logits = normalize_binary_logits(logits)

        if logits.shape[0] != num_current_frames:
            raise ValueError(
                "Model output length does not match feature length: "
                f"logits={tuple(logits.shape)}, "
                f"features={tuple(current_features.shape)}."
            )

        new_logits = logits[self.num_frames_emitted :]
        if new_logits.numel() == 0:
            return None

        probabilities, predictions = logits_to_predictions(new_logits, self.threshold)

        start_frame = self.num_frames_emitted
        end_frame = num_current_frames
        self.num_frames_emitted = end_frame

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
        Flush any remaining buffered audio through the pipeline.

        This method does not add new audio. It simply reruns inference on the
        current buffer and returns any frame predictions that have not yet been
        emitted.

        Returns:
            A ``StreamingPrediction`` for any remaining newly emitted frames,
            or ``None`` if nothing remains to emit.
        """
        if self.sample_buffer.numel() == 0:
            return None

        return self.process_chunk(torch.empty(0, dtype=torch.float32))
