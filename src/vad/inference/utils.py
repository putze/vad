from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class FeatureExtractorProtocol(Protocol):
    """
    Protocol for waveform-to-feature extractors used during inference.

    Implementations are expected to transform a mono waveform into a feature
    representation suitable for the VAD model.
    """

    def extract(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """
        Extract features from a waveform.

        Args:
            waveform: Mono waveform of shape ``[N]``.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            Feature tensor derived from the input waveform.
        """
        ...


def ensure_time_major_features(
    features: Tensor,
    *,
    feature_dim: int | None = None,
) -> Tensor:
    """
    Normalize features to shape ``[T, F]`` when the layout can be inferred.

    Supported inputs:
        - ``[T, F]``
        - ``[F, T]``
        - ``[1, T, F]``
        - ``[1, F, T]``

    If ``feature_dim`` is provided, the function uses it to infer which axis
    corresponds to the feature dimension and transposes when needed.

    If ``feature_dim`` is not provided, 2D inputs are returned unchanged.
    In that case, the caller is responsible for ensuring that the tensor is
    already time-major.

    Args:
        features: Feature tensor.
        feature_dim: Expected feature dimension. When provided, one dimension
            of the 2D tensor must match this value.

    Returns:
        Feature tensor of shape ``[T, F]``.

    Raises:
        ValueError: If the feature shape is unsupported or ambiguous.
    """
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError(
                f"Expected batched features with batch size 1, got {tuple(features.shape)}."
            )
        features = features.squeeze(0)

    if features.ndim != 2:
        raise ValueError(
            "Expected features with shape [T, F], [F, T], or batched variants, "
            f"got {tuple(features.shape)}."
        )

    if feature_dim is not None:
        if features.shape[0] == feature_dim and features.shape[1] == feature_dim:
            raise ValueError(
                "Ambiguous feature shape: both dimensions match feature_dim="
                f"{feature_dim}. Got {tuple(features.shape)}."
            )

        if features.shape[0] == feature_dim:
            return features.transpose(0, 1)

        if features.shape[1] == feature_dim:
            return features

        raise ValueError(
            f"Expected one feature dimension to equal {feature_dim}, got {tuple(features.shape)}."
        )

    return features


def prepare_conv1d_input(features: Tensor, device: torch.device) -> Tensor:
    """
    Convert time-major features to Conv1d input format.

    PyTorch ``Conv1d`` expects input shaped as ``[batch, channels, time]``.
    This function converts time-major features ``[T, F]`` into ``[1, F, T]``
    and moves the tensor to the target device.

    Args:
        features: Feature tensor of shape ``[T, F]``.
        device: Target device.

    Returns:
        Tensor of shape ``[1, F, T]``.
    """
    return features.transpose(0, 1).unsqueeze(0).to(device)


def normalize_binary_logits(logits: Tensor) -> Tensor:
    """
    Normalize binary model outputs to shape ``[T]``.

    Supported inputs:
        - ``[1, T]``
        - ``[1, 1, T]``
        - ``[1, T, 1]``

    Args:
        logits: Raw model output.

    Returns:
        Logit tensor of shape ``[T]``.

    Raises:
        ValueError: If the output shape is unsupported.
    """
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected logits of shape [1, T], got {tuple(logits.shape)}.")
        return logits.squeeze(0)

    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected logits with batch size 1, got {tuple(logits.shape)}.")

        if logits.shape[1] == 1:
            return logits.squeeze(0).squeeze(0)

        if logits.shape[2] == 1:
            return logits.squeeze(0).squeeze(-1)

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}.")


def logits_to_predictions(logits: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    """
    Convert logits to probabilities and binary predictions.

    Args:
        logits: Frame-level logits of shape ``[T]``.
        threshold: Probability threshold for speech detection.

    Returns:
        Tuple ``(probabilities, predictions)``, where:
            - ``probabilities`` has shape ``[T]`` and dtype float
            - ``predictions`` has shape ``[T]`` and dtype int64

    Raises:
        ValueError: If ``threshold`` is outside the interval [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    probabilities = torch.sigmoid(logits).cpu()
    predictions = (probabilities >= threshold).to(torch.int64)
    return probabilities, predictions


def predictions_to_segments(
    predictions: Tensor,
    frame_shift_ms: float,
    min_speech_ms: float = 100.0,
) -> list[tuple[float, float]]:
    """
    Convert binary frame predictions into speech segments.

    Consecutive frames predicted as speech are merged into a single segment.
    Segments shorter than ``min_speech_ms`` are discarded.

    Args:
        predictions: Binary frame predictions of shape ``[T]``.
        frame_shift_ms: Frame hop in milliseconds.
        min_speech_ms: Minimum speech segment duration to keep.

    Returns:
        List of ``(start_sec, end_sec)`` speech segments.

    Raises:
        ValueError: If ``predictions`` is not 1D, or if timing arguments are invalid.
    """
    if predictions.ndim != 1:
        raise ValueError(f"Expected predictions with shape [T], got {tuple(predictions.shape)}.")
    if frame_shift_ms <= 0:
        raise ValueError(f"frame_shift_ms must be positive, got {frame_shift_ms}")
    if min_speech_ms < 0:
        raise ValueError(f"min_speech_ms must be non-negative, got {min_speech_ms}")

    segments: list[tuple[float, float]] = []
    start_idx: int | None = None

    for i, value in enumerate(predictions.tolist()):
        if value == 1 and start_idx is None:
            start_idx = i
        elif value == 0 and start_idx is not None:
            end_idx = i
            start_sec = start_idx * frame_shift_ms / 1000.0
            end_sec = end_idx * frame_shift_ms / 1000.0
            if (end_sec - start_sec) * 1000.0 >= min_speech_ms:
                segments.append((start_sec, end_sec))
            start_idx = None

    if start_idx is not None:
        end_idx = len(predictions)
        start_sec = start_idx * frame_shift_ms / 1000.0
        end_sec = end_idx * frame_shift_ms / 1000.0
        if (end_sec - start_sec) * 1000.0 >= min_speech_ms:
            segments.append((start_sec, end_sec))

    return segments
