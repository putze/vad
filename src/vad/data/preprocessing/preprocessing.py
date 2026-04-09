from __future__ import annotations

from torch import Tensor

from vad.data.preprocessing.features import LogMelFeatureExtractor
from vad.data.preprocessing.labels import LabelAligner
from vad.data.preprocessing.waveform import WaveformPreprocessor


class VADPreprocessor:
    """
    End-to-end preprocessing pipeline for voice activity detection.

    This class converts a raw sample consisting of:
    - a waveform of shape ``[num_samples]``,
    - sample-level labels of shape ``[num_samples]``,
    - an input sample rate,

    into model-ready tensors:
    - log-Mel features of shape ``[n_mels, num_frames]``,
    - frame-level labels of shape ``[num_frames]``.

    The pipeline consists of three stages:
    1. audio preprocessing,
    2. feature extraction,
    3. label alignment to the feature frame grid.
    """

    def __init__(
        self,
        waveform_preprocessor: WaveformPreprocessor,
        feature_extractor: LogMelFeatureExtractor,
        label_aligner: LabelAligner,
    ) -> None:
        """
        Initialize the VAD preprocessing pipeline.

        Args:
            waveform_preprocessor: Module that preprocesses waveform, labels, and
                sample rate while preserving their alignment.
            feature_extractor: Module that extracts acoustic features from the
                processed waveform.
            label_aligner: Module that converts sample-level labels into
                frame-level labels matching the feature time axis.
        """
        self.waveform_preprocessor = waveform_preprocessor
        self.feature_extractor = feature_extractor
        self.label_aligner = label_aligner

    def __call__(
        self,
        waveform: Tensor,
        labels: Tensor,
        sample_rate: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Run the full preprocessing pipeline on one sample.

        Args:
            waveform: Raw mono waveform of shape ``[num_samples]``.
            labels: Sample-level binary labels of shape ``[num_samples]``.
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            A tuple ``(features, aligned_labels)`` where:
            - ``features`` has shape ``[n_mels, num_frames]``,
            - ``aligned_labels`` has shape ``[num_frames]``.

        Raises:
            ValueError: If the input waveform or labels are not one-dimensional,
                if their lengths do not match, or if the final frame counts are
                inconsistent.
        """
        if waveform.ndim != 1:
            raise ValueError(f"`waveform` must be 1D, got shape {tuple(waveform.shape)}")

        if labels.ndim != 1:
            raise ValueError(f"`labels` must be 1D, got shape {tuple(labels.shape)}")

        if waveform.shape[0] != labels.shape[0]:
            raise ValueError(
                "`waveform` and `labels` must have the same length, "
                f"got {waveform.shape[0]} and {labels.shape[0]}"
            )

        waveform, labels, sample_rate = self.waveform_preprocessor(
            waveform,
            labels,
            sample_rate,
        )
        features = self.feature_extractor(waveform)
        aligned_labels = self.label_aligner(labels, num_frames=features.shape[-1])

        if aligned_labels.shape[0] != features.shape[-1]:
            raise ValueError(
                "Aligned labels and extracted features must have matching frame counts, "
                f"got labels={aligned_labels.shape[0]} and features={features.shape[-1]}"
            )

        return features, aligned_labels
