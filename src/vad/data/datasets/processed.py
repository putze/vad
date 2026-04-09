from __future__ import annotations

from collections.abc import Sized
from typing import cast

from torch import Tensor
from torch.utils.data import Dataset

from vad.data.preprocessing.preprocessing import VADPreprocessor


class ProcessedVADDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Dataset wrapper that applies VAD preprocessing on top of a raw dataset.

    The wrapped dataset is expected to yield raw samples as:

    ``(waveform, labels, sample_rate)``

    where:
    - ``waveform`` has shape ``[num_samples]``,
    - ``labels`` has shape ``[num_samples]`` and contains sample-level targets,
    - ``sample_rate`` is given in Hz.

    This wrapper applies ``VADPreprocessor`` to convert each raw sample into:

    - ``features`` of shape ``[n_mels, num_frames]``,
    - ``aligned_labels`` of shape ``[num_frames]``.

    Here, ``num_frames`` refers to the frame axis after feature extraction and
    label alignment, not the original waveform sample count.
    """

    def __init__(
        self,
        base_dataset: Dataset[tuple[Tensor, Tensor, int]],
        processor: VADPreprocessor,
    ) -> None:
        """
        Initialize the processed dataset wrapper.

        Args:
            base_dataset: Dataset yielding raw samples as
                ``(waveform, labels, sample_rate)``.
            processor: Preprocessing pipeline that converts raw waveforms and
                sample-level labels into model-ready features and frame-level
                labels.
        """
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        """Return the number of samples in the wrapped dataset."""
        return len(cast(Sized, self.base_dataset))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve and preprocess one sample.

        Args:
            index: Sample index.

        Returns:
            A tuple ``(features, aligned_labels)`` where:
            - ``features`` has shape ``[n_mels, num_frames]``,
            - ``aligned_labels`` has shape ``[num_frames]``.

        Raises:
            ValueError: If the processor output has invalid dimensions or
                inconsistent frame counts.
        """
        waveform, labels, sample_rate = self.base_dataset[index]

        features, aligned_labels = self.processor(
            waveform,
            labels,
            sample_rate,
        )

        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape [n_mels, num_frames], got {tuple(features.shape)}"
            )

        if aligned_labels.ndim != 1:
            raise ValueError(
                f"Expected aligned labels with shape [num_frames], "
                f"got {tuple(aligned_labels.shape)}"
            )

        if features.shape[1] != aligned_labels.shape[0]:
            raise ValueError(
                f"Feature/label mismatch at index={index}: "
                f"features={tuple(features.shape)}, "
                f"labels={tuple(aligned_labels.shape)}"
            )

        return features.float(), aligned_labels.float()
