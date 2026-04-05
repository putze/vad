from __future__ import annotations

from collections.abc import Sized
from typing import cast

from torch import Tensor
from torch.utils.data import Dataset

from src.vad.data.preprocessing.preprocessing import VADPreprocessor


class ProcessedVADDataset(Dataset):
    """
    Wrap a base VAD dataset and apply preprocessing.

    Expects base samples as (waveform [T], labels [T], sample_rate),
    and returns (features [n_mels, T], aligned_labels [T]).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        processor: VADPreprocessor,
    ) -> None:
        """
        Args:
            base_dataset (Dataset): Dataset yielding raw audio samples.
            processor (VADPreprocessor): Preprocessing pipeline.
        """
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(cast(Sized, self.base_dataset))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve and preprocess a dataset sample.

        Args:
            index (int): Sample index.

        Returns:
            tuple[Tensor, Tensor]:
                - features: [n_mels, T]
                - aligned_labels: [T]

        Raises:
            ValueError: If output shapes are invalid or mismatched.
        """
        waveform, labels, sample_rate = self.base_dataset[index]

        features, aligned_labels = self.processor(
            waveform,
            labels,
            sample_rate,
        )

        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape [n_mels, T], got {tuple(features.shape)}"
            )

        if aligned_labels.ndim != 1:
            raise ValueError(
                f"Expected aligned labels with shape [T], got {tuple(aligned_labels.shape)}"
            )

        if features.shape[1] != aligned_labels.shape[0]:
            raise ValueError(
                f"Feature/label mismatch at index={index}: "
                f"features={tuple(features.shape)}, "
                f"labels={tuple(aligned_labels.shape)}"
            )

        return features.float(), aligned_labels.long()
