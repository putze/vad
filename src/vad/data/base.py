from __future__ import annotations

import warnings
from collections.abc import Sized
from pathlib import Path
from typing import Generic, Protocol, Sequence, TypeVar, cast

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.vad.data.preprocessing.preprocessing import VADPreprocessor


class VADSampleProtocol(Protocol):
    """Protocol for dataset samples containing audio and label paths."""

    audio_path: Path
    label_path: Path


SampleType = TypeVar("SampleType", bound=VADSampleProtocol)


class BaseVADDataset(Dataset, Generic[SampleType]):
    """
    Base dataset class for Voice Activity Detection (VAD).

    Handles loading of audio waveforms and corresponding label arrays.
    """

    def __init__(self, samples: Sequence[SampleType]) -> None:
        """
        Initialize VAD Dataset.

        Args:
            samples (Sequence[SampleType]): Collection of dataset samples.
        """
        self.samples = list(samples)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def _load_audio(self, path: Path) -> tuple[Tensor, int]:
        """
        Load an audio file as a mono waveform.

        Args:
            path (Path): Path to audio file.

        Returns:
            tuple[Tensor, int]: Waveform tensor [T] and sample rate.
        """
        try:
            waveform, sample_rate = torchaudio.load(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {path}") from e

        if waveform.ndim != 2:
            raise ValueError(
                f"Expected audio tensor with shape [channels, time], "
                f"got {tuple(waveform.shape)} for file: {path}"
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        return waveform.float(), sample_rate

    def _load_labels(self, path: Path) -> Tensor:
        """
        Load label array from a .npy file.

        Args:
            path (Path): Path to label file.

        Returns:
            Tensor: Label tensor.
        """
        try:
            labels = np.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load label file: {path}") from e

        labels_tensor = torch.from_numpy(labels)

        if labels_tensor.ndim != 1:
            raise ValueError(
                f"Expected 1D label array, got shape {tuple(labels_tensor.shape)} for file: {path}"
            )

        return labels_tensor.float()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        """
        Retrieve one raw dataset sample.

        Args:
            index (int): Sample index.

        Returns:
            tuple[Tensor, Tensor, int]: Waveform, labels, and sample rate.
        """
        sample = self.samples[index]

        waveform, sample_rate = self._load_audio(sample.audio_path)
        labels = self._load_labels(sample.label_path)

        waveform_len = waveform.shape[0]
        labels_len = labels.shape[0]

        # Check if waveform and labels have the same lengths
        diff = abs(waveform_len - labels_len)

        if waveform_len != labels_len:
            # Crop if small difference
            if diff <= 1:
                warnings.warn(
                    f"Cropping audio/label mismatch: "
                    f"{sample.audio_path.name} waveform={waveform_len}, labels={labels_len}"
                )
                min_len = min(waveform_len, labels_len)
                waveform = waveform[:min_len]
                labels = labels[:min_len]
            else:
                raise ValueError(
                    "Waveform and labels must have the same length, "
                    f"got waveform={waveform_len} and labels={labels_len} "
                    f"(diff={diff}) "
                    f"for sample audio={sample.audio_path} label={sample.label_path}"
                )

        return waveform, labels, sample_rate


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
