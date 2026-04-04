from __future__ import annotations

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

    Handles loading of audio waveforms and corresponding label arrays,
    with optional preprocessing.
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

        if waveform.shape[0] != labels.shape[0]:
            raise ValueError(
                "Waveform and labels must have the same length, "
                f"got waveform={waveform.shape[0]} and labels={labels.shape[0]} "
                f"for sample audio={sample.audio_path} label={sample.label_path}"
            )

        return waveform, labels, sample_rate


class ProcessedVADDataset(Dataset):
    """
    Wrap a raw VAD dataset and apply sample processing.

    The base dataset must return:
        waveform: Tensor [T]
        labels: Tensor [T]
        sample_rate: int

    This dataset returns:
        features: Tensor [n_mels, num_frames]
        aligned_labels: Tensor [num_frames]
    """

    def __init__(
        self,
        base_dataset: Dataset,
        processor: VADPreprocessor,
    ) -> None:
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(cast(Sized, self.base_dataset))

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        waveform, labels, sample_rate = self.base_dataset[idx]

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
                f"Feature/label mismatch at idx={idx}: "
                f"features={tuple(features.shape)}, "
                f"labels={tuple(aligned_labels.shape)}"
            )

        return features.float(), aligned_labels.long()
