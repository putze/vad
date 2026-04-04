from pathlib import Path
from typing import Generic, Protocol, Sequence, TypeVar

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


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

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve one raw dataset sample.

        Args:
            index (int): Sample index.

        Returns:
            tuple[Tensor, Tensor]: Waveform and labels.
        """
        sample = self.samples[index]

        waveform, _ = self._load_audio(sample.audio_path)
        labels = self._load_labels(sample.label_path)

        if waveform.shape[0] != labels.shape[0]:
            raise ValueError(
                "Waveform and labels must have the same length, "
                f"got waveform={waveform.shape[0]} and labels={labels.shape[0]} "
                f"for sample audio={sample.audio_path} label={sample.label_path}"
            )

        return waveform, labels
