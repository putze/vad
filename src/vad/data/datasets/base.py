from __future__ import annotations

import warnings
from pathlib import Path
from typing import Generic, Protocol, Sequence, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.vad.data.file_utils import load_audio


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

        waveform, sample_rate = load_audio(sample.audio_path)
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
