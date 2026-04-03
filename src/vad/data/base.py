from __future__ import annotations

from pathlib import Path
from typing import Generic, Sequence, TypeVar

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.vad.data.preprocessing.preprocessing import VADPreprocessor

SampleType = TypeVar("SampleType")


class BaseVADDataset(Dataset, Generic[SampleType]):
    """
    Base dataset class for Voice Activity Detection (VAD).

    Handles loading of audio waveforms and corresponding label arrays,
    with optional preprocessing.
    """

    def __init__(
        self,
        samples: Sequence[SampleType],
        sample_rate: int = 16000,
        extensions: tuple[str, ...] = (".wav",),
        preprocessor: VADPreprocessor | None = None,
    ) -> None:
        """
        Initialize VAD Dataset.

        Args:
            samples (Sequence[SampleType]): Collection of dataset samples.
            sample_rate (int): Target audio sample rate.
            extensions (tuple[str, ...]): Allowed audio file extensions.
            preprocessor (VADPreprocessor | None): Optional preprocessing callable.
        """
        self.samples = list(samples)
        self.sample_rate = sample_rate
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def _is_audio_file(self, path: Path) -> bool:
        """
        Check if a path corresponds to a valid audio file.

        Args:
            path (Path): File path to check.

        Returns:
            bool: True if valid audio file, else False.
        """
        return path.is_file() and path.suffix.lower() in self.extensions

    def _iter_audio_files(self, root: Path) -> list[Path]:
        """
        Recursively collect all valid audio files under a directory.

        Args:
            root (Path): Root directory.

        Returns:
            list[Path]: Sorted list of audio file paths.
        """
        return [path for path in sorted(root.rglob("*")) if self._is_audio_file(path)]

    def _load_audio(self, path: Path) -> tuple[Tensor, int]:
        """
        Load an audio file as a mono waveform.

        Args:
            path (Path): Path to audio file.

        Returns:
            tuple[Tensor, int]: Waveform tensor [T] and sample rate.
        """
        waveform, sample_rate = torchaudio.load(str(path))

        # Convert to mono [T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        waveform = waveform.float()
        return waveform, sample_rate

    def _load_labels(self, path: Path) -> Tensor:
        """
        Load label array from a .npy file.

        Args:
            path (Path): Path to label file.

        Returns:
            Tensor: Label tensor.
        """
        labels = np.load(path)
        return torch.from_numpy(labels).float()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve a dataset sample.

        Loads audio and labels, and applies optional preprocessing.

        Args:
            index (int): Sample index.

        Returns:
            tuple[Tensor, Tensor]: Waveform and aligned labels.
        """
        sample = self.samples[index]

        waveform, sample_rate = self._load_audio(sample.audio_path)
        labels = self._load_labels(sample.label_path)

        if self.preprocessor is not None:
            waveform, labels = self.preprocessor(waveform, labels, sample_rate)

        return waveform, labels
