from __future__ import annotations

import warnings
from pathlib import Path
from typing import Generic, Protocol, Sequence, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from vad.data.file_utils import load_audio


class VADSampleProtocol(Protocol):
    """
    Protocol for dataset sample metadata.

    Any sample used by ``BaseVADDataset`` must provide paths to:
    - an audio file,
    - a NumPy label file containing sample-level speech labels.
    """

    audio_path: Path
    label_path: Path


SampleType = TypeVar("SampleType", bound=VADSampleProtocol)


class BaseVADDataset(Dataset[tuple[Tensor, Tensor, int]], Generic[SampleType]):
    """
    Base dataset for sample-level voice activity detection.

    Each item loads:
    - a mono waveform of shape ``[num_samples]``,
    - a 1D label tensor of shape ``[num_samples]``,
    - the sample rate in Hz.

    This dataset assumes labels are defined at the sample level, not the frame
    level. Small off-by-one mismatches between waveform and label lengths are
    tolerated and resolved by cropping both to the shorter length.
    """

    def __init__(
        self,
        samples: Sequence[SampleType],
        max_length_mismatch: int = 1,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            samples: Collection of dataset samples.
            max_length_mismatch: Maximum allowed absolute difference between
                waveform and label lengths. If the mismatch is within this
                threshold, both are cropped to the shorter length. Larger
                mismatches raise an error.
        """
        if max_length_mismatch < 0:
            raise ValueError(
                f"`max_length_mismatch` must be non-negative, got {max_length_mismatch}"
            )

        self.samples = list(samples)
        self.max_length_mismatch = max_length_mismatch

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def _load_labels(self, path: Path) -> Tensor:
        """
        Load sample-level labels from a ``.npy`` file.

        Args:
            path: Path to the label file.

        Returns:
            A float tensor of shape ``[num_samples]``.

        Raises:
            RuntimeError: If the file cannot be loaded.
            ValueError: If the loaded array is not one-dimensional.
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
        Load one dataset item.

        Args:
            index: Sample index.

        Returns:
            A tuple ``(waveform, labels, sample_rate)`` where:
            - ``waveform`` has shape ``[num_samples]``,
            - ``labels`` has shape ``[num_samples]``,
            - ``sample_rate`` is in Hz.

        Raises:
            ValueError: If waveform and label lengths differ by more than the
                configured tolerance.
        """
        sample = self.samples[index]

        waveform, sample_rate = load_audio(sample.audio_path)
        labels = self._load_labels(sample.label_path)

        waveform_len = waveform.shape[0]
        labels_len = labels.shape[0]
        diff = abs(waveform_len - labels_len)

        if waveform_len != labels_len:
            # Allow tiny mismatches caused by preprocessing or serialization.
            if diff <= self.max_length_mismatch:
                warnings.warn(
                    "Cropping audio/label length mismatch: "
                    f"{sample.audio_path.name} waveform={waveform_len}, labels={labels_len}"
                )
                min_len = min(waveform_len, labels_len)
                waveform = waveform[:min_len]
                labels = labels[:min_len]
            else:
                raise ValueError(
                    "Waveform and labels must have the same length, "
                    f"got waveform={waveform_len} and labels={labels_len} "
                    f"(diff={diff}) for sample "
                    f"audio={sample.audio_path} label={sample.label_path}"
                )

        return waveform, labels, sample_rate
