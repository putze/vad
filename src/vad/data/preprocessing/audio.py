from __future__ import annotations

import torch
import torchaudio
from torch import Tensor


class AudioPreprocessor:
    """
    Preprocess raw audio waveforms by resampling and optional normalization.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            target_sample_rate (int): Desired output sample rate.
            normalize (bool): Whether to apply peak normalization.
        """
        if target_sample_rate <= 0:
            raise ValueError(f"`target_sample_rate` must be positive, got {target_sample_rate}")

        self.target_sample_rate = target_sample_rate
        self.normalize = normalize

    def _resize_labels(self, labels: Tensor, new_length: int) -> Tensor:
        """
        Resize 1D sample-level labels to a new length using nearest-neighbor mapping.

        This preserves binary label semantics better than continuous interpolation.

        Args:
            labels: Sample-level label tensor of shape [T].
            new_length: Desired output length.

        Returns:
            Resized label tensor of shape [new_length].
        """
        if labels.ndim != 1:
            raise ValueError(f"`labels` must be 1D, got shape {tuple(labels.shape)}")

        if new_length <= 0:
            raise ValueError(f"`new_length` must be positive, got {new_length}")

        old_length = labels.shape[0]

        if old_length == new_length:
            return labels

        indices = torch.linspace(
            0,
            old_length - 1,
            steps=new_length,
            device=labels.device,
        )
        indices = indices.round().long().clamp(0, old_length - 1)

        return labels[indices]

    def process_waveform(self, waveform: Tensor, sample_rate: int) -> tuple[Tensor, int]:
        """
        Apply preprocessing to a waveform only, without labels.

        Args:
            waveform: Input mono waveform [T].
            sample_rate: Original sample rate.

        Returns:
            tuple[Tensor, int]: Processed waveform [T] and updated sample rate.
        """
        if waveform.ndim != 1:
            raise ValueError(f"`waveform` must be 1D, got shape {tuple(waveform.shape)}")

        if sample_rate <= 0:
            raise ValueError(f"`sample_rate` must be positive, got {sample_rate}")

        waveform = waveform.float()

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            ).squeeze(0)
            sample_rate = self.target_sample_rate

        if self.normalize:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        return waveform, sample_rate

    def __call__(
        self, waveform: Tensor, labels: Tensor, sample_rate: int
    ) -> tuple[Tensor, Tensor, int]:
        """
        Apply preprocessing to a waveform.

        Args:
            waveform (Tensor): Input mono waveform [T].
            labels (Tensor): Input sample-level labels [T]
            sample_rate (int): Original sample rate.

        Returns:
            tuple[Tensor, int]: Processed waveform [T], resized sample-labels [T],
                and updated sample rate.
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

        if sample_rate <= 0:
            raise ValueError(f"`sample_rate` must be positive, got {sample_rate}")

        waveform = waveform.float()
        labels = labels.float()

        if sample_rate != self.target_sample_rate:
            original_length = waveform.shape[0]

            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            ).squeeze(0)

            new_length = waveform.shape[0]

            if new_length <= 0:
                raise ValueError(
                    "Resampling produced an invalid waveform length: "
                    f"{new_length} from original length {original_length}"
                )

            labels = self._resize_labels(labels, new_length)
            sample_rate = self.target_sample_rate

        if self.normalize:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        if waveform.shape[0] != labels.shape[0]:
            raise ValueError(
                "Processed waveform and labels must have the same length, "
                f"got {waveform.shape[0]} and {labels.shape[0]}"
            )

        return waveform, labels, sample_rate
