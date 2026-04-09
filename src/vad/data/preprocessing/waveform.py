from __future__ import annotations

import torch
import torchaudio
from torch import Tensor


class WaveformPreprocessor:
    """
    Preprocess raw waveforms while preserving label alignment.

    This class operates on time-domain mono waveforms and optional sample-level
    labels. It can:
    - resample the waveform to a target sample rate,
    - resize sample-level labels to match the resampled waveform length,
    - optionally apply peak normalization.

    The class is used both:
    - during training, where waveform and labels must remain synchronized,
    - during inference, where only waveform preprocessing is needed.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the waveform preprocessor.

        Args:
            target_sample_rate: Desired output sample rate in Hz.
            normalize: Whether to apply peak normalization after resampling.
        """
        if target_sample_rate <= 0:
            raise ValueError(f"`target_sample_rate` must be positive, got {target_sample_rate}")

        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _get_resampler(self, sample_rate: int) -> torchaudio.transforms.Resample:
        """
        Return a cached resampler from ``sample_rate`` to ``target_sample_rate``.

        Args:
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            A torchaudio resampler for the requested conversion.

        Raises:
            ValueError: If ``sample_rate`` is invalid or already matches the
                target sample rate.
        """
        if sample_rate <= 0:
            raise ValueError(f"`sample_rate` must be positive, got {sample_rate}")

        if sample_rate == self.target_sample_rate:
            raise ValueError(
                "_get_resampler should not be called when sample_rate "
                "already matches target_sample_rate"
            )

        resampler = self._resamplers.get(sample_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            self._resamplers[sample_rate] = resampler

        return resampler

    def _resize_labels(self, labels: Tensor, new_length: int) -> Tensor:
        """
        Resize sample-level labels to a new waveform length.

        Labels are remapped with nearest-neighbor indexing rather than
        continuous interpolation. This preserves discrete binary label values.

        Args:
            labels: Sample-level label tensor of shape ``[num_samples]``.
            new_length: Desired output length.

        Returns:
            Resized label tensor of shape ``[new_length]``.

        Raises:
            ValueError: If ``labels`` is not one-dimensional or if
                ``new_length`` is not positive.
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

    def _resample_waveform(self, waveform: Tensor, sample_rate: int) -> tuple[Tensor, int]:
        """
        Resample a mono waveform if needed.

        Args:
            waveform: Input waveform of shape ``[num_samples]``.
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            A tuple ``(waveform, sample_rate)`` containing the resampled
            waveform and the updated sample rate.
        """
        if sample_rate == self.target_sample_rate:
            return waveform, sample_rate

        original_length = waveform.shape[0]
        resampler = self._get_resampler(sample_rate)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        new_length = waveform.shape[0]
        if new_length <= 0:
            raise ValueError(
                "Resampling produced an invalid waveform length: "
                f"{new_length} from original length {original_length}"
            )

        return waveform, self.target_sample_rate

    def _normalize_waveform(self, waveform: Tensor) -> Tensor:
        """
        Peak-normalize a waveform if normalization is enabled.

        Args:
            waveform: Input waveform of shape ``[num_samples]``.

        Returns:
            Normalized waveform with the same shape.
        """
        if not self.normalize:
            return waveform

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        return waveform

    def process_waveform(self, waveform: Tensor, sample_rate: int) -> tuple[Tensor, int]:
        """
        Preprocess a waveform without labels.

        This method is intended for inference or other situations where only
        the waveform needs to be transformed.

        Args:
            waveform: Input mono waveform of shape ``[num_samples]``.
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            A tuple ``(waveform, sample_rate)`` containing the processed
            waveform and the updated sample rate.

        Raises:
            ValueError: If the waveform is not one-dimensional or the sample
                rate is invalid.
        """
        if waveform.ndim != 1:
            raise ValueError(f"`waveform` must be 1D, got shape {tuple(waveform.shape)}")

        if sample_rate <= 0:
            raise ValueError(f"`sample_rate` must be positive, got {sample_rate}")

        waveform = waveform.float()
        waveform, sample_rate = self._resample_waveform(waveform, sample_rate)
        waveform = self._normalize_waveform(waveform)

        return waveform, sample_rate

    def __call__(
        self,
        waveform: Tensor,
        labels: Tensor,
        sample_rate: int,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Preprocess a waveform and synchronized sample-level labels.

        If resampling changes the waveform length, labels are resized to the
        same length so that waveform samples and labels remain aligned.

        Args:
            waveform: Input mono waveform of shape ``[num_samples]``.
            labels: Sample-level labels of shape ``[num_samples]``.
            sample_rate: Original waveform sample rate in Hz.

        Returns:
            A tuple ``(waveform, labels, sample_rate)`` where:
            - ``waveform`` has shape ``[resampled_num_samples]``,
            - ``labels`` has shape ``[resampled_num_samples]``,
            - ``sample_rate`` is the updated sample rate in Hz.

        Raises:
            ValueError: If waveform or labels are not one-dimensional, if their
                input lengths differ, if ``sample_rate`` is invalid, or if the
                processed waveform and labels end up misaligned.
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

        original_length = waveform.shape[0]
        waveform, sample_rate = self._resample_waveform(waveform, sample_rate)

        if waveform.shape[0] != original_length:
            labels = self._resize_labels(labels, waveform.shape[0])

        waveform = self._normalize_waveform(waveform)

        if waveform.shape[0] != labels.shape[0]:
            raise ValueError(
                "Processed waveform and labels must have the same length, "
                f"got {waveform.shape[0]} and {labels.shape[0]}"
            )

        return waveform, labels, sample_rate
