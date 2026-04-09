from __future__ import annotations

import torch
import torchaudio
from torch import Tensor


class LogMelFeatureExtractor:
    """
    Extract log-Mel spectrogram features from mono waveforms.

    The extractor expects input waveforms to already be sampled at the
    configured ``sample_rate``. It computes a Mel spectrogram on the waveform
    and returns its natural logarithm.

    Output features follow the shape convention:
    ``[n_mels, num_frames]``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 400,
        hop_length: int = 160,
        n_fft: int = 400,
        n_mels: int = 40,
        center: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize the log-Mel feature extractor.

        Args:
            sample_rate: Expected input sample rate in Hz.
            frame_length: Analysis window length in samples.
            hop_length: Frame hop in samples.
            n_fft: FFT size.
            n_mels: Number of Mel filter banks.
            center: Whether to use centered STFT-style framing.
            eps: Minimum value used before applying the logarithm for
                numerical stability.

        Raises:
            ValueError: If any configuration value is not positive.
        """
        if sample_rate <= 0:
            raise ValueError(f"`sample_rate` must be positive, got {sample_rate}")
        if frame_length <= 0:
            raise ValueError(f"`frame_length` must be positive, got {frame_length}")
        if hop_length <= 0:
            raise ValueError(f"`hop_length` must be positive, got {hop_length}")
        if n_fft <= 0:
            raise ValueError(f"`n_fft` must be positive, got {n_fft}")
        if n_mels <= 0:
            raise ValueError(f"`n_mels` must be positive, got {n_mels}")
        if eps <= 0:
            raise ValueError(f"`eps` must be positive, got {eps}")
        if frame_length > n_fft:
            raise ValueError(
                f"`frame_length` must be <= `n_fft`, got frame_length={frame_length}, n_fft={n_fft}"
            )

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.center = center
        self.eps = eps

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.frame_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=self.center,
            power=2.0,
        )

    def __call__(self, waveform: Tensor) -> Tensor:
        """
        Compute log-Mel features from a mono waveform.

        Args:
            waveform: Input waveform of shape ``[num_samples]``.

        Returns:
            Log-Mel feature tensor of shape ``[n_mels, num_frames]``.

        Raises:
            ValueError: If the input waveform is not one-dimensional.
        """
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected 1D waveform [num_samples], got shape {tuple(waveform.shape)}"
            )

        waveform = waveform.float()

        mel = self.transform(waveform.unsqueeze(0)).squeeze(0)
        mel = torch.clamp(mel, min=self.eps)  # Clamp to avoid log(0).
        log_mel = torch.log(mel)  # Natural log

        return log_mel

    @property
    def frame_hop_seconds(self) -> float:
        """
        Return the temporal spacing between consecutive feature frames.

        Returns:
            Frame hop duration in seconds.
        """
        return self.hop_length / self.sample_rate
