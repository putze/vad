import torch
import torchaudio
from torch import Tensor


class LogMelFeatureExtractor:
    """
    Extract log-Mel spectrogram features from audio waveforms.
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
        Args:
            sample_rate (int): Expected input sample rate.
            frame_length (int): Window size (in samples) for STFT.
            hop_length (int): Hop size (in samples) between frames.
            n_fft (int): FFT size.
            n_mels (int): Number of Mel filter banks.
            center (bool): Whether STFT-style framing is centered.
            eps (float): Minimum value before applying logarithm for numerical stability.
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
        Compute log-Mel spectrogram features.

        Args:
            waveform (Tensor): Input mono waveform [T].

        Returns:
            Tensor: Log-Mel features [n_mels, num_frames].
        """
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform [T], got shape {tuple(waveform.shape)}")

        waveform = waveform.float()

        mel = self.transform(waveform.unsqueeze(0)).squeeze(0)
        mel = torch.clamp(mel, min=self.eps)
        log_mel = torch.log(mel)

        return log_mel

    @property
    def frame_hop_seconds(self) -> float:
        """
        Duration (in seconds) between consecutive feature frames.

        Returns:
            float: Hop length in seconds.
        """
        return self.hop_length / self.sample_rate
