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
    ) -> None:
        """
        Args:
            sample_rate (int): Expected input sample rate.
            frame_length (int): Window size (in samples) for STFT.
            hop_length (int): Hop size (in samples) between frames.
            n_fft (int): FFT size.
            n_mels (int): Number of Mel filter banks.
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=frame_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """
        Compute log-Mel spectrogram features.

        Args:
            waveform (Tensor): Input mono waveform [T].
            sample_rate (int): Sample rate of the waveform.

        Returns:
            Tensor: Log-Mel features [n_mels, num_frames].
        """
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform [T], got shape {tuple(waveform.shape)}")

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Feature extractor expected sample_rate={self.sample_rate}, got {sample_rate}"
            )

        mel = self.transform(waveform.unsqueeze(0)).squeeze(0)  # [n_mels, frames]
        log_mel = torch.log(mel + 1e-6)
        return log_mel

    @property
    def frame_hop_seconds(self) -> float:
        """
        Duration (in seconds) between consecutive feature frames.

        Returns:
            float: Hop length in seconds.
        """
        return self.hop_length / self.sample_rate
