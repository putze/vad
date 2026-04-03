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
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize

    def __call__(self, waveform: Tensor, sample_rate: int) -> tuple[Tensor, int]:
        """
        Apply preprocessing to a waveform.

        Args:
            waveform (Tensor): Input mono waveform [T].
            sample_rate (int): Original sample rate.

        Returns:
            tuple[Tensor, int]: Processed waveform [T] and updated sample rate.
        """
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform [T], got shape {tuple(waveform.shape)}")

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
