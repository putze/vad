from torch import Tensor

from vad.data.audio_utils import ensure_mono_waveform
from vad.data.preprocessing.features import LogMelFeatureExtractor
from vad.data.preprocessing.waveform import WaveformPreprocessor
from vad.inference.utils import ensure_time_major_features


class StreamingFeatureExtractorAdapter:
    """Adapt preprocessing and feature extraction for streaming inference."""

    def __init__(
        self,
        waveform_preprocessor: WaveformPreprocessor,
        feature_extractor: LogMelFeatureExtractor,
        n_mels: int,
    ) -> None:
        self.waveform_preprocessor = waveform_preprocessor
        self.feature_extractor = feature_extractor
        self.n_mels = n_mels

    def extract(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """
        Extract time-major features from a waveform.

        Args:
            waveform: Input waveform.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            Feature tensor of shape ``[T, F]``.
        """
        waveform = ensure_mono_waveform(waveform)
        waveform, sample_rate = self.waveform_preprocessor.process_waveform(
            waveform,
            sample_rate,
        )
        features = self.feature_extractor(waveform)
        return ensure_time_major_features(features, feature_dim=self.n_mels)
