from torch import Tensor

from src.vad.data.preprocessing.audio import AudioPreprocessor
from src.vad.data.preprocessing.features import LogMelFeatureExtractor
from src.vad.data.preprocessing.labels import LabelAligner


class VADPreprocessor:
    """
    End-to-end preprocessing pipeline for VAD.

    Applies audio preprocessing, feature extraction, and label alignment.
    """

    def __init__(
        self,
        audio_preprocessor: AudioPreprocessor,
        feature_extractor: LogMelFeatureExtractor,
        label_aligner: LabelAligner,
    ) -> None:
        """
        Args:
            audio_preprocessor (AudioPreprocessor): Audio preprocessing step.
            feature_extractor (LogMelFeatureExtractor): Feature extraction module.
            label_aligner (LabelAligner): Label alignment module.
        """
        self.audio_preprocessor = audio_preprocessor
        self.feature_extractor = feature_extractor
        self.label_aligner = label_aligner

    def __call__(
        self,
        waveform: Tensor,
        labels: Tensor,
        sample_rate: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Run the full preprocessing pipeline.

        Args:
            waveform (Tensor): Raw waveform [T].
            labels (Tensor): Sample-level labels [L].
            sample_rate (int): Original sample rate.

        Returns:
            tuple[Tensor, Tensor]:
                - features: Log-Mel features [n_mels, num_frames]
                - aligned_labels: Frame-level labels [num_frames]
        """
        waveform, sample_rate = self.audio_preprocessor(waveform, sample_rate)
        features = self.feature_extractor(waveform, sample_rate)
        aligned_labels = self.label_aligner(labels, num_frames=features.shape[-1])

        return features, aligned_labels
