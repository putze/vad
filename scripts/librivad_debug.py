from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

from src.vad.data.datasets.librivad import LibriVADDataset
from src.vad.data.file_utils import load_audio
from src.vad.data.preprocessing import (
    AudioPreprocessor,
    LabelAligner,
    LogMelFeatureExtractor,
    VADPreprocessor,
)
from src.vad.visualization import (
    debug_plot_alignment,
    debug_plot_features_with_label_overlay,
    debug_plot_features_with_labels,
    debug_plot_waveform_with_labels,
)


@dataclass
class DebugConfig:
    results_root: str | Path
    labels_root: str | Path
    datasets: list[str]
    splits: list[str]
    sample_index: int = 0
    target_sample_rate: int = 16000
    frame_length: int = 400
    hop_length: int = 160
    n_fft: int = 400
    n_mels: int = 40
    normalize: bool = True
    center: bool = True
    alignment_start_frame: int = 0


class LibriVADDebugger:
    """
    Helper class for loading a raw LibriVAD sample, preprocessing it, and
    visualizing the result with debugging plots.
    """

    def __init__(self, config: DebugConfig) -> None:
        self.config = config

        self.dataset = LibriVADDataset(
            results_root=config.results_root,
            labels_root=config.labels_root,
            datasets=config.datasets,
            splits=config.splits,
        )

        self.audio_preprocessor = AudioPreprocessor(
            target_sample_rate=config.target_sample_rate,
            normalize=config.normalize,
        )

        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=config.target_sample_rate,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
        )

        self.label_aligner = LabelAligner(
            hop_length=config.hop_length,
            frame_length=config.frame_length,
            center=config.center,
        )

        self.preprocessor = VADPreprocessor(
            audio_preprocessor=self.audio_preprocessor,
            feature_extractor=self.feature_extractor,
            label_aligner=self.label_aligner,
        )

    def load_raw_sample(self, index: int | None = None) -> tuple[Tensor, Tensor, int]:
        """
        Load one raw sample and return waveform, labels, and original sample rate.
        """
        idx = self.config.sample_index if index is None else index
        sample = self.dataset.samples[idx]

        waveform, sample_rate = load_audio(sample.audio_path)
        labels = self.dataset._load_labels(sample.label_path)

        return waveform, labels, sample_rate

    def preprocess_sample(
        self,
        index: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        Load one raw sample and preprocess it.

        Returns:
            waveform: raw waveform [T]
            labels: raw sample-level labels [T]
            features: extracted features [F, T_frames]
            aligned_labels: frame-level labels [T_frames]
            sample_rate: original sample rate
        """
        waveform, labels, sample_rate = self.load_raw_sample(index=index)
        features, aligned_labels = self.preprocessor(waveform, labels, sample_rate)
        return waveform, labels, features, aligned_labels, sample_rate

    def print_sample_summary(self, index: int | None = None) -> None:
        """
        Print a compact summary of the selected sample.
        """
        idx = self.config.sample_index if index is None else index
        sample = self.dataset.samples[idx]

        waveform, labels, features, aligned_labels, sample_rate = self.preprocess_sample(index=idx)

        print(f"Dataset size: {len(self.dataset)}")
        print(f"Sample index: {idx}")
        print(f"Audio path: {sample.audio_path}")
        print(f"Label path: {sample.label_path}")
        print(f"Original sample rate: {sample_rate}")
        print(f"Raw waveform shape: {tuple(waveform.shape)}")
        print(f"Raw labels shape: {tuple(labels.shape)}")
        print(f"Features shape: {tuple(features.shape)}")
        print(f"Aligned labels shape: {tuple(aligned_labels.shape)}")
        print(f"Raw label unique values: {labels.unique(sorted=True)}")
        print(f"Aligned label unique values: {aligned_labels.unique(sorted=True)}")
        print(f"Raw positive ratio: {labels.float().mean().item():.4f}")
        print(f"Frame positive ratio: {aligned_labels.float().mean().item():.4f}")

    def plot_waveform_and_raw_labels(self, index: int | None = None) -> None:
        """
        Plot the raw waveform with raw sample-level labels.
        """
        waveform, labels, sample_rate = self.load_raw_sample(index=index)
        debug_plot_waveform_with_labels(waveform, labels, sample_rate)

    def plot_features_and_aligned_labels(self, index: int | None = None) -> None:
        """
        Plot extracted features and aligned frame-level labels.
        """
        _, _, features, aligned_labels, _ = self.preprocess_sample(index=index)
        debug_plot_features_with_labels(features, aligned_labels)

    def plot_features_with_overlay(self, index: int | None = None) -> None:
        """
        Plot extracted features with label overlay.
        """
        _, _, features, aligned_labels, _ = self.preprocess_sample(index=index)
        debug_plot_features_with_label_overlay(features, aligned_labels)

    def plot_alignment(self, index: int | None = None) -> None:
        """
        Plot the sample-to-frame label alignment.
        """
        waveform, labels, features, aligned_labels, _ = self.preprocess_sample(index=index)

        debug_plot_alignment(
            sample_labels=labels,
            num_frames=features.shape[-1],
            hop_length=self.feature_extractor.hop_length,
            frame_length=self.feature_extractor.frame_length,
            start_frame=self.config.alignment_start_frame,
            aligner_output=aligned_labels,
        )

    def run_all(self, index: int | None = None) -> None:
        """
        Run summary and all debugging visualizations for one sample.
        """
        self.print_sample_summary(index=index)
        self.plot_waveform_and_raw_labels(index=index)
        self.plot_features_and_aligned_labels(index=index)
        self.plot_features_with_overlay(index=index)
        self.plot_alignment(index=index)


if __name__ == "__main__":
    config = DebugConfig(
        results_root="/Users/antje/Blynt/LibriVAD/Results",
        labels_root="/Users/antje/Blynt/LibriVAD/Files/Labels",
        datasets=["LibriSpeech"],
        splits=["train-clean-100"],
        sample_index=0,
        target_sample_rate=16000,
        frame_length=400,
        hop_length=160,
        n_fft=400,
        n_mels=40,
        normalize=True,
        center=True,
        alignment_start_frame=35,
    )

    debugger = LibriVADDebugger(config)
    debugger.run_all()
