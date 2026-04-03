from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.vad.data.base import BaseVADDataset
from src.vad.data.preprocessing.preprocessing import VADPreprocessor
from src.vad.data.samples import AudioSample


class LibriVADDataset(BaseVADDataset):
    """
    Dataset for LibriVAD audio files and their corresponding label files.

    Supports multiple LibriVAD source datasets and splits, and builds
    matched audio-label sample pairs.
    """

    def __init__(
        self,
        results_root: str,
        labels_root: str,
        datasets: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        sample_rate: int = 16000,
        extensions: tuple[str, ...] = (".wav",),
        preprocessor: VADPreprocessor | None = None,
    ) -> None:
        """
        Args:
            results_root (str): Root directory containing noisy audio files.
            labels_root (str): Root directory containing label files.
            datasets (Sequence[str] | None): Dataset names to include.
            splits (Sequence[str] | None): Split names to include.
            sample_rate (int): Target sample rate.
            extensions (tuple[str, ...]): Allowed audio file extensions.
            preprocessor (VADPreprocessor | None): Optional preprocessing pipeline.
        """
        self.results_root = Path(results_root)
        self.labels_root = Path(labels_root)
        self.datasets = (
            list(datasets)
            if datasets is not None
            else [
                "LibriSpeech",
                "LibriSpeechConcat",
            ]
        )
        self.splits = (
            list(splits)
            if splits is not None
            else [
                "train-clean-100",
                "dev-clean",
                "test-clean",
            ]
        )

        self.extensions = tuple(ext.lower() for ext in extensions)
        samples = self._build_samples()

        super().__init__(
            samples=samples,
            sample_rate=sample_rate,
            extensions=self.extensions,
            preprocessor=preprocessor,
        )

    def _build_samples(self) -> List[AudioSample]:
        """
        Build the list of matched audio-label samples.

        Returns:
            List[AudioSample]: All valid audio/label pairs found.

        Raises:
            ValueError: If no matching pairs are found.
        """
        samples: List[AudioSample] = []
        missing_labels = 0

        for dataset_name in self.datasets:
            for split_name in self.splits:
                audio_split_dir = self.results_root / dataset_name / split_name
                label_split_dir = self.labels_root / dataset_name / split_name

                if not audio_split_dir.exists() or not audio_split_dir.is_dir():
                    continue

                if not label_split_dir.exists() or not label_split_dir.is_dir():
                    continue

                for audio_path in self._iter_audio_files(audio_split_dir):
                    label_path = self._audio_to_label_path(
                        audio_path=audio_path,
                        dataset_name=dataset_name,
                        split_name=split_name,
                    )

                    if label_path.exists():
                        samples.append(AudioSample(audio_path, label_path))
                    else:
                        missing_labels += 1

        if missing_labels > 0:
            print(
                f"Warning: skipped {missing_labels} audio files without labels. "
                f"Using {len(samples)} matched pairs."
            )

        if not samples:
            raise ValueError(
                "No matching LibriVAD audio/label pairs found.\n"
                f"results_root={self.results_root}\n"
                f"labels_root={self.labels_root}\n"
                f"datasets={self.datasets}\n"
                f"splits={self.splits}"
            )

        return samples

    def _audio_to_label_path(
        self,
        audio_path: Path,
        dataset_name: str,
        split_name: str,
    ) -> Path:
        """
        Map an audio file path to its corresponding label file path.

        Args:
            audio_path (Path): Path to the audio file.
            dataset_name (str): Dataset name.
            split_name (str): Split name.

        Returns:
            Path: Expected label file path.

        Raises:
            ValueError: If the audio filename format is invalid.
        """
        stem = audio_path.stem
        stem_parts = stem.split("-")

        if len(stem_parts) < 3:
            raise ValueError(f"Unexpected LibriVAD filename format: {audio_path.name}")

        speaker_id = stem_parts[0]
        chapter_id = stem_parts[1]

        return (
            self.labels_root / dataset_name / split_name / speaker_id / chapter_id / f"{stem}.npy"
        )


def pad_collate_fn(
    batch: List[Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
    waveforms, labels = zip(*batch)

    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())

    batch_size = len(waveforms)
    padded_waveforms = torch.zeros(batch_size, max_len, dtype=torch.float32)
    padded_labels = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, (waveform, label) in enumerate(zip(waveforms, labels)):
        length = waveform.shape[0]
        padded_waveforms[i, :length] = waveform
        padded_labels[i, :length] = label

    return padded_waveforms, padded_labels, lengths


def create_dataloader(
    dataset: Dataset[Tuple[Tensor, Tensor]],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[Tuple[Tensor, Tensor]]:
    """
    Create a DataLoader for a VAD dataset.

    Args:
        dataset (Dataset[Tuple[Tensor, Tensor]]): Dataset yielding waveform-label pairs.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes.

    Returns:
        DataLoader[Tuple[Tensor, Tensor]]: Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
    )


if __name__ == "__main__":
    import numpy as np

    from src.vad.data.preprocessing import (
        AudioPreprocessor,
        LabelAligner,
        LogMelFeatureExtractor,
        VADPreprocessor,
    )

    audio_preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        normalize=True,
    )

    feature_extractor = LogMelFeatureExtractor(
        sample_rate=16000,
        frame_length=400,
        hop_length=160,
        n_fft=400,
        n_mels=40,
    )

    label_aligner = LabelAligner(
        hop_length=feature_extractor.hop_length,
        frame_length=feature_extractor.frame_length,
        center=True,
    )

    preprocessor = VADPreprocessor(
        audio_preprocessor=audio_preprocessor,
        feature_extractor=feature_extractor,
        label_aligner=label_aligner,
    )

    dataset = LibriVADDataset(
        results_root="/Users/antje/Blynt/LibriVAD/Results",
        labels_root="/Users/antje/Blynt/LibriVAD/Files/Labels",
        datasets=["LibriSpeech"],
        splits=["train-clean-100"],
        sample_rate=16000,
        preprocessor=preprocessor,
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset.samples[0]
    labels = np.load(sample.label_path)

    features, labels = dataset[0]
    print("Single sample:")
    print("  features shape:", features.shape)  # [n_mels, frames]
    print("  labels shape:  ", labels.shape)  # [frames]

    waveform, sr = dataset._load_audio(dataset.samples[0].audio_path)

    duration = waveform.shape[0] / sr
    num_frames = 1409
    frame_duration = num_frames * (160 / sr)

    print("audio duration:", duration)
    print("feature duration:", frame_duration)
