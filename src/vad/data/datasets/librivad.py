from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Sequence

from src.vad.data.base import BaseVADDataset
from src.vad.data.samples import AudioSample
from src.vad.data.utils.file_utils import iter_audio_files


class LibriVADDataset(BaseVADDataset[AudioSample]):
    """
    Dataset for LibriVAD audio files and their corresponding label files.

    Supports multiple LibriVAD source datasets and splits, and builds
    matched audio-label sample pairs.
    """

    def __init__(
        self,
        results_root: str | Path,
        labels_root: str | Path,
        datasets: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        extensions: tuple[str, ...] = (".wav",),
    ) -> None:
        """
        Args:
            results_root (str): Root directory containing noisy audio files.
            labels_root (str): Root directory containing label files.
            datasets (Sequence[str] | None): Dataset names to include.
            splits (Sequence[str] | None): Split names to include.
            extensions (tuple[str, ...]): Allowed audio file extensions.
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

        super().__init__(samples=samples)

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

                for audio_path in iter_audio_files(
                    audio_split_dir,
                    extensions=self.extensions,
                ):
                    label_path = self._audio_to_label_path(
                        audio_path=audio_path,
                        dataset_name=dataset_name,
                        split_name=split_name,
                    )

                    if label_path.exists():
                        samples.append(
                            AudioSample(
                                audio_path=audio_path,
                                label_path=label_path,
                            )
                        )
                    else:
                        missing_labels += 1

        if missing_labels > 0:
            warnings.warn(
                f"Skipped {missing_labels} audio files without labels. "
                f"Using {len(samples)} matched pairs.",
                stacklevel=2,
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
