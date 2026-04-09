from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

from vad.data.datasets.base import BaseVADDataset
from vad.data.datasets.samples import AudioExample
from vad.data.file_utils import iter_audio_files, match_audio_label_pairs


class LibriVADDataset(BaseVADDataset[AudioExample]):
    """
    Dataset indexer for LibriVAD audio files and sample-level label files.

    This class scans selected LibriVAD datasets and splits, matches each audio
    file to its corresponding ``.npy`` label file, and builds a list of
    ``AudioSample`` objects for the base dataset.

    Expected directory structure:

    - audio files under:
      ``results_root / dataset_name / split_name / ... / <utt_id>.wav``
    - label files under:
      ``labels_root / dataset_name / split_name / <speaker_id> / <chapter_id> / <utt_id>.npy``

    where ``<utt_id>`` follows the LibriSpeech-style naming convention
    ``speaker-chapter-utterance`` such as ``61-70968-0019``.

    Missing dataset or split directories are ignored. Audio files without a
    matching label file are skipped and counted.
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
        Initialize the LibriVAD dataset index.

        Args:
            results_root: Root directory containing audio files.
            labels_root: Root directory containing label files.
            datasets: Dataset names to include. If omitted, a default subset of
                LibriVAD source datasets is used.
            splits: Split names to include. If omitted, default train/dev/test
                splits are used.
            extensions: Allowed audio file extensions, matched case-insensitively.
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

    def _build_samples(self) -> list[AudioExample]:
        """
        Build the list of valid audio/label sample pairs.

        The method scans all configured dataset/split directories, derives the
        expected label path for each discovered audio file, and keeps only pairs
        for which the label file exists.

        Returns:
            List of matched ``AudioSample`` objects.

        Raises:
            ValueError: If no valid audio/label pairs are found.
        """
        samples: list[AudioExample] = []
        missing_labels = 0

        for dataset_name in self.datasets:
            for split_name in self.splits:
                audio_split_dir = self.results_root / dataset_name / split_name
                label_split_dir = self.labels_root / dataset_name / split_name

                if not audio_split_dir.exists() or not audio_split_dir.is_dir():
                    continue

                if not label_split_dir.exists() or not label_split_dir.is_dir():
                    continue

                audio_files = iter_audio_files(
                    audio_split_dir,
                    extensions=self.extensions,
                )

                pairs, missing = match_audio_label_pairs(
                    audio_files,
                    map_fn=lambda audio_path, dn=dataset_name, sn=split_name: (
                        self._audio_to_label_path(
                            audio_path=audio_path,
                            dataset_name=dn,
                            split_name=sn,
                        )
                    ),
                )
                missing_labels += missing

                samples.extend(
                    AudioExample(audio_path=audio_path, label_path=label_path)
                    for audio_path, label_path in pairs
                )

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
        Derive the expected label path for a LibriVAD audio file.

        The label path is reconstructed from the LibriSpeech-style utterance ID
        in the audio filename. For an audio stem such as ``61-70968-0019``, the
        label is expected at:

        ``labels_root / dataset_name / split_name / 61 / 70968 / 61-70968-0019.npy``

        Args:
            audio_path: Path to the audio file.
            dataset_name: Dataset name used in the directory hierarchy.
            split_name: Split name used in the directory hierarchy.

        Returns:
            Expected label file path.

        Raises:
            ValueError: If the audio filename does not follow the expected
                LibriSpeech-style ``speaker-chapter-utterance`` pattern.
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
