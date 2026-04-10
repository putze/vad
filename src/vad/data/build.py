from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from vad.data.datasets.base import BaseVADDataset
from vad.data.datasets.librivad import LibriVADDataset
from vad.data.datasets.processed import ProcessedVADDataset
from vad.data.preprocessing.preprocessing import VADPreprocessor


@dataclass(slots=True)
class LibriVADConfig:
    """
    Configuration for building a LibriVAD dataset.
    """

    results_root: str | Path
    labels_root: str | Path
    datasets: Sequence[str] | None = None
    splits: Sequence[str] | None = None
    extensions: tuple[str, ...] = (".wav",)


# Alias kept for future extensibility. At present, only LibriVAD is supported.
DatasetConfig = LibriVADConfig
DatasetBuilder = Callable[[DatasetConfig], BaseVADDataset]


def _build_librivad(config: LibriVADConfig) -> BaseVADDataset:
    """
    Build a raw LibriVAD dataset from configuration.

    Args:
        config: Configuration describing where to find audio and label files.

    Returns:
        A ``LibriVADDataset`` instance.
    """
    return LibriVADDataset(
        results_root=config.results_root,
        labels_root=config.labels_root,
        datasets=config.datasets,
        splits=config.splits,
        extensions=config.extensions,
    )


DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "librivad": _build_librivad,
}


def get_dataset_builder(dataset_name: str) -> DatasetBuilder:
    """
    Return the dataset builder registered for a given name.

    Dataset names are matched case-insensitively.

    Args:
        dataset_name: Name of the dataset builder to retrieve.

    Returns:
        Builder function associated with the dataset name.

    Raises:
        ValueError: If no builder is registered for the requested name.
    """
    key = dataset_name.lower()

    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")

    return DATASET_REGISTRY[key]


def build_raw_dataset(
    dataset_name: str,
    config: DatasetConfig,
) -> BaseVADDataset:
    """
    Build a raw dataset from its name and configuration.

    Args:
        dataset_name: Registered dataset name.
        config: Dataset configuration object.

    Returns:
        Raw dataset yielding waveform, sample-level labels, and sample rate.
    """
    builder = get_dataset_builder(dataset_name)
    return builder(config)


def build_processed_dataset(
    dataset_name: str,
    config: DatasetConfig,
    processor: VADPreprocessor,
) -> ProcessedVADDataset:
    """
    Build a processed dataset from its name and configuration.

    The returned dataset is a lazy wrapper around the raw dataset: preprocessing
    is applied in ``__getitem__`` rather than precomputed in advance.

    Args:
        dataset_name: Registered dataset name.
        config: Dataset configuration object.
        processor: Preprocessing pipeline applied to each raw sample.

    Returns:
        Processed dataset yielding model-ready features and frame-level labels.
    """
    raw_dataset = build_raw_dataset(
        dataset_name=dataset_name,
        config=config,
    )
    return ProcessedVADDataset(
        base_dataset=raw_dataset,
        processor=processor,
    )


def build_raw_datasets(
    dataset_name: str,
    train_config: DatasetConfig,
    val_config: DatasetConfig,
    test_config: DatasetConfig,
) -> tuple[BaseVADDataset, BaseVADDataset, BaseVADDataset]:
    """
    Build raw train, validation, and test datasets.

    Args:
        dataset_name: Registered dataset name.
        train_config: Configuration for the training split.
        val_config: Configuration for the validation split.
        test_config: Configuration for the test split.

    Returns:
        Tuple ``(train_dataset, val_dataset, test_dataset)`` of raw datasets.
    """
    return (
        build_raw_dataset(dataset_name, train_config),
        build_raw_dataset(dataset_name, val_config),
        build_raw_dataset(dataset_name, test_config),
    )


def build_processed_datasets(
    dataset_name: str,
    train_config: DatasetConfig,
    val_config: DatasetConfig,
    test_config: DatasetConfig,
    processor: VADPreprocessor,
) -> tuple[ProcessedVADDataset, ProcessedVADDataset, ProcessedVADDataset]:
    """
    Build processed train, validation, and test datasets.

    Args:
        dataset_name: Registered dataset name.
        train_config: Configuration for the training split.
        val_config: Configuration for the validation split.
        test_config: Configuration for the test split.
        processor: Preprocessing pipeline applied to each raw sample.

    Returns:
        Tuple ``(train_dataset, val_dataset, test_dataset)`` of processed datasets.
    """
    return (
        build_processed_dataset(dataset_name, train_config, processor),
        build_processed_dataset(dataset_name, val_config, processor),
        build_processed_dataset(dataset_name, test_config, processor),
    )
