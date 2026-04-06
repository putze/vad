from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.vad.data.datasets import BaseVADDataset, LibriVADDataset, ProcessedVADDataset
from src.vad.data.preprocessing import VADPreprocessor


@dataclass(slots=True)
class LibriVADConfig:
    """
    Configuration for LibriVAD dataset creation.
    """

    results_root: str | Path
    labels_root: str | Path
    datasets: Sequence[str] | None = None
    splits: Sequence[str] | None = None
    extensions: tuple[str, ...] = (".wav",)


DatasetConfig = LibriVADConfig
DatasetBuilder = Callable[[DatasetConfig], BaseVADDataset]


def _build_librivad(config: LibriVADConfig) -> BaseVADDataset:
    """
    Build a LibriVAD dataset from its configuration.
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
    Return the builder registered for a dataset name.
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
    Build a raw dataset from a name and configuration.
    """
    builder = get_dataset_builder(dataset_name)
    return builder(config)


def build_processed_dataset(
    dataset_name: str,
    config: DatasetConfig,
    processor: VADPreprocessor,
) -> ProcessedVADDataset:
    """
    Build a processed dataset from a name and configuration.
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
    """
    return (
        build_processed_dataset(dataset_name, train_config, processor),
        build_processed_dataset(dataset_name, val_config, processor),
        build_processed_dataset(dataset_name, test_config, processor),
    )
