from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from vad.data.collate import pad_collate_fn
from vad.data.datasets.processed import ProcessedVADDataset
from vad.data.preprocessing import VADPreprocessor


@dataclass(slots=True)
class DataLoaderConfig:
    """
    Configuration for PyTorch DataLoader creation.

    Attributes:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes used for data loading.
        pin_memory: Whether to pin host memory for faster device transfer.
        train_shuffle: Whether to shuffle the training dataset.
        drop_last_train: Whether to drop the last incomplete training batch.
        persistent_workers: Whether worker processes stay alive across epochs.
    """

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    train_shuffle: bool = True
    drop_last_train: bool = False
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        """Validate DataLoader configuration values."""
        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"`num_workers` must be non-negative, got {self.num_workers}")


def build_dataloader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    config: DataLoaderConfig,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for an already processed dataset.

    The dataset is expected to yield processed samples compatible with
    ``pad_collate_fn``.

    Args:
        dataset: Dataset yielding processed samples, typically
            ``(features, labels)``.
        config: DataLoader settings.
        shuffle: Whether to shuffle dataset items.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Configured ``DataLoader`` instance.
    """
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=pad_collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=drop_last,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
    )


def build_processed_dataloader(
    raw_dataset: Dataset[tuple[Tensor, Tensor, int]],
    processor: VADPreprocessor,
    config: DataLoaderConfig,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader from a raw dataset by wrapping it in ``ProcessedVADDataset``.

    Preprocessing is applied in the wrapped dataset's ``__getitem__``.

    Args:
        raw_dataset: Dataset yielding raw samples as
            ``(waveform, labels, sample_rate)``.
        processor: Preprocessing pipeline applied to each raw sample.
        config: DataLoader configuration.
        shuffle: Whether to shuffle dataset items.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Configured ``DataLoader`` instance over processed samples.
    """
    dataset = ProcessedVADDataset(
        base_dataset=raw_dataset,
        processor=processor,
    )
    return build_dataloader(
        dataset=dataset,
        config=config,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def build_train_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build the training DataLoader from an already processed dataset.

    Args:
        dataset: Processed training dataset.
        config: DataLoader configuration.

    Returns:
        Training ``DataLoader``.
    """
    return build_dataloader(
        dataset=dataset,
        config=config,
        shuffle=config.train_shuffle,
        drop_last=config.drop_last_train,
    )


def build_eval_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build a validation or test DataLoader from an already processed dataset.

    Args:
        dataset: Processed validation or test dataset.
        config: DataLoader configuration.

    Returns:
        Evaluation ``DataLoader``.
    """
    return build_dataloader(
        dataset=dataset,
        config=config,
        shuffle=False,
        drop_last=False,
    )


def build_train_processed_loader(
    raw_dataset: Dataset[tuple[Tensor, Tensor, int]],
    processor: VADPreprocessor,
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build the training DataLoader directly from a raw dataset.

    Args:
        raw_dataset: Raw training dataset.
        processor: Preprocessing pipeline applied per sample.
        config: DataLoader configuration.

    Returns:
        Training ``DataLoader`` over processed samples.
    """
    return build_processed_dataloader(
        raw_dataset=raw_dataset,
        processor=processor,
        config=config,
        shuffle=config.train_shuffle,
        drop_last=config.drop_last_train,
    )


def build_eval_processed_loader(
    raw_dataset: Dataset[tuple[Tensor, Tensor, int]],
    processor: VADPreprocessor,
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build a validation or test DataLoader directly from a raw dataset.

    Args:
        raw_dataset: Raw validation or test dataset.
        processor: Preprocessing pipeline applied per sample.
        config: DataLoader configuration.

    Returns:
        Evaluation ``DataLoader`` over processed samples.
    """
    return build_processed_dataloader(
        raw_dataset=raw_dataset,
        processor=processor,
        config=config,
        shuffle=False,
        drop_last=False,
    )


def build_dataloaders(
    train_dataset: Dataset[tuple[Tensor, Tensor]],
    val_dataset: Dataset[tuple[Tensor, Tensor]],
    test_dataset: Dataset[tuple[Tensor, Tensor]],
    config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test loaders from processed datasets.

    Args:
        train_dataset: Processed training dataset.
        val_dataset: Processed validation dataset.
        test_dataset: Processed test dataset.
        config: DataLoader configuration.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)``.
    """
    return (
        build_train_loader(train_dataset, config),
        build_eval_loader(val_dataset, config),
        build_eval_loader(test_dataset, config),
    )


def build_processed_dataloaders(
    train_raw_dataset: Dataset[tuple[Tensor, Tensor, int]],
    val_raw_dataset: Dataset[tuple[Tensor, Tensor, int]],
    test_raw_dataset: Dataset[tuple[Tensor, Tensor, int]],
    processor: VADPreprocessor,
    config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test loaders directly from raw datasets.

    Args:
        train_raw_dataset: Raw training dataset.
        val_raw_dataset: Raw validation dataset.
        test_raw_dataset: Raw test dataset.
        processor: Preprocessing pipeline applied per sample.
        config: DataLoader configuration.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)``.
    """
    return (
        build_train_processed_loader(train_raw_dataset, processor, config),
        build_eval_processed_loader(val_raw_dataset, processor, config),
        build_eval_processed_loader(test_raw_dataset, processor, config),
    )
