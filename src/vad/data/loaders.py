from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from src.vad.data.collate import pad_collate_fn
from src.vad.data.datasets.processed import ProcessedVADDataset
from src.vad.data.preprocessing import VADPreprocessor


@dataclass(slots=True)
class DataLoaderConfig:
    """
    Configuration for DataLoader creation.
    """

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    train_shuffle: bool = True
    drop_last_train: bool = False
    persistent_workers: bool = False


def build_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for a prepared dataset.

    Args:
        dataset (Dataset): Dataset returning processed samples.
        config (DataLoaderConfig): DataLoader settings.
        shuffle (bool): Whether to shuffle samples.
        drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
        Configured DataLoader instance.
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
    raw_dataset: Dataset,
    processor: VADPreprocessor,
    config: DataLoaderConfig,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader from a raw dataset by wrapping it in ProcessedVADDataset.

    Args:
        raw_dataset (Dataset): Dataset returning (waveform, labels, sample_rate).
        processor (VADPreprocessor): Preprocessor applied on each sample.
        config (DataLoaderConfig): DataLoader configuration.
        shuffle (bool): Whether to shuffle samples.
        drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
        Configured DataLoader.
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
    dataset: Dataset,
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build the training DataLoader from an already prepared dataset.
    """
    return build_dataloader(
        dataset=dataset,
        config=config,
        shuffle=config.train_shuffle,
        drop_last=config.drop_last_train,
    )


def build_eval_loader(
    dataset: Dataset,
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build a validation or test DataLoader from an already prepared dataset.
    """
    return build_dataloader(
        dataset=dataset,
        config=config,
        shuffle=False,
        drop_last=False,
    )


def build_train_processed_loader(
    raw_dataset: Dataset,
    processor: VADPreprocessor,
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build the training DataLoader directly from a raw dataset.
    """
    return build_processed_dataloader(
        raw_dataset=raw_dataset,
        processor=processor,
        config=config,
        shuffle=config.train_shuffle,
        drop_last=config.drop_last_train,
    )


def build_eval_processed_loader(
    raw_dataset: Dataset,
    processor: VADPreprocessor,
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build a validation or test DataLoader directly from a raw dataset.
    """
    return build_processed_dataloader(
        raw_dataset=raw_dataset,
        processor=processor,
        config=config,
        shuffle=False,
        drop_last=False,
    )


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test loaders from prepared datasets.
    """
    return (
        build_train_loader(train_dataset, config),
        build_eval_loader(val_dataset, config),
        build_eval_loader(test_dataset, config),
    )


def build_processed_dataloaders(
    train_raw_dataset: Dataset,
    val_raw_dataset: Dataset,
    test_raw_dataset: Dataset,
    processor: VADPreprocessor,
    config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test loaders directly from raw datasets.
    """
    return (
        build_train_processed_loader(train_raw_dataset, processor, config),
        build_eval_processed_loader(val_raw_dataset, processor, config),
        build_eval_processed_loader(test_raw_dataset, processor, config),
    )
