from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import src.vad.data.preprocessing
from src.vad.data.build import LibriVADConfig, build_processed_datasets, build_raw_datasets
from src.vad.data.datasets import BaseVADDataset, ProcessedVADDataset
from src.vad.data.loaders import DataLoaderConfig, build_dataloaders


@dataclass(slots=True)
class DebugConfig:
    """
    Configuration for dataset and dataloader debugging.
    """

    dataset_name: str = "librivad"

    results_root: Path = Path("/Users/antje/Blynt/LibriVAD/Results")
    labels_root: Path = Path("/Users/antje/Blynt/LibriVAD/Files/Labels")
    datasets: tuple[str, ...] = ("LibriSpeech",)

    train_splits: tuple[str, ...] = ("train-clean-100",)
    val_splits: tuple[str, ...] = ("dev-clean",)
    test_splits: tuple[str, ...] = ("test-clean",)

    target_sample_rate: int = 16000
    frame_length: int = 400
    hop_length: int = 160
    n_fft: int = 400
    n_mels: int = 40
    normalize: bool = True

    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    train_shuffle: bool = False

    num_train_batches_to_check: int = 3
    num_eval_batches_to_check: int = 2


def describe_tensor(name: str, x: Tensor) -> None:
    """
    Print basic information about a tensor.
    """
    print(f"{name}:")
    print(f"  shape       : {tuple(x.shape)}")
    print(f"  dtype       : {x.dtype}")
    print(f"  device      : {x.device}")
    if x.numel() > 0:
        print(f"  min         : {x.min().item()}")
        print(f"  max         : {x.max().item()}")
    print()


def inspect_raw_dataset(dataset_name: str, dataset: BaseVADDataset) -> None:
    """
    Inspect one sample from a raw dataset.
    """
    print("=" * 80)
    print(f"RAW DATASET INSPECTION: {dataset_name}")
    print("=" * 80)
    print(f"dataset type : {type(dataset).__name__}")
    print(f"dataset size : {len(dataset)}")
    print()

    waveform, labels, sample_rate = dataset[0]

    print("Single raw sample:")
    describe_tensor("waveform", waveform)
    describe_tensor("labels", labels)
    print(f"sample_rate  : {sample_rate}")
    print(f"label uniques: {torch.unique(labels)}")
    print(f"label sum    : {labels.sum().item()}")
    print()


def inspect_processed_dataset(dataset_name: str, dataset: ProcessedVADDataset) -> None:
    """
    Inspect one sample from a processed dataset.
    """
    print("=" * 80)
    print(f"PROCESSED DATASET INSPECTION: {dataset_name}")
    print("=" * 80)
    print(f"dataset type : {type(dataset).__name__}")
    print(f"dataset size : {len(dataset)}")
    print()

    features, labels = dataset[0]

    print("Single processed sample:")
    describe_tensor("features", features)
    describe_tensor("labels", labels)

    print(f"feature ndim : {features.ndim}")
    print(f"label ndim   : {labels.ndim}")
    print(f"label uniques: {torch.unique(labels)}")
    print(f"num frames   : {features.shape[1]}")
    print(f"label length : {labels.shape[0]}")
    print()

    if features.ndim != 2:
        raise ValueError(f"Expected features [n_mels, T], got {tuple(features.shape)}")

    if labels.ndim != 1:
        raise ValueError(f"Expected labels [T], got {tuple(labels.shape)}")

    if features.shape[1] != labels.shape[0]:
        raise ValueError(
            f"Frame mismatch: features have {features.shape[1]} frames "
            f"but labels have length {labels.shape[0]}"
        )


def inspect_batch(loader_name: str, loader: DataLoader) -> None:
    """
    Inspect one batch from a DataLoader.
    """
    print("=" * 80)
    print(f"DATALOADER INSPECTION: {loader_name}")
    print("=" * 80)

    x, y, lengths = next(iter(loader))

    print("Single batch:")
    describe_tensor("x", x)
    describe_tensor("y", y)
    describe_tensor("lengths", lengths)

    print("x expected format : [B, n_mels, T_max]")
    print("y expected format : [B, T_max]")
    print("lengths format    : [B]")
    print()

    print(f"batch size        : {x.shape[0]}")
    print(f"n_mels            : {x.shape[1]}")
    print(f"T_max             : {x.shape[2]}")
    print(f"y shape matches x : {y.shape[0] == x.shape[0] and y.shape[1] == x.shape[2]}")
    print()

    if x.ndim != 3:
        raise ValueError(f"Expected x [B, n_mels, T], got {tuple(x.shape)}")

    if y.ndim != 2:
        raise ValueError(f"Expected y [B, T], got {tuple(y.shape)}")

    if lengths.ndim != 1:
        raise ValueError(f"Expected lengths [B], got {tuple(lengths.shape)}")

    if x.shape[0] != y.shape[0]:
        raise ValueError("Batch size mismatch between x and y")

    if x.shape[2] != y.shape[1]:
        raise ValueError("Time dimension mismatch between x and y")

    if x.shape[0] != lengths.shape[0]:
        raise ValueError("Batch size mismatch between tensors and lengths")

    print("Per-item checks:")
    for i in range(x.shape[0]):
        valid_len = int(lengths[i].item())
        padded_part = y[i, valid_len:]

        print(
            f"  sample {i:02d} | valid_len={valid_len:4d} "
            f"| valid_label_uniques={torch.unique(y[i, :valid_len]).tolist()}"
        )

        if valid_len > x.shape[2]:
            raise ValueError(f"Invalid length for sample {i}: {valid_len} > padded T {x.shape[2]}")

        if padded_part.numel() > 0:
            print(f"            padded_label_uniques={torch.unique(padded_part).tolist()}")

    print()


def inspect_multiple_batches(
    loader_name: str,
    loader: DataLoader,
    num_batches: int,
) -> None:
    """
    Print shape summaries for multiple batches.
    """
    print("=" * 80)
    print(f"MULTI-BATCH CHECK: {loader_name}")
    print("=" * 80)

    for batch_idx, (x, y, lengths) in enumerate(loader):
        print(
            f"batch {batch_idx:02d} | "
            f"x={tuple(x.shape)} | "
            f"y={tuple(y.shape)} | "
            f"lengths={tuple(lengths.shape)} | "
            f"min_len={int(lengths.min().item())} | "
            f"max_len={int(lengths.max().item())}"
        )

        if batch_idx + 1 >= num_batches:
            break

    print()


def build_dataset_configs(
    config: DebugConfig,
) -> tuple[LibriVADConfig, LibriVADConfig, LibriVADConfig]:
    """
    Build train, validation, and test dataset configs.
    """
    common: dict[str, Any] = {
        "results_root": config.results_root,
        "labels_root": config.labels_root,
        "datasets": config.datasets,
    }

    train_config = LibriVADConfig(
        **common,
        splits=config.train_splits,
    )
    val_config = LibriVADConfig(
        **common,
        splits=config.val_splits,
    )
    test_config = LibriVADConfig(
        **common,
        splits=config.test_splits,
    )

    return train_config, val_config, test_config


def build_processor(config: DebugConfig) -> src.vad.data.preprocessing.VADPreprocessor:
    """
    Build the preprocessing pipeline used for debugging.
    """
    audio_preprocessor = src.vad.data.preprocessing.AudioPreprocessor(
        target_sample_rate=config.target_sample_rate,
        normalize=config.normalize,
    )

    feature_extractor = src.vad.data.preprocessing.LogMelFeatureExtractor(
        sample_rate=config.target_sample_rate,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
    )

    label_aligner = src.vad.data.preprocessing.LabelAligner(
        hop_length=feature_extractor.hop_length,
        frame_length=feature_extractor.frame_length,
        center=feature_extractor.center,
    )

    return src.vad.data.preprocessing.VADPreprocessor(
        audio_preprocessor=audio_preprocessor,
        feature_extractor=feature_extractor,
        label_aligner=label_aligner,
    )


def run_raw_dataset_checks(
    train_raw: BaseVADDataset,
    val_raw: BaseVADDataset,
    test_raw: BaseVADDataset,
) -> None:
    """
    Run inspection checks on raw datasets.
    """
    print("\nBuilding raw datasets...\n")
    inspect_raw_dataset("train_raw", train_raw)
    inspect_raw_dataset("val_raw", val_raw)
    inspect_raw_dataset("test_raw", test_raw)


def run_processed_dataset_checks(
    train_dataset: ProcessedVADDataset,
    val_dataset: ProcessedVADDataset,
    test_dataset: ProcessedVADDataset,
) -> None:
    """
    Run inspection checks on processed datasets.
    """
    print("\nBuilding processed datasets...\n")
    inspect_processed_dataset("train", train_dataset)
    inspect_processed_dataset("val", val_dataset)
    inspect_processed_dataset("test", test_dataset)


def run_loader_checks(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: DebugConfig,
) -> None:
    """
    Run inspection checks on DataLoaders.
    """
    print("\nBuilding dataloaders...\n")
    inspect_batch("train_loader", train_loader)
    inspect_batch("val_loader", val_loader)
    inspect_batch("test_loader", test_loader)

    inspect_multiple_batches(
        "train_loader",
        train_loader,
        num_batches=config.num_train_batches_to_check,
    )
    inspect_multiple_batches(
        "val_loader",
        val_loader,
        num_batches=config.num_eval_batches_to_check,
    )
    inspect_multiple_batches(
        "test_loader",
        test_loader,
        num_batches=config.num_eval_batches_to_check,
    )


def main() -> None:
    """
    Run end-to-end dataset and dataloader debugging checks.
    """
    config = DebugConfig()

    train_config, val_config, test_config = build_dataset_configs(config)
    processor = build_processor(config)

    loader_config = DataLoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        train_shuffle=config.train_shuffle,
    )

    train_raw, val_raw, test_raw = build_raw_datasets(
        dataset_name=config.dataset_name,
        train_config=train_config,
        val_config=val_config,
        test_config=test_config,
    )
    run_raw_dataset_checks(train_raw, val_raw, test_raw)

    train_dataset, val_dataset, test_dataset = build_processed_datasets(
        dataset_name=config.dataset_name,
        train_config=train_config,
        val_config=val_config,
        test_config=test_config,
        processor=processor,
    )
    run_processed_dataset_checks(train_dataset, val_dataset, test_dataset)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=loader_config,
    )
    run_loader_checks(train_loader, val_loader, test_loader, config)

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
