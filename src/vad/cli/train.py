from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from vad.data import (
    DataLoaderConfig,
    DatasetConfig,
    build_dataloaders,
    build_processed_datasets,
)
from vad.data.preprocessing import (
    LabelAligner,
    LogMelFeatureExtractor,
    VADPreprocessor,
    WaveformPreprocessor,
)
from vad.models.causal_vad import CausalVAD
from vad.training.loops import train_model


def build_preprocessor() -> VADPreprocessor:
    """
    Build the preprocessing pipeline used for VAD training.
    """
    waveform_preprocessor = WaveformPreprocessor(
        target_sample_rate=16000,
    )

    feature_extractor = LogMelFeatureExtractor(
        sample_rate=16000,
        n_mels=40,
        n_fft=400,
        hop_length=160,
        frame_length=400,
        center=False,
    )

    label_aligner = LabelAligner(
        hop_length=160,
        frame_length=400,
        center=False,
    )

    return VADPreprocessor(
        waveform_preprocessor=waveform_preprocessor,
        feature_extractor=feature_extractor,
        label_aligner=label_aligner,
    )


def build_dataset_configs(
    results_root: Path,
    labels_root: Path,
) -> tuple[DatasetConfig, DatasetConfig]:
    """
    Build train and validation dataset configs.
    """
    train_config = DatasetConfig(
        results_root=results_root,
        labels_root=labels_root,
        datasets=("LibriSpeech",),
        splits=("train-clean-100",),
    )

    val_config = DatasetConfig(
        results_root=results_root,
        labels_root=labels_root,
        datasets=("LibriSpeech",),
        splits=("dev-clean",),
    )

    return train_config, val_config


def build_loaders(
    processor: VADPreprocessor,
    results_root: Path,
    labels_root: Path,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and validation dataloaders.

    Since the current dataset builder requires a test config/dataset,
    the validation split is temporarily reused as a placeholder and ignored.
    """
    train_config, val_config = build_dataset_configs(
        results_root=results_root,
        labels_root=labels_root,
    )

    train_dataset, val_dataset, _ = build_processed_datasets(
        dataset_name="librivad",
        train_config=train_config,
        val_config=val_config,
        test_config=val_config,
        processor=processor,
    )

    loader_config = DataLoaderConfig(
        batch_size=16,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        train_shuffle=True,
        drop_last_train=False,
        persistent_workers=False,
    )

    train_loader, val_loader, _ = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset,
        config=loader_config,
    )

    return train_loader, val_loader


def build_model(device: torch.device) -> CausalVAD:
    """
    Build the VAD model.

    Note:
        The training loop expects model outputs shaped [B, T] or [B, 1, T].
        If your current model returns [B, 2, T], it must be adapted for binary
        BCE-with-logits training.
    """
    model = CausalVAD(n_mels=40)
    return cast(CausalVAD, model.to(device))


def train() -> None:
    """
    Build all training components and launch training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_root = Path("/Users/antje/Blynt/LibriVAD/Results")
    labels_root = Path("/Users/antje/Blynt/LibriVAD/Files/Labels")

    log_dir = Path("runs")
    experiment_name = "causal_conv"
    checkpoint_path = Path("checkpoints")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    processor = build_preprocessor()
    train_loader, val_loader = build_loaders(
        processor=processor,
        results_root=results_root,
        labels_root=labels_root,
    )

    model = build_model(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        log_dir=log_dir,
        experiment_name=experiment_name,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    train()
