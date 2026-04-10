from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

from vad.config import AudioConfig, TrainingConfig
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the causal VAD model.")

    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--frame-length-ms", type=float, default=25.0)
    parser.add_argument("--frame-shift-ms", type=float, default=10.0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--train-split", type=str, default="train-clean-100")
    parser.add_argument("--val-split", type=str, default="dev-clean")
    parser.add_argument("--experiment-name", type=str, default="causal_conv")

    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_preprocessor(audio_config: AudioConfig) -> VADPreprocessor:
    waveform_preprocessor = WaveformPreprocessor(
        target_sample_rate=audio_config.sample_rate,
    )

    feature_extractor = LogMelFeatureExtractor(
        sample_rate=audio_config.sample_rate,
        n_mels=audio_config.n_mels,
        n_fft=audio_config.frame_length_samples,
        hop_length=audio_config.hop_length_samples,
        frame_length=audio_config.frame_length_samples,
        center=False,
    )

    label_aligner = LabelAligner(
        hop_length=audio_config.hop_length_samples,
        frame_length=audio_config.frame_length_samples,
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
    training_config: TrainingConfig,
) -> tuple[DatasetConfig, DatasetConfig]:
    """
    Build train and validation dataset configs.
    """
    train_config = DatasetConfig(
        results_root=results_root,
        labels_root=labels_root,
        datasets=("LibriSpeech",),
        splits=(training_config.train_split,),
    )

    val_config = DatasetConfig(
        results_root=results_root,
        labels_root=labels_root,
        datasets=("LibriSpeech",),
        splits=(training_config.val_split,),
    )

    return train_config, val_config


def build_loaders(
    processor: VADPreprocessor,
    results_root: Path,
    labels_root: Path,
    training_config: TrainingConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and validation dataloaders.

    Since the current dataset builder requires a test config/dataset,
    the validation split is temporarily reused as a placeholder and ignored.
    """
    train_config, val_config = build_dataset_configs(
        results_root=results_root,
        labels_root=labels_root,
        training_config=training_config,
    )

    train_dataset, val_dataset, _ = build_processed_datasets(
        dataset_name="librivad",
        train_config=train_config,
        val_config=val_config,
        test_config=val_config,
        processor=processor,
    )

    loader_config = DataLoaderConfig(
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        train_shuffle=True,
        drop_last_train=False,
        persistent_workers=training_config.num_workers > 0,
    )

    train_loader, val_loader, _ = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset,
        config=loader_config,
    )
    return train_loader, val_loader


def build_model(device: torch.device, audio_config: AudioConfig) -> CausalVAD:
    model = CausalVAD(n_mels=audio_config.n_mels)
    return cast(CausalVAD, model.to(device))


def train() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    audio_config = AudioConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        frame_length_ms=args.frame_length_ms,
        frame_shift_ms=args.frame_shift_ms,
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        train_split=args.train_split,
        val_split=args.val_split,
        experiment_name=args.experiment_name,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    processor = build_preprocessor(audio_config)
    train_loader, val_loader = build_loaders(
        processor=processor,
        results_root=args.results_root,
        labels_root=args.labels_root,
        training_config=training_config,
    )

    model = build_model(device, audio_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=training_config.num_epochs,
        log_dir=args.log_dir,
        experiment_name=training_config.experiment_name,
        checkpoint_path=args.checkpoint_dir,
        audio_config=audio_config,
        training_config=training_config,
    )


if __name__ == "__main__":
    train()
