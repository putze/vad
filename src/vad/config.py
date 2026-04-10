from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0

    @property
    def frame_length_samples(self) -> int:
        return int(self.sample_rate * self.frame_length_ms / 1000)

    @property
    def hop_length_samples(self) -> int:
        return int(self.sample_rate * self.frame_shift_ms / 1000)


@dataclass(slots=True)
class StreamingConfig:
    chunk_seconds: float = 0.5
    min_buffer_seconds: float = 0.025


@dataclass(slots=True)
class InferenceConfig:
    threshold: float = 0.5


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 10
    train_split: str = "train-clean-100"
    val_split: str = "dev-clean"
    experiment_name: str = "causal_conv"
