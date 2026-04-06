from dataclasses import dataclass


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0


@dataclass(slots=True)
class StreamingConfig:
    chunk_seconds: float = 0.5
    rolling_window_seconds: float = 10.0


@dataclass(slots=True)
class InferenceConfig:
    threshold: float = 0.5
