from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torchaudio
from torch import Tensor

from vad.data.utils import ensure_mono_waveform


def is_audio_file(path: Path, extensions: tuple[str, ...]) -> bool:
    """
    Check whether a path is a valid audio file based on extension.

    Args:
        path (Path): File path to check.
        extensions (tuple[str, ...]): Allowed audio file extensions.

    Returns:
        bool: True if valid audio file, else False.
    """
    return path.is_file() and path.suffix.lower() in extensions


def iter_audio_files(
    root: Path,
    extensions: tuple[str, ...] = (".wav",),
) -> list[Path]:
    """
    Recursively collect all audio files under a directory.

    Args:
        root (Path): Root directory.
        extensions (tuple[str, ...]): Allowed audio file extensions.

    Returns:
        list[Path]: Sorted list of audio file paths.
    """
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid directory: {root}")

    extensions = tuple(ext.lower() for ext in extensions)

    return [path for path in sorted(root.rglob("*")) if is_audio_file(path, extensions)]


def match_audio_label_pairs(
    audio_files: Iterable[Path],
    map_fn,
) -> tuple[list[tuple[Path, Path]], int]:
    """
    Match audio files with corresponding label files.

    Args:
        audio_files (Iterable[Path]): Audio file paths.
        map_fn (Callable[[Path], Path]): Function mapping audio_path → label_path.

    Returns:
        tuple[list[tuple[Path, Path]], int]:
            Matched (audio_path, label_path) pairs and count of missing labels.
    """
    pairs: list[tuple[Path, Path]] = []
    missing = 0

    for audio_path in audio_files:
        label_path = map_fn(audio_path)

        if label_path.exists():
            pairs.append((audio_path, label_path))
        else:
            missing += 1

    return pairs, missing


def load_audio(
    path: str | Path,
) -> tuple[Tensor, int]:
    """
    Load an audio file and return a mono waveform.

    Args:
        path: Path to the audio file.

    Returns:
        waveform: Mono waveform of shape [N].
        sample_rate: Sample rate in Hz.

    Raises:
        RuntimeError: If loading fails.
        ValueError: If the waveform shape is invalid.
    """
    path = Path(path)

    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {path}") from e

    # Expect [channels, time]
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected waveform shape [channels, time], "
            f"got {tuple(waveform.shape)} for file: {path}"
        )

    # Convert to mono
    waveform = ensure_mono_waveform(waveform)

    return waveform, sample_rate
