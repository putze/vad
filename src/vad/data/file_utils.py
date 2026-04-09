from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torchaudio
from torch import Tensor

from vad.data.audio_utils import ensure_mono_waveform


def is_audio_file(path: Path, extensions: tuple[str, ...]) -> bool:
    """
    Check whether a path points to an audio file with an allowed extension.

    Args:
        path: File path to check.
        extensions: Allowed file extensions, compared case-insensitively.

    Returns:
        ``True`` if the path is a file and its extension is allowed,
        otherwise ``False``.
    """
    return path.is_file() and path.suffix.lower() in extensions


def iter_audio_files(
    root: Path,
    extensions: tuple[str, ...] = (".wav",),
) -> list[Path]:
    """
    Recursively collect audio files under a directory.

    Args:
        root: Root directory to scan.
        extensions: Allowed file extensions, compared case-insensitively.

    Returns:
        Sorted list of matching audio file paths.

    Raises:
        ValueError: If ``root`` does not exist or is not a directory.
    """
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid directory: {root}")

    normalized_extensions = tuple(ext.lower() for ext in extensions)

    return [path for path in sorted(root.rglob("*")) if is_audio_file(path, normalized_extensions)]


def match_audio_label_pairs(
    audio_files: Iterable[Path],
    map_fn: Callable[[Path], Path],
) -> tuple[list[tuple[Path, Path]], int]:
    """
    Match audio files with their corresponding label files.

    Args:
        audio_files: Iterable of audio file paths.
        map_fn: Function that maps an audio path to the expected label path.

    Returns:
        A tuple ``(pairs, missing_count)`` where:
        - ``pairs`` is a list of ``(audio_path, label_path)`` tuples for files
          whose labels exist,
        - ``missing_count`` is the number of audio files without a matching
          label file.
    """
    pairs: list[tuple[Path, Path]] = []
    missing_count = 0

    for audio_path in audio_files:
        label_path = map_fn(audio_path)

        if label_path.exists():
            pairs.append((audio_path, label_path))
        else:
            missing_count += 1

    return pairs, missing_count


def load_audio(path: str | Path) -> tuple[Tensor, int]:
    """
    Load an audio file and return a mono waveform.

    Audio is loaded with torchaudio, which returns a tensor of shape
    ``[channels, num_samples]``. Multi-channel audio is converted to mono by
    averaging channels.

    Args:
        path: Path to the audio file.

    Returns:
        A tuple ``(waveform, sample_rate)`` where:
        - ``waveform`` has shape ``[num_samples]``,
        - ``sample_rate`` is in Hz.

    Raises:
        RuntimeError: If the audio file cannot be loaded.
        ValueError: If the loaded waveform does not have the expected shape.
    """
    path = Path(path)

    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {path}") from e

    if waveform.ndim != 2:
        raise ValueError(
            "Expected waveform with shape [channels, num_samples], "
            f"got {tuple(waveform.shape)} for file: {path}"
        )

    waveform = ensure_mono_waveform(waveform)

    return waveform, sample_rate
