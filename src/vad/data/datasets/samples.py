from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AudioExample:
    """
    Metadata for a single dataset example.

    This object does not contain audio data itself. It only stores file paths
    used to load:
    - the waveform
    - the corresponding label sequence
    """

    audio_path: Path
    label_path: Path
