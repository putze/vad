from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioSample:
    """
    Container for a single audio sample and its corresponding label.

    Attributes:
        audio_path (Path): Path to the audio file.
        label_path (Path): Path to the label file associated with the audio.
    """

    audio_path: Path
    label_path: Path
