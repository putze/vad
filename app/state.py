from dataclasses import dataclass


@dataclass(slots=True)
class StreamingState:
    """State container used for simulated online inference."""

    times: list[float]
    probabilities: list[float]
    predictions: list[int]
    waveform_times: list[float]
    waveform_values: list[float]
