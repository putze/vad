from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from app.state import StreamingState
from src.vad.inference import OfflineVADPrediction
from src.vad.inference.utils import ensure_mono_waveform


def plot_waveform(waveform: torch.Tensor, sample_rate: int) -> go.Figure:
    """Plot waveform alone."""
    mono = ensure_mono_waveform(waveform).detach().cpu()
    times = np.arange(mono.numel(), dtype=np.float32) / float(sample_rate)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=mono.numpy(),
            mode="lines",
            name="Waveform",
        )
    )
    fig.update_layout(
        title="Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_probabilities(prediction: OfflineVADPrediction) -> go.Figure:
    """Plot speech probabilities."""
    times = prediction.frame_times.detach().cpu().numpy()
    probs = prediction.probabilities.detach().cpu().numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=probs,
            mode="lines",
            name="Speech probability",
        )
    )
    fig.update_yaxes(range=[0.0, 1.0])
    fig.update_layout(
        title="Speech probabilities",
        xaxis_title="Time (s)",
        yaxis_title="Probability",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_predictions(prediction: OfflineVADPrediction) -> go.Figure:
    """Plot binary VAD predictions."""
    times = prediction.frame_times.detach().cpu().numpy()
    preds = prediction.predictions.detach().cpu().numpy().astype(np.int32)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=preds,
            mode="lines",
            line_shape="hv",
            name="Prediction",
        )
    )
    fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1])
    fig.update_layout(
        title="Binary predictions",
        xaxis_title="Time (s)",
        yaxis_title="Speech / non-speech",
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_overview(
    waveform: torch.Tensor, sample_rate: int, prediction: OfflineVADPrediction
) -> go.Figure:
    """Create a stacked overview plot: waveform, probabilities, predictions."""
    mono = ensure_mono_waveform(waveform).detach().cpu()
    waveform_times = np.arange(mono.numel(), dtype=np.float32) / float(sample_rate)

    frame_times = prediction.frame_times.detach().cpu().numpy()
    probs = prediction.probabilities.detach().cpu().numpy()
    preds = prediction.predictions.detach().cpu().numpy().astype(np.int32)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=("Waveform", "Speech probability", "Binary prediction"),
    )

    fig.add_trace(
        go.Scatter(x=waveform_times, y=mono.numpy(), mode="lines", name="Waveform"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame_times, y=probs, mode="lines", name="Probability"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame_times, y=preds, mode="lines", line_shape="hv", name="Prediction"),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="Amp.", row=1, col=1)
    fig.update_yaxes(title_text="Prob.", range=[0.0, 1.0], row=2, col=1)
    fig.update_yaxes(title_text="Pred.", range=[-0.1, 1.1], tickvals=[0, 1], row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(height=720, showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_streaming_state(state: StreamingState) -> go.Figure:
    """Plot rolling online state."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=(
            "Rolling waveform",
            "Rolling speech probability",
            "Rolling binary prediction",
        ),
    )

    fig.add_trace(
        go.Scatter(x=state.waveform_times, y=state.waveform_values, mode="lines", name="Waveform"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=state.times, y=state.probabilities, mode="lines", name="Probability"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=state.times, y=state.predictions, mode="lines", line_shape="hv", name="Prediction"
        ),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="Amp.", row=1, col=1)
    fig.update_yaxes(title_text="Prob.", range=[0.0, 1.0], row=2, col=1)
    fig.update_yaxes(title_text="Pred.", range=[-0.1, 1.1], tickvals=[0, 1], row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(height=720, showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig
