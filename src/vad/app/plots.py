from __future__ import annotations

import base64
import json

import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import torch
from plotly.subplots import make_subplots

from vad.app.state import StreamingState
from vad.data.audio_utils import ensure_mono_waveform
from vad.inference.offline import OfflineVADPrediction


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


def render_synced_audio_plot(
    audio_bytes: bytes,
    waveform: torch.Tensor,
    sample_rate: int,
    prediction: OfflineVADPrediction,
    height: int = 860,
) -> None:
    """Render a Plotly-based audio player synced with a 3-row VAD overview plot."""
    mono = ensure_mono_waveform(waveform).detach().cpu()
    waveform_values = mono.numpy().tolist()
    waveform_times = (torch.arange(mono.numel(), dtype=torch.float32) / sample_rate).tolist()

    frame_times = prediction.frame_times.detach().cpu().tolist()
    probabilities = prediction.probabilities.detach().cpu().tolist()
    predictions = prediction.predictions.detach().cpu().to(torch.int32).tolist()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "waveform_times": waveform_times,
        "waveform_values": waveform_values,
        "frame_times": frame_times,
        "probabilities": probabilities,
        "predictions": predictions,
    }

    html = f"""
    <html>
    <head>
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
      <style>
        body {{
          margin: 0;
          background: transparent;
          color: white;
          font-family: sans-serif;
        }}
        #player {{
          width: 100%;
          margin-top: 8px;
        }}
      </style>
    </head>
    <body>
      <div id="plot" style="width:100%; height:{height - 70}px;"></div>
      <audio id="player" controls>
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
      </audio>

      <script>
        const data = {json.dumps(payload)};
        const plotDiv = document.getElementById("plot");
        const player = document.getElementById("player");

        const traces = [
          {{
            x: data.waveform_times,
            y: data.waveform_values,
            mode: "lines",
            name: "Waveform",
            xaxis: "x",
            yaxis: "y",
          }},
          {{
            x: data.frame_times,
            y: data.probabilities,
            mode: "lines",
            name: "Probability",
            xaxis: "x2",
            yaxis: "y2",
          }},
          {{
            x: data.frame_times,
            y: data.predictions,
            mode: "lines",
            line: {{ shape: "hv" }},
            name: "Prediction",
            xaxis: "x3",
            yaxis: "y3",
          }}
        ];

        const layout = {{
          height: {height - 70},
          showlegend: false,
          margin: {{ l: 20, r: 20, t: 60, b: 20 }},
          paper_bgcolor: "#0e1117",
          plot_bgcolor: "#0e1117",
          font: {{ color: "#fafafa" }},

          annotations: [
            {{
              text: "Waveform",
              x: 0.5, y: 1.0,
              xref: "paper", yref: "paper",
              yshift: 20,
              showarrow: false,
              font: {{ size: 16 }}
            }},
            {{
              text: "Speech probability",
              x: 0.5, y: 0.52,
              xref: "paper", yref: "paper",
              yshift: 20,
              showarrow: false,
              font: {{ size: 16 }}
            }},
            {{
              text: "Binary prediction",
              x: 0.5, y: 0.185,
              xref: "paper", yref: "paper",
              yshift: 20,
              showarrow: false,
              font: {{ size: 16 }}
            }}
          ],

          xaxis: {{
            domain: [0, 1],
            anchor: "y",
            matches: "x3",
            showticklabels: false,
            gridcolor: "#283042",
            zerolinecolor: "#283042",
          }},
          yaxis: {{
            domain: [0.59, 1.0],
            title: "Amp.",
            gridcolor: "#283042",
            zerolinecolor: "#283042",
          }},

          xaxis2: {{
            domain: [0, 1],
            anchor: "y2",
            matches: "x3",
            showticklabels: false,
            gridcolor: "#283042",
            zerolinecolor: "#283042",
          }},
          yaxis2: {{
            domain: [0.22, 0.54],
            title: "Prob.",
            range: [0.0, 1.0],
            gridcolor: "#283042",
            zerolinecolor: "#283042",
          }},

          xaxis3: {{
            domain: [0, 1],
            anchor: "y3",
            title: "Time (s)",
            gridcolor: "#283042",
            zerolinecolor: "#283042",
          }},
          yaxis3: {{
            domain: [0.0, 0.17],
            title: "Pred.",
            range: [-0.1, 1.1],
            tickvals: [0, 1],
            gridcolor: "#283042",
            zerolinecolor: "#283042",
          }},

          shapes: [
            {{
              type: "line",
              x0: 0,
              x1: 0,
              y0: 0,
              y1: 1,
              xref: "x3",
              yref: "paper",
              line: {{
                color: "#FFFFFF",
                width: 2
              }}
            }}
          ]
        }};

        Plotly.newPlot(plotDiv, traces, layout, {{
          responsive: true,
          displayModeBar: true
        }});

        function updateCursor(timeSec) {{
          Plotly.relayout(plotDiv, {{
            "shapes[0].x0": timeSec,
            "shapes[0].x1": timeSec
          }});
        }}

        player.addEventListener("timeupdate", () => {{
          updateCursor(player.currentTime);
        }});

        plotDiv.on("plotly_click", (event) => {{
          if (!event.points || event.points.length === 0) return;
          const t = event.points[0].x;
          player.currentTime = t;
          updateCursor(t);
        }});
      </script>
    </body>
    </html>
    """

    components.html(html, height=height)
