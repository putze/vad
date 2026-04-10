from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from vad.app.audio.upload import load_audio_from_upload
from vad.app.plots import (
    plot_overview,
    plot_predictions,
    plot_probabilities,
    plot_waveform,
)
from vad.config import AudioConfig
from vad.inference.offline import OfflineVADInferencer, OfflineVADPrediction


def run_offline_inference(
    inferencer: OfflineVADInferencer,
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float,
) -> OfflineVADPrediction:
    """Run offline inference and apply selected threshold if needed."""
    prediction = inferencer.predict_waveform(waveform, sample_rate)

    # If your current inferencer already thresholds internally, remove this block.
    prediction.predictions = (prediction.probabilities >= threshold).to(torch.int64)
    return prediction


def build_prediction_dataframe(prediction: OfflineVADPrediction) -> pd.DataFrame:
    """Convert prediction output to a table for display and export."""
    return pd.DataFrame(
        {
            "time_s": prediction.frame_times.detach().cpu().numpy(),
            "probability": prediction.probabilities.detach().cpu().numpy(),
            "prediction": prediction.predictions.detach().cpu().numpy().astype(int),
        }
    )


def render_offline_tab(inferencer: OfflineVADInferencer, threshold: float) -> None:
    """Render offline inference workflow."""
    st.subheader("Offline mode")
    st.write("Upload an audio file and run VAD on the full signal.")

    uploaded_file = st.file_uploader(
        "Upload audio", type=["wav", "flac", "mp3"], key="offline_uploader"
    )
    if uploaded_file is None:
        return

    audio_config = AudioConfig()

    file_bytes = uploaded_file.getvalue()
    st.audio(file_bytes)

    waveform, sample_rate = load_audio_from_upload(uploaded_file, audio_config.sample_rate)

    with st.spinner("Running offline inference..."):
        prediction = run_offline_inference(inferencer, waveform, sample_rate, threshold)
        df = build_prediction_dataframe(prediction)

    duration_s = waveform.numel() / float(sample_rate)
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{duration_s:.2f} s")
    col2.metric("Frames", f"{len(df):d}")
    col3.metric("Speech ratio", f"{100.0 * df['prediction'].mean():.1f}%")

    st.plotly_chart(plot_overview(waveform, sample_rate, prediction), use_container_width=True)

    with st.expander("Separate plots"):
        st.plotly_chart(plot_waveform(waveform, sample_rate), use_container_width=True)
        st.plotly_chart(plot_probabilities(prediction), use_container_width=True)
        st.plotly_chart(plot_predictions(prediction), use_container_width=True)

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="vad_predictions.csv",
        mime="text/csv",
    )
