from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from vad.app.audio.upload import load_audio_from_upload
from vad.app.plots import render_synced_audio_plot
from vad.config import AudioConfig, InferenceConfig
from vad.inference.offline import OfflineVADInferencer, OfflineVADPrediction


def run_offline_inference(
    inferencer: OfflineVADInferencer,
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float,
) -> OfflineVADPrediction:
    """Run offline inference and apply the selected threshold."""
    prediction = inferencer.predict_waveform(waveform, sample_rate)
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


def render_offline_tab(
    inferencer: OfflineVADInferencer,
    inference_config: InferenceConfig,
) -> None:
    """Render offline inference workflow."""
    st.subheader("Offline mode")
    st.write("Upload an audio file and run VAD on the full signal.")

    uploaded_file = st.file_uploader(
        "Upload audio",
        type=["wav", "flac", "mp3"],
        key="offline_uploader",
    )
    if uploaded_file is None:
        return

    audio_config = AudioConfig()

    file_bytes = uploaded_file.getvalue()

    waveform, sample_rate = load_audio_from_upload(
        uploaded_file,
        audio_config.sample_rate,
    )

    with st.spinner("Running offline inference..."):
        prediction = run_offline_inference(
            inferencer=inferencer,
            waveform=waveform,
            sample_rate=sample_rate,
            threshold=inference_config.threshold,
        )
        df = build_prediction_dataframe(prediction)

    duration_s = waveform.numel() / float(sample_rate)

    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{duration_s:.2f} s")
    col2.metric("Frames", f"{len(df):d}")
    col3.metric("Speech ratio", f"{100.0 * df['prediction'].mean():.1f}%")

    render_synced_audio_plot(
        audio_bytes=file_bytes,
        waveform=waveform,
        sample_rate=sample_rate,
        prediction=prediction,
    )
    st.download_button(
        "Download predictions as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="vad_predictions.csv",
        mime="text/csv",
    )
