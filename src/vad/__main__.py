from __future__ import annotations


def main() -> None:
    print("vad package installed successfully.")
    print("Available commands:")
    print("  vad-train")
    print("  vad-infer-offline")
    print("  vad-stream-file")
    print("  vad-compare-models")
    print("")
    print("For the Streamlit demo, install demo dependencies and run:")
    print('  pip install "vad[demo]"')
    print("  streamlit run -m vad.app.streamlit_app")


if __name__ == "__main__":
    main()
