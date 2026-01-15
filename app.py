import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

if "recordings" not in st.session_state:
    st.session_state.recordings = []

audio_bytes = audio_recorder("Click to Record")

if audio_bytes:
    st.session_state.recordings.append(audio_bytes)

for i, audio in enumerate(st.session_state.recordings):
    st.audio(audio, format="audio/wav")

    if st.button(f"Transcribe #{i+1}", key=i):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio)
            path = f.name

        audio_data, sr = sf.read(path)

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        with st.spinner("Transcribing..."):
            result = model.transcribe(audio_data, fp16=False)

        os.remove(path)

        st.text_area("Text", result["text"], height=120)
