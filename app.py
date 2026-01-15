import streamlit as st
import whisper
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Speech-to-Text", layout="wide")
st.title("🎤 Speech-to-Text Recorder")

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

# Init storage
if "recordings" not in st.session_state:
    st.session_state.recordings = []

st.subheader("🎙️ Record Audio")

audio_bytes = audio_recorder(
    text="Click to Record",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
)

# SAVE AUDIO EXPLICITLY
if audio_bytes:
    st.session_state.recordings.append(audio_bytes)
    st.success("Audio saved")

st.divider()
st.subheader("📂 Saved Recordings")

if not st.session_state.recordings:
    st.info("No recordings yet")
else:
    for idx, audio in enumerate(st.session_state.recordings):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.audio(audio, format="audio/wav")

        with col2:
            if st.button(f"📝 Transcribe #{idx+1}", key=f"t_{idx}"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio)
                    path = f.name

                with st.spinner("Transcribing..."):
                    result = model.transcribe(path, fp16=False)

                os.remove(path)

                st.text_area(
                    f"Text #{idx+1}",
                    result["text"],
                    height=120
                )
