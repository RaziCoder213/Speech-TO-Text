import streamlit as st
import whisper
import tempfile
import os
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

SAMPLE_RATE = 16000

SUPPORTED_LANGUAGES = {
    "Auto Detect": None,
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}

st.set_page_config(page_title="Live Speech-to-Text", layout="wide")
st.title("🎤 Live Browser Speech-to-Text")
st.write("Record your voice live in the browser and transcribe using Whisper")

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

webrtc_ctx = webrtc_streamer(
    key="speech-recorder",
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True, "video": False},
)

language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()))

if st.button("📝 Stop & Transcribe"):
    if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.frames:
        audio_data = np.concatenate(webrtc_ctx.audio_processor.frames, axis=1)
        audio_data = audio_data.astype(np.float32) / 32768.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import scipy.io.wavfile as wav
            wav.write(tmp.name, SAMPLE_RATE, audio_data.T)
            tmp_path = tmp.name

        with st.spinner("Transcribing..."):
            result = model.transcribe(
                tmp_path,
                language=SUPPORTED_LANGUAGES[language],
                fp16=False,
            )

        os.remove(tmp_path)

        st.success("✅ Transcription complete!")
        st.write(f"**Detected Language:** {result.get('language', 'unknown').upper()}")
        st.text_area("Transcribed Text", result["text"], height=200)

    else:
        st.warning("No audio recorded yet.")
