import streamlit as st
import whisper
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Supported languages
SUPPORTED_LANGUAGES = {
    "Auto Detect": None,
    "English": "en",
    "Urdu": "ur",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
}

st.set_page_config(page_title="Speech-to-Text", layout="wide")
st.title("🎤 Speech-to-Text Transcriber")
st.write("Convert speech to text in any language")

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

def transcribe_audio(audio_path, language=None):
    with st.spinner("Transcribing..."):
        return model.transcribe(audio_path, language=language, fp16=False)

# Tabs
tab1, tab2 = st.tabs(["🎙️ Live Recording", "📁 Upload File"])

# ---------------- LIVE RECORDING TAB ----------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()))

    with col2:
        st.write("1. Click record")
        st.write("2. Speak")
        st.write("3. Stop recording")
        st.write("4. Click transcribe")

    audio_bytes = audio_recorder(
        text="🎤 Record",
        recording_color="#e74c3c",
        neutral_color="#2ecc71",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("📝 Transcribe Recording"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                audio_path = f.name

            result = transcribe_audio(audio_path, SUPPORTED_LANGUAGES[language])
            os.remove(audio_path)

            st.success("✅ Transcription complete!")
            st.write(f"**Detected Language:** {result.get('language', 'unknown').upper()}")
            st.text_area("Transcribed Text", result["text"], height=200)

# ---------------- UPLOAD FILE TAB ----------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Select Language",
            list(SUPPORTED_LANGUAGES.keys()),
            key="upload_lang"
        )
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"]
        )

    with col2:
        st.write("Supported formats:")
        st.write("• MP3 • WAV • M4A • OGG • FLAC")

    if uploaded_file and st.button("📝 Transcribe File"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.getbuffer())
            audio_path = f.name

        result = transcribe_audio(audio_path, SUPPORTED_LANGUAGES[language])
        os.remove(audio_path)

        st.success("✅ Transcription complete!")
        st.write(f"**Detected Language:** {result.get('language', 'unknown').upper()}")
        st.text_area("Transcribed Text", result["text"], height=200)

st.divider()
st.info("Whisper supports 99+ languages. Use Auto Detect or select manually.")
