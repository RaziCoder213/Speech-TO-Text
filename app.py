import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os

SAMPLE_RATE = 16000  # Whisper works best at 16kHz

# Supported languages (Whisper supports 99+ languages)
SUPPORTED_LANGUAGES = {
    "Auto Detect": None,
    "English": "en",
    "Urdu": "urd",
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
    "Turkish": "tr",
    "Polish": "pl",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    "Greek": "el",
    "Thai": "th",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Filipino": "fil",
    "Hebrew": "he",
    "Ukrainian": "uk",
}

st.set_page_config(page_title="Speech-to-Text", layout="wide")
st.title("🎤 Speech-to-Text Transcriber")
st.write("Convert speech to text in any language")

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

# Transcribe function
def transcribe_audio(audio_path, language=None):
    with st.spinner("Transcribing..."):
        result = model.transcribe(audio_path, language=language, fp16=False)
    return result

# Record audio function
def record_audio(duration_sec=5):
    st.info(f"🔴 Recording for {duration_sec} seconds...")
    recording = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return recording

# UI Tabs
tab1, tab2 = st.tabs(["🎙️ Live Recording", "📁 Upload File"])

# Tab 1: Live Recording
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Settings")
        language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()))
        duration = st.slider("Recording Duration (seconds)", 1, 30, 5)

    with col2:
        st.subheader("Instructions")
        st.write("1. Click the 'Record' button")
        st.write("2. Speak clearly")
        st.write("3. Wait for transcription")

    if st.button("🎤 Record & Transcribe", key="record_btn"):
        audio_data = record_audio(duration)

        # Save temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_normalized = audio_data / np.max(np.abs(audio_data))
            write(tmp.name, SAMPLE_RATE, audio_normalized.astype(np.float32))

            result = transcribe_audio(tmp.name, SUPPORTED_LANGUAGES[language])
            os.remove(tmp.name)

        st.success("✅ Transcription complete!")
        st.write(f"**Detected Language:** {result.get('language', 'unknown').upper()}")
        st.text_area("Transcribed Text", value=result['text'], height=200, disabled=True)

# Tab 2: Upload File
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Audio File")
        language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()), key="upload_lang")
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

    with col2:
        st.subheader("Supported Formats")
        st.write("• MP3")
        st.write("• WAV")
        st.write("• M4A")
        st.write("• OGG")
        st.write("• FLAC")

    if uploaded_file is not None:
        if st.button("📝 Transcribe", key="upload_btn"):
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=False) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            result = transcribe_audio(tmp_path, SUPPORTED_LANGUAGES[language])
            os.remove(tmp_path)

            st.success("✅ Transcription complete!")
            st.write(f"**Detected Language:** {result.get('language', 'unknown').upper()}")
            st.text_area("Transcribed Text", value=result['text'], height=200, disabled=True)

st.divider()
st.info("📌 **Tip:** Whisper supports 99+ languages. Select your language for better accuracy, or use 'Auto Detect' to let Whisper figure it out!")
