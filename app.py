import streamlit as st
import whisper
import tempfile
import os

# Supported languages
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
st.write("Upload an audio file and convert speech to text in any language")

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

# UI: Upload File
st.subheader("📁 Upload Audio File")
language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()))
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    if st.button("📝 Transcribe"):
        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Transcribe
        result = transcribe_audio(tmp_path, SUPPORTED_LANGUAGES[language])
        os.remove(tmp_path)

        st.success("✅ Transcription complete!")
        st.write(f"**Detected Language:** {result.get('language', 'unknown').upper()}")
        st.text_area("Transcribed Text", value=result['text'], height=200, disabled=True)

st.divider()
st.info("📌 **Tip:** Whisper supports 99+ languages. Use 'Auto Detect' or select your language for better accuracy!")
