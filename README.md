# Speech-TO-Text

Simple Speech-to-Text app using OpenAI Whisper and Streamlit.

## Run locally

Install dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (if not already).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, choose this repository and the `main` branch, and set the file to `app.py`.
4. Click **Deploy**. Streamlit will install packages from `requirements.txt` and launch the app.

Notes:
- The app uses Whisper and may require more memory/compute than the free tier allows. If you hit resource limits, consider using a smaller Whisper model or hosting on a VM with GPUs.
- If you need system packages, add an `apt.txt` or custom setup as documented by Streamlit Cloud.
