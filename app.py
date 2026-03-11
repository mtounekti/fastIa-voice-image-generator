import io
import base64
import requests
import streamlit as st
from loguru import logger

API_URL = "http://127.0.0.1:9000"

logger.add(
    "logs/app.log",
    rotation="1 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

st.set_page_config(
    page_title="FastIA — Voix vers Image",
    page_icon="🎙️",
    layout="centered",
)


def transcribe_audio(audio_file) -> str:
    """Sends audio file to the API and returns the transcribed text."""
    response = requests.post(
        f"{API_URL}/transcription/",
        files={"audio": (audio_file.name, audio_file.getvalue(), "audio/wav")},
    )
    response.raise_for_status()
    texte = response.json()["texte"]
    logger.info(f"Transcription: {texte}")
    return texte


def generate_prompt(texte: str) -> str:
    """Sends transcribed text to the API and returns an enriched prompt."""
    response = requests.post(
        f"{API_URL}/generation_prompt/",
        json={"texte": texte},
    )
    response.raise_for_status()
    prompt = response.json()["prompt"]
    logger.info(f"Prompt: {prompt}")
    return prompt


def generate_image(prompt: str) -> bytes:
    """Sends prompt to the API and returns the generated image as bytes."""
    response = requests.post(
        f"{API_URL}/generation_image/",
        json={"prompt": prompt},
        timeout=600,
    )
    response.raise_for_status()
    image_bytes = base64.b64decode(response.json()["image"])
    logger.info("Image generated successfully.")
    return image_bytes


# UI
st.title("🎙️ Générateur d'Images par la Voix")
st.write("Uploadez un fichier audio `.wav` — l'IA génère une image à partir de ce que vous dites.")
st.divider()

# Step 1 — Audio upload
st.subheader("1️⃣ Chargez votre fichier audio")
st.info("Format `.wav` requis, enregistré à **16 000 Hz** (16kHz).")
audio_file = st.file_uploader(label="Choisissez un fichier audio :", type=["wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")
    logger.info(f"Audio file loaded: {audio_file.name}")

st.divider()

# Step 2 — Generate button
if st.button("🚀 Générer l'image", type="primary", disabled=not audio_file):

    # Transcription
    st.subheader("2️⃣ Transcription de la voix")
    with st.spinner("Transcription en cours avec Wav2Vec2..."):
        try:
            texte = transcribe_audio(audio_file)
            st.success(f"**Texte transcrit :** {texte}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de contacter l'API. Vérifiez que FastAPI tourne sur le port 9000.")
            logger.error("API connection error.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur lors de la transcription : {e}")
            logger.error(f"Transcription error: {e}")
            st.stop()

    # Prompt generation
    st.subheader("3️⃣ Génération du prompt")
    with st.spinner("Enrichissement du texte avec GPT-2..."):
        try:
            prompt = generate_prompt(texte)
            st.success(f"**Prompt généré :** {prompt}")
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération du prompt : {e}")
            logger.error(f"Prompt generation error: {e}")
            st.stop()

    # Image generation
    st.subheader("4️⃣ Génération de l'image")
    with st.spinner("Génération avec Stable Diffusion... ⏳ (1 à 5 minutes)"):
        try:
            image_bytes = generate_image(prompt)
            st.divider()
            st.subheader("🎨 Image générée")
            st.image(image_bytes, caption=f'"{prompt}"', use_column_width=True)
            st.download_button(
                label="⬇️ Télécharger l'image",
                data=image_bytes,
                file_name="image_generee.png",
                mime="image/png",
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ La génération a pris trop de temps. Réessayez.")
            logger.error("Image generation timeout.")
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de l'image : {e}")
            logger.error(f"Image generation error: {e}")

elif not audio_file:
    st.warning("⚠️ Veuillez d'abord charger un fichier audio `.wav`.")