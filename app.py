import io
import base64
import requests
import streamlit as st
from loguru import logger

logger.add(
    "logs/app.log",
    rotation="1 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

API_URL = "http://127.0.0.1:9000"

st.set_page_config(
    page_title="FastIA — Voix vers Image",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Générateur d'Images par la Voix")
st.write("Enregistrez votre voix ou uploadez un fichier audio `.wav` — l'IA génère une image à partir de ce que vous dites.")
st.divider()

# Upload audio file
st.subheader("1️⃣ Chargez votre fichier audio")
st.info("💡 Le fichier doit être au format `.wav`, enregistré à **16 000 Hz** (16kHz).")

audio_file = st.file_uploader(
    label="Choisissez un fichier audio :",
    type=["wav"],
)

if audio_file:
    st.audio(audio_file, format="audio/wav")
    logger.info(f"Fichier audio chargé : {audio_file.name}")

st.divider()

# Bouton de lancement
if st.button("🚀 Générer l'image", type="primary", disabled=not audio_file):

    # transcription
    st.subheader("2️⃣ Transcription de la voix")
    with st.spinner("Transcription en cours avec Wav2Vec2..."):
        try:
            response = requests.post(
                f"{API_URL}/transcription/",
                files={"audio": (audio_file.name, audio_file.getvalue(), "audio/wav")},
            )
            response.raise_for_status()
            texte = response.json()["texte"]
            logger.info(f"Texte transcrit : {texte}")
            st.success(f"**Texte transcrit :** {texte}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de contacter l'API. Vérifiez que FastAPI tourne sur le port 9000.")
            logger.error("Erreur de connexion à l'API.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur lors de la transcription : {e}")
            logger.error(f"Erreur transcription : {e}")
            st.stop()

    # Génération du prompt
    st.subheader("3️⃣ Génération du prompt")
    with st.spinner("Enrichissement du texte avec GPT-2..."):
        try:
            response = requests.post(
                f"{API_URL}/generation_prompt/",
                json={"texte": texte},
            )
            response.raise_for_status()
            prompt = response.json()["prompt"]
            logger.info(f"Prompt généré : {prompt}")
            st.success(f"**Prompt généré :** {prompt}")

        except Exception as e:
            st.error(f"❌ Erreur lors de la génération du prompt : {e}")
            logger.error(f"Erreur génération prompt : {e}")
            st.stop()

    # Génération de l'image
    st.subheader("4️⃣ Génération de l'image")
    with st.spinner("Génération de l'image avec Stable Diffusion... ⏳ (peut prendre 1 à 5 minutes)"):
        try:
            response = requests.post(
                f"{API_URL}/generation_image/",
                json={"prompt": prompt},
                timeout=600,  # 10 min max!
            )
            response.raise_for_status()
            image_base64 = response.json()["image"]

            # Décodage base64 → image affichable
            image_bytes = base64.b64decode(image_base64)
            logger.info("Image reçue et décodée avec succès.")

            st.divider()
            st.subheader("🎨 Image générée")
            st.image(image_bytes, caption=f'"{prompt}"', use_column_width=True)

            # download button
            st.download_button(
                label="⬇️ Télécharger l'image",
                data=image_bytes,
                file_name="image_generee.png",
                mime="image/png",
            )

        except requests.exceptions.Timeout:
            st.error("⏱️ La génération a pris trop de temps. Réessayez ou réduisez le nombre d'étapes.")
            logger.error("Timeout lors de la génération d'image.")
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de l'image : {e}")
            logger.error(f"Erreur génération image : {e}")

elif not audio_file:
    st.warning("⚠️ Veuillez d'abord charger un fichier audio `.wav`.")