import io
import base64
import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from loguru import logger
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)
from diffusers import StableDiffusionPipeline

# Loguru
logger.add(
    "logs/api.log",
    rotation="1 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# Détection du device (MPS pour Apple Silicon, sinon CPU)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

logger.info(f"Device utilisé : {DEVICE}")

# Chargement des modèles 
# Modèle Speech-to-Text : Wav2Vec2
logger.info("Chargement du modèle Wav2Vec2...")
STT_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
STT_MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
logger.info("Wav2Vec2 chargé.")

# Modèle de génération de texte : GPT-2
logger.info("Chargement du modèle GPT-2...")
GPT2_PIPELINE = pipeline("text-generation", model="gpt2", device=0 if DEVICE == "cuda" else -1)
logger.info("GPT-2 chargé.")

# Modèle de génération d'images : Stable Diffusion
logger.info("Chargement de Stable Diffusion...")
SD_PIPELINE = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32,
)
SD_PIPELINE = SD_PIPELINE.to(DEVICE)
logger.info("Stable Diffusion chargé.")

# Application FastAPI
app = FastAPI(
    title="FastIA — API Voix vers Image",
    description="API qui transcrit la voix, génère un prompt et produit une image.",
    version="1.0.0",
)

class Texte(BaseModel):
    texte: str

class Prompt(BaseModel):
    prompt: str


@app.get("/")
def root():
    """Vérifie que l'API est en ligne."""
    logger.info("Route racine appelée")
    return {"message": "Bienvenue sur l'API FastIA — Voix vers Image 🎙️🎨"}


# Route POST /transcription/
@app.post("/transcription/")
async def transcription(audio: UploadFile = File(...)):
    """
    Transcrit un fichier audio .wav en texte.
    - **audio** : fichier .wav envoyé en multipart/form-data
    """
    logger.info(f"Fichier audio reçu : {audio.filename}")
    try:
        # Lecture du fichier audio
        contents = await audio.read()
        audio_array, sample_rate = sf.read(io.BytesIO(contents))

        # Wav2Vec2 attend à priori 16kHz => reéchantillonnage à 16kHz si nécessaire
        if sample_rate != 16000:
            raise HTTPException(
                status_code=400,
                detail=f"Taux d'échantillonnage invalide : {sample_rate}Hz. Attendu : 16000Hz."
            )

        inputs = STT_PROCESSOR(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values.to(DEVICE)

        with torch.no_grad():
            logits = STT_MODEL(inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        texte = STT_PROCESSOR.decode(predicted_ids[0])
        texte = texte.lower().strip()

        logger.info(f"Transcription obtenue : {texte}")
        return {"texte": texte}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la transcription : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Route POST /generation_prompt/
@app.post("/generation_prompt/")
async def generation_prompt(texte_object: Texte):
    """
    Enrichit un texte transcrit en prompt pour Stable Diffusion.
    - **texte** : texte brut issu de la transcription
    """
    logger.info(f"Texte reçu pour génération de prompt : {texte_object.texte}")
    try:
        # GPT-2 enrichit le texte en prompt descriptif
        resultat = GPT2_PIPELINE(
            texte_object.texte,
            max_new_tokens=30,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )
        prompt_brut = resultat[0]["generated_text"]

        # On ajoute des modificateurs visuels pour améliorer la qualité de l'image
        prompt_final = f"{prompt_brut}, highly detailed, 4k, digital art, masterpiece"

        logger.info(f"Prompt généré : {prompt_final}")
        return {"prompt": prompt_final}

    except Exception as e:
        logger.error(f"Erreur lors de la génération du prompt : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Route POST /generation_image/
@app.post("/generation_image/")
async def generation_image(prompt_object: Prompt):
    """
    Génère une image à partir d'un prompt textuel avec Stable Diffusion
    - **prompt** : description textuelle de l'image à générer
    """
    logger.info(f"Prompt reçu pour génération d'image : {prompt_object.prompt}")
    try:
        # Génération de l'image
        image = SD_PIPELINE(
            prompt_object.prompt,
            num_inference_steps=30,       # 30 étapes = bon compromis qualité/vitesse
            guidance_scale=7.5,           # fidélité au prompt
        ).images[0]

        # Conversion en base64 pour l'envoyer via JSON
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.info("Image générée avec succès.")
        return {"image": image_base64}

    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'image : {e}")
        raise HTTPException(status_code=500, detail=str(e))