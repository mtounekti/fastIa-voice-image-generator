import io
import base64
import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from loguru import logger
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from diffusers import StableDiffusionPipeline

# Configuration
SAMPLE_RATE = 16000
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
PROMPT_SUFFIX = ", highly detailed, 4k, digital art, masterpiece"

logger.add(
    "logs/api.log",
    rotation="1 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


# Device detection
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
logger.info(f"Device: {DEVICE}")


# Models loading
def load_stt_model():
    logger.info("Loading Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    logger.info("Wav2Vec2 loaded.")
    return processor, model


def load_gpt2_pipeline():
    logger.info("Loading GPT-2...")
    gpt2 = pipeline("text-generation", model="gpt2", device=0 if DEVICE == "cuda" else -1)
    logger.info("GPT-2 loaded.")
    return gpt2


def load_sd_pipeline():
    logger.info("Loading Stable Diffusion...")
    dtype = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32
    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    ).to(DEVICE)
    logger.info("Stable Diffusion loaded.")
    return sd


STT_PROCESSOR, STT_MODEL = load_stt_model()
GPT2_PIPELINE = load_gpt2_pipeline()
SD_PIPELINE = load_sd_pipeline()


# Business logic helpers
def run_transcription(audio_bytes: bytes) -> str:
    """Runs Wav2Vec2 inference on raw audio bytes and returns transcribed text."""
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Invalid sample rate: {sample_rate}Hz. Expected: {SAMPLE_RATE}Hz.")

    inputs = STT_PROCESSOR(
        audio_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    ).input_values.to(DEVICE)

    with torch.no_grad():
        logits = STT_MODEL(inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return STT_PROCESSOR.decode(predicted_ids[0]).lower().strip()


def run_prompt_generation(texte: str) -> str:
    """Enriches transcribed text into a Stable Diffusion prompt using GPT-2."""
    result = GPT2_PIPELINE(
        texte,
        max_new_tokens=30,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
    )
    return result[0]["generated_text"] + PROMPT_SUFFIX


def run_image_generation(prompt: str) -> str:
    """Generates an image from a prompt and returns it as a base64 string."""
    image = SD_PIPELINE(
        prompt,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# FastAPI app & routes
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
    logger.info("Root route called.")
    return {"message": "Bienvenue sur l'API FastIA — Voix vers Image 🎙️🎨"}


@app.post("/transcription/")
async def transcription(audio: UploadFile = File(...)):
    """Transcribes a .wav audio file to text."""
    logger.info(f"Audio file received: {audio.filename}")
    try:
        texte = run_transcription(await audio.read())
        logger.info(f"Transcription: {texte}")
        return {"texte": texte}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generation_prompt/")
async def generation_prompt(texte_object: Texte):
    """Enriches transcribed text into a Stable Diffusion prompt."""
    logger.info(f"Text received for prompt generation: {texte_object.texte}")
    try:
        prompt = run_prompt_generation(texte_object.texte)
        logger.info(f"Prompt generated: {prompt}")
        return {"prompt": prompt}
    except Exception as e:
        logger.error(f"Prompt generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generation_image/")
async def generation_image(prompt_object: Prompt):
    """Generates an image from a text prompt using Stable Diffusion."""
    logger.info(f"Prompt received for image generation: {prompt_object.prompt}")
    try:
        image_base64 = run_image_generation(prompt_object.prompt)
        logger.info("Image generated successfully.")
        return {"image": image_base64}
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))