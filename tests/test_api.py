import pytest
import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


# Fixture: fichier audio WAV synthétique
@pytest.fixture
def fake_audio_wav():
    """Génère un fichier .wav silencieux à 16kHz pour les tests."""
    sample_rate = 16000
    duration = 1  # 1 seconde
    audio_array = np.zeros(sample_rate * duration, dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer


@pytest.fixture
def texte_exemple():
    return {"texte": "a dog running in a park"}


@pytest.fixture
def prompt_exemple():
    return {"prompt": "a dog running in a park, highly detailed, 4k, digital art"}


def test_root_status_code():
    response = client.get("/")
    assert response.status_code == 200


def test_root_message():
    response = client.get("/")
    assert "message" in response.json()


# Tests route POST /transcription/
def test_transcription_status_code(fake_audio_wav):
    response = client.post(
        "/transcription/",
        files={"audio": ("test.wav", fake_audio_wav, "audio/wav")},
    )
    assert response.status_code == 200


def test_transcription_retourne_texte(fake_audio_wav):
    response = client.post(
        "/transcription/",
        files={"audio": ("test.wav", fake_audio_wav, "audio/wav")},
    )
    assert "texte" in response.json()


def test_transcription_texte_est_string(fake_audio_wav):
    response = client.post(
        "/transcription/",
        files={"audio": ("test.wav", fake_audio_wav, "audio/wav")},
    )
    assert isinstance(response.json()["texte"], str)


def test_transcription_sans_fichier():
    response = client.post("/transcription/")
    assert response.status_code == 422


# Tests route POST /generation_prompt/
def test_generation_prompt_status_code(texte_exemple):
    response = client.post("/generation_prompt/", json=texte_exemple)
    assert response.status_code == 200


def test_generation_prompt_retourne_prompt(texte_exemple):
    response = client.post("/generation_prompt/", json=texte_exemple)
    assert "prompt" in response.json()


def test_generation_prompt_contient_texte_original(texte_exemple):
    """Le prompt généré doit contenir le texte original."""
    response = client.post("/generation_prompt/", json=texte_exemple)
    prompt = response.json()["prompt"]
    assert texte_exemple["texte"] in prompt


def test_generation_prompt_champ_manquant():
    response = client.post("/generation_prompt/", json={})
    assert response.status_code == 422


# Tests route POST /generation_image/
def test_generation_image_status_code(prompt_exemple):
    """La route génération d'image doit retourner 200."""
    response = client.post("/generation_image/", json=prompt_exemple)
    assert response.status_code == 200


def test_generation_image_retourne_image(prompt_exemple):
    """La réponse doit contenir la clé 'image'."""
    response = client.post("/generation_image/", json=prompt_exemple)
    assert "image" in response.json()


def test_generation_image_est_base64(prompt_exemple):
    """L'image retournée doit être une chaîne base64 valide."""
    import base64
    response = client.post("/generation_image/", json=prompt_exemple)
    image_base64 = response.json()["image"]
    try:
        base64.b64decode(image_base64)
        assert True
    except Exception:
        assert False, "L'image n'est pas un base64 valide"


def test_generation_image_champ_manquant():
    response = client.post("/generation_image/", json={})
    assert response.status_code == 422