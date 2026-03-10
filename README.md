# 🎙️ FastIA — Générateur d'Images par la Voix

Application web qui capture la parole de l'utilisateur, la transcrit en texte, génère un prompt enrichi et produit une image à partir de ce prompt grâce à Stable Diffusion

---

## 📋 Description

L'utilisateur enregistre sa voix via l'interface Streamlit. La parole est transcrite en texte grâce au modèle **Wav2Vec2** de Facebook. Ce texte est ensuite enrichi en prompt par le modèle de langage **GPT-2**. Enfin, le prompt est envoyé au modèle **Stable Diffusion** qui génère une image affichée dans l'interface.

### Modèles IA utilisés (Hugging Face)

| Rôle | Modèle |
|------|--------|
| 🎙️ Speech-to-Text | `facebook/wav2vec2-base-960h` |
| ✍️ Génération de prompt | `gpt2` |
| 🎨 Génération d'image | `runwayml/stable-diffusion-v1-5` |

---

## 🏗️ Architecture

```
fastia-voice-to-image/
├── .venv/                  ← environnement virtuel (non versionné)
├── api.py                  ← serveur FastAPI (3 routes IA)
├── app.py                  ← interface utilisateur Streamlit
├── requirements.txt        ← dépendances du projet
├── .gitignore              ← exclut .venv, logs, modèles
└── README.md               ← documentation du projet
```

### Flux de données

```
Utilisateur (voix)
      │
      ▼
[ Streamlit - app.py ]         → enregistrement audio (port 8501)
      │  fichier .wav + requête HTTP POST
      ▼
[ FastAPI - api.py ]           → API REST (port 9000)
      │
      ├──▶ Wav2Vec2            → transcription audio → texte
      │
      ├──▶ GPT-2               → texte → prompt enrichi
      │
      └──▶ Stable Diffusion    → prompt → image générée
               │
               ▼
[ Streamlit - app.py ]         → affichage de l'image
```

---

## ⚙️ Installation

### Prérequis

- Python 3.9+
- Git
- ~6 Go d'espace disque

### 1. Cloner le dépôt

```bash
git clone https://github.com/TON_USERNAME/fastia-voice-to-image.git
cd fastia-voice-to-image
```

### 2. Créer et activer l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⚠️ Le téléchargement des modèles Hugging Face se fait automatiquement
> au premier lancement. Prévoir ~6 Go et une bonne connexion internet.

### 4. Installer les dépendances système (macOS)

```bash
brew install portaudio ffmpeg
```

---

## 🚀 Lancement

L'application nécessite **deux terminaux** ouverts simultanément.

### Terminal 1 — Lancer l'API FastAPI

```bash
uvicorn api:app --reload --port 9000
```

API accessible sur : [http://127.0.0.1:9000](http://127.0.0.1:9000)
Documentation interactive : [http://127.0.0.1:9000/docs](http://127.0.0.1:9000/docs)

### Terminal 2 — Lancer l'interface Streamlit

```bash
streamlit run app.py
```

Interface accessible sur : [http://localhost:8501](http://localhost:8501)

---

## 🛣️ Routes de l'API

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | Vérifie que l'API est en ligne |
| `POST` | `/transcription/` | Transcrit un fichier audio en texte |
| `POST` | `/generation_prompt/` | Enrichit un texte en prompt |
| `POST` | `/generation_image/` | Génère une image à partir d'un prompt |

### Détail des routes POST

#### `POST /transcription/`
**Corps de la requête :** fichier audio `.wav`
```json
{ "audio": "<fichier binaire>" }
```
**Réponse :**
```json
{ "texte": "a cat sitting on a red chair" }
```

#### `POST /generation_prompt/`
**Corps de la requête :**
```json
{ "texte": "a cat sitting on a red chair" }
```
**Réponse :**
```json
{ "prompt": "a cat sitting on a red chair, highly detailed, 4k, digital art" }
```

#### `POST /generation_image/`
**Corps de la requête :**
```json
{ "prompt": "a cat sitting on a red chair, highly detailed, 4k, digital art" }
```
**Réponse :** image PNG encodée en base64
```json
{ "image": "<base64_string>" }
```

---

## 📦 Dépendances

```
fastapi
uvicorn
streamlit
requests
transformers
torch
diffusers
accelerate
soundfile
speechrecognition
pyaudio
loguru
```

> Toutes les dépendances sont listées dans `requirements.txt`.

---

## 🍎 Accélération Apple Silicon (MPS)

Sur Mac M1/M2/M3, Stable Diffusion utilise automatiquement le **Metal Performance Shaders (MPS)** pour accélérer la génération d'images, sans nécessiter de GPU NVIDIA.

```python
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

---

## 📝 Journalisation

L'application utilise **Loguru** pour journaliser les étapes clés : réception audio, transcription, génération de prompt et génération d'image.

---

## 🧪 Tests (Bonus)

```bash
pip install pytest httpx
pytest tests/test_api.py -v
```

### ⏱️ Durée des tests

Les tests sont longs à cause du **chargement des modèles IA** au démarrage de chaque session de test

### 💡 Astuce pour tester plus vite

```bash
# Ignorer les tests de génération d'image (les plus lents)
pytest tests/test_api.py -v -k "not generation_image"

# Lancer un seul test
pytest tests/test_api.py::test_root_status_code -v
```

---

## 👤 Auteur

Projet réalisé dans le cadre de la formation **FastIA** — Module 0, Brief 2 - By Maroua Tounekti