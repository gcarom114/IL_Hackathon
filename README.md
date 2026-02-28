# 🌱 Crop Doctor

An on-device AI agent that diagnoses plant diseases from a photo, then takes voice input to generate a tailored treatment plan. Built with Gemma 3n — fully offline, no API costs, works in the field.

Built at the **InstalILY x Google DeepMind On-Device AI Hackathon**.

---

## How It Works

1. Farmer points camera at a sick plant → image sent to `/diagnose`
2. Model returns diagnosis + confidence level
3. Farmer speaks follow-up context ("I have copper fungicide, 2 acres affected, rained this morning")
4. Transcribed voice sent to `/treatment` → model returns specific spray plan

---

## Stack

- **Gemma 3n E4B** (on-device multimodal LLM via HuggingFace Transformers)
- **FastAPI** backend
- **Docker** for reproducible setup

---

## Setup

### Prerequisites
- Docker + Docker Compose
- NVIDIA GPU recommended (CPU fallback works for demo)
- HuggingFace account with Gemma 3n access

### 1. Accept the Gemma 3n license
Go to https://huggingface.co/google/gemma-3n-E4B-it and click **Acknowledge license**.

### 2. Clone and configure
```bash
git clone <this-repo>
cd crop-doctor
cp .env.example .env
# Edit .env and add your HuggingFace token
```

### 3. Pre-download the model (do this before the demo, saves time)
```bash
HF_HOME=./model_cache python preload_model.py
```

### 4. Run
```bash
docker-compose up --build
```

API available at `http://localhost:8000`

---

## API

### `POST /diagnose`
Upload a plant image. Returns a diagnosis and a `session_id`.

```bash
curl -X POST http://localhost:8000/diagnose \
  -F "file=@sick_plant.jpg"
```

Response:
```json
{
  "session_id": "abc-123",
  "diagnosis": "This appears to be Early Blight (Alternaria solani) with ~85% confidence..."
}
```

### `POST /treatment`
Send farmer's follow-up (transcribed voice). Requires `session_id` from `/diagnose`.

```bash
curl -X POST http://localhost:8000/treatment \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "message": "I have copper fungicide, 2 acres affected, rained this morning"}'
```

Response:
```json
{
  "session_id": "abc-123",
  "treatment": "Apply copper fungicide at 2 tbsp per gallon..."
}
```

### `GET /health`
Check if the API is running.

---

## No GPU?
Remove the `deploy` block from `docker-compose.yml`. The model will run on CPU — slower but functional for demos.
