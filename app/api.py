from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import uuid
import json

from model import diagnose, get_treatment_plan
from typing import Dict, List

app = FastAPI(title="Crop Doctor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> conversation history
# For a hackathon this is fine; swap for Redis in production
sessions: dict[str, list] = {}


class FollowUpRequest(BaseModel):
    session_id: str
    message: str  # transcribed voice or typed text


class DiagnosisResponse(BaseModel):
    session_id: str
    diagnosis: str


class TreatmentResponse(BaseModel):
    session_id: str
    treatment: str


class GemmaRequest(BaseModel):
    features: Dict[str, float]
    notes: str = ""
    top_k: int = 3


class Hypothesis(BaseModel):
    name: str
    confidence: float
    rationale: str = ""


class GemmaResponse(BaseModel):
    hypotheses: List[Hypothesis]


@app.get("/health")
def health():
    return {"status": "ok"}


def rank_from_features(feats: Dict[str, float], notes: str, top_k: int = 3) -> List[Hypothesis]:
    chlorosis = max(0.0, feats.get("yellowing", 0.0) * 1.5)
    dryness = feats.get("brown_fraction", 0.0)
    fungal = feats.get("dark_spot_fraction", 0.0)
    chewing = feats.get("edge_contrast", 0.0)

    candidates = [
        ("Nitrogen deficiency (chlorosis)", min(1.0, chlorosis), "Yellowing relative to green baseline suggests chlorosis."),
        ("Under-watering or heat scorch", min(1.0, dryness * 1.2), "Brown desiccated patches are consistent with water stress."),
        ("Fungal leaf spot / early blight", min(1.0, fungal * 1.4), "Dark low-luminance spots indicate possible fungal lesions."),
        ("Chewing pest damage", min(1.0, chewing * 1.1), "High edge contrast may come from irregular bite marks."),
    ]

    notes_lower = notes.lower()
    boosted = []
    for name, score, rationale in candidates:
        if any(k in notes_lower for k in ["bug", "chew", "hole", "aphid"]) and "pest" in name:
            score += 0.1
        if "yellow" in notes_lower and "chlorosis" in name:
            score += 0.05
        boosted.append((name, min(score, 1.0), rationale))

    ranked = sorted(boosted, key=lambda x: x[1], reverse=True)[:top_k]
    return [Hypothesis(name=n, confidence=round(c, 3), rationale=r) for n, c, r in ranked]


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_plant(file: UploadFile = File(...)):
    """
    Upload a plant image. Returns a diagnosis and a session_id
    to use for follow-up questions.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    diagnosis, history = diagnose(image)

    session_id = str(uuid.uuid4())
    sessions[session_id] = history

    return DiagnosisResponse(session_id=session_id, diagnosis=diagnosis)


@app.post("/treatment", response_model=TreatmentResponse)
async def get_treatment(request: FollowUpRequest):
    """
    Send a follow-up message (farmer's voice input, transcribed).
    Requires a valid session_id from a prior /diagnose call.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please diagnose first.")

    history = sessions[request.session_id]
    treatment, updated_history = get_treatment_plan(history, request.message)
    sessions[request.session_id] = updated_history

    return TreatmentResponse(session_id=request.session_id, treatment=treatment)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear a session when done."""
    sessions.pop(session_id, None)
    return {"status": "cleared"}


@app.post("/gemma3n", response_model=GemmaResponse)
def gemma_endpoint(req: GemmaRequest):
    """
    Endpoint for the CLI Gemma3nAgent. Currently uses heuristic feature
    ranking; swap in model-based scoring if you include images in the payload.
    """
    top_k = max(1, min(req.top_k, 5))
    hyps = rank_from_features(req.features, req.notes, top_k=top_k)
    return GemmaResponse(hypotheses=hyps)
