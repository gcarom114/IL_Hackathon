from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import uuid
import json

from model import diagnose, get_treatment_plan  # real model — uncomment when ready
# from model_mock import diagnose, get_treatment_plan
from model import diagnose, get_treatment_plan  
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


@app.get("/health")
def health():
    return {"status": "ok"}


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


# Serve frontend — must be mounted LAST after all API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
