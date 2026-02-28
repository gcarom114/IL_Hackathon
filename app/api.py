from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
import uuid

from model import diagnose, get_treatment_plan

app = FastAPI(title="Crop Doctor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, list] = {}

LANG_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'sw': 'Swahili',
    'pt': 'Portuguese',
}


class FollowUpRequest(BaseModel):
    session_id: str
    message: str
    lang: Optional[str] = 'en'


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
async def diagnose_plant(
    file: UploadFile = File(...),
    lang: str = Form(default='en')
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    lang_name = LANG_NAMES.get(lang, 'English')
    diagnosis, history = diagnose(image, lang_name)

    session_id = str(uuid.uuid4())
    sessions[session_id] = history

    return DiagnosisResponse(session_id=session_id, diagnosis=diagnosis)


@app.post("/treatment", response_model=TreatmentResponse)
async def get_treatment(request: FollowUpRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please diagnose first.")

    lang_name = LANG_NAMES.get(request.lang or 'en', 'English')
    history = sessions[request.session_id]
    treatment, updated_history = get_treatment_plan(history, request.message, lang_name)
    sessions[request.session_id] = updated_history

    return TreatmentResponse(session_id=request.session_id, treatment=treatment)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
