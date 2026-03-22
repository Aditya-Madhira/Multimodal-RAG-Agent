"""
Whisper transcription endpoint for speech-to-text.
Run with: uvicorn transcribe:app --host 0.0.0.0 --port 8001
"""
import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper

app = FastAPI(title="Whisper Transcription Service")

# Add CORS to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model (tiny for CPU efficiency)
MODEL = None
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")

def load_whisper_model():
    global MODEL
    if MODEL is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = whisper.load_model(MODEL_NAME, device=device)
    return MODEL

@app.get("/")
async def root():
    return {"status": "ready", "model": MODEL_NAME}

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio file to text using Whisper."""
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read audio data
        audio_bytes = await audio.read()
        
        # Load Whisper model
        model = load_whisper_model()
        
        # Transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            result = model.transcribe(tmp_path, word_timestamps=False)
            text = result["text"].strip()
        finally:
            os.unlink(tmp_path)
        
        return JSONResponse(content={"text": text})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
