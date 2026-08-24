from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pathlib import Path
import os
import shutil
import uuid

app = FastAPI(title="ActaAI Audio API", version="0.2.0")

allowed_origins = [
    "https://alfredocespedes-c.github.io",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "actaai-audio",
        "version": "0.2.0",
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac"}:
        raise HTTPException(status_code=400, detail="Formato no soportado")

    target = UPLOADS / f"{uuid.uuid4()}{suffix}"

    try:
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        segments, info = model.transcribe(
            str(target),
            language="es",
            vad_filter=True,
            beam_size=1,
        )

        result_segments = []
        full_text = []

        for seg in segments:
            text = seg.text.strip()
            result_segments.append(
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "speaker": "SPEAKER_00",
                    "text": text,
                }
            )
            if text:
                full_text.append(text)

        return {
            "status": "completed",
            "filename": file.filename,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
            "speakers": 1,
            "segments": result_segments,
            "transcript": " ".join(full_text),
            "diarization": "pending",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {exc}")
    finally:
        if target.exists():
            target.unlink(missing_ok=True)
