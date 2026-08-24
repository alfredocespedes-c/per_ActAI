from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, uuid

app = FastAPI(title="ActaAI Audio API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
UPLOADS = Path("uploads"); UPLOADS.mkdir(exist_ok=True)

@app.get('/health')
def health():
    return {"ok": True, "service": "actaai-audio", "mode": "integration-ready"}

@app.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.mp3','.wav','.m4a','.mp4','.ogg','.flac'}:
        raise HTTPException(400, 'Formato no soportado')
    target = UPLOADS / f"{uuid.uuid4()}{suffix}"
    with target.open('wb') as out:
        shutil.copyfileobj(file.file, out)
    # TODO producción:
    # 1. faster-whisper -> segmentos con timestamps
    # 2. pyannote.audio -> diarización SPEAKER_00, SPEAKER_01...
    # 3. alinear ambos resultados
    # 4. enviar transcript estructurado a un LLM para resumen/tareas/temas
    return {"status":"received","filename":file.filename,"stored_as":target.name,
            "next":"Conectar pipeline Whisper + pyannote en este endpoint."}
