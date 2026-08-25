from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pathlib import Path
from collections import Counter
import os, re, uuid, time, threading

app = FastAPI(title="ActaAI Audio API", version="0.7.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "3"))
VAD_ENABLED = os.getenv("WHISPER_VAD", "true").lower() not in {"0", "false", "no"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1000"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CHUNK_MB = int(os.getenv("MAX_CHUNK_MB", "12"))
MAX_CHUNK_BYTES = MAX_CHUNK_MB * 1024 * 1024
MAX_LIVE_CHUNK_MB = int(os.getenv("MAX_LIVE_CHUNK_MB", "40"))
MAX_LIVE_CHUNK_BYTES = MAX_LIVE_CHUNK_MB * 1024 * 1024
DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "false").lower() in {"1", "true", "yes"}
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ALLOWED_UPLOADS = {".mp3", ".wav", ".m4a", ".m4b", ".mp4", ".mp4a", ".aac", ".caf", ".ogg", ".flac", ".webm", ".3gp", ".3g2", ".mov"}
MIME_SUFFIXES = {"audio/mp4":".m4a","audio/x-m4a":".m4a","audio/m4a":".m4a","audio/aac":".aac","audio/x-aac":".aac","audio/x-caf":".caf","audio/caf":".caf","video/mp4":".mp4","video/quicktime":".mov","audio/mpeg":".mp3","audio/wav":".wav","audio/x-wav":".wav","audio/ogg":".ogg","audio/flac":".flac","audio/webm":".webm"}

_model = None
_model_error = None
_model_lock = threading.Lock()
_diarization_pipeline = None
_diarization_error = None
_diarization_lock = threading.Lock()
JOBS = {}
JOBS_LOCK = threading.Lock()

STOPWORDS = {"para","como","pero","porque","cuando","donde","desde","hasta","sobre","entre","este","esta","estos","estas","esto","eso","esa","ese","aqui","ahi","muy","mas","menos","tambien","entonces","bueno","bien","vamos","tiene","tener","hacer","hace","hecho","hay","que","del","las","los","una","uno","unos","unas","por","con","sin","y","o","de","la","el","en","un","al","se","es","lo","le","me","te","nos","su","sus","mi","mis","ya","si"}

def get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            _model_error = None
            kwargs = {"device": DEVICE, "compute_type": COMPUTE_TYPE}
            if CPU_THREADS > 0:
                kwargs["cpu_threads"] = CPU_THREADS
            _model = WhisperModel(MODEL_NAME, **kwargs)
            return _model
        except Exception as exc:
            _model_error = str(exc)
            raise RuntimeError(f"No se pudo iniciar Whisper ({MODEL_NAME}): {exc}") from exc

def get_diarization_pipeline():
    global _diarization_pipeline, _diarization_error
    if not DIARIZATION_ENABLED:
        return None
    if _diarization_pipeline is not None:
        return _diarization_pipeline
    with _diarization_lock:
        if _diarization_pipeline is not None:
            return _diarization_pipeline
        try:
            from pyannote.audio import Pipeline
            _diarization_error = None
            kwargs = {}
            if HF_TOKEN:
                kwargs["token"] = HF_TOKEN
            try:
                _diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, **kwargs)
            except TypeError:
                if HF_TOKEN:
                    kwargs = {"use_auth_token": HF_TOKEN}
                _diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, **kwargs)
            return _diarization_pipeline
        except Exception as exc:
            _diarization_error = str(exc)
            raise RuntimeError(f"No se pudo iniciar diarización ({DIARIZATION_MODEL}): {exc}") from exc

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "actaai-audio",
        "version": "0.7.0",
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_chunk_mb": MAX_CHUNK_MB,
        "live_transcription": True,
        "chunked_uploads": True,
        "apple_mpeg4_audio": True,
        "whisper": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "vad": VAD_ENABLED,
            "status": "ready" if _model is not None else ("error" if _model_error else "not_loaded"),
            "error": _model_error,
        },
        "diarization": {
            "enabled": DIARIZATION_ENABLED,
            "model": DIARIZATION_MODEL,
            "status": "ready" if _diarization_pipeline is not None else ("error" if _diarization_error else "not_loaded"),
            "error": _diarization_error,
        },
    }

def resolve_suffix_name(filename: str, content_type: str = "", default: str = ".m4a"):
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_UPLOADS: return suffix
    mapped = MIME_SUFFIXES.get((content_type or "").split(";",1)[0].strip().lower())
    if mapped: return mapped
    return default if not suffix else suffix

def resolve_suffix(file: UploadFile, default: str = ".m4a"):
    return resolve_suffix_name(file.filename or "", file.content_type or "", default)

def tokenize(text):
    return [w for w in re.findall(r"[a-záéíóúñü]{4,}", text.lower()) if w not in STOPWORDS]

def build_summary(segments):
    if not segments: return "No se detectó contenido suficiente para generar un resumen."
    freq = Counter(tokenize(" ".join(s["text"] for s in segments)))
    scored=[]
    for i,s in enumerate(segments):
        words=tokenize(s["text"]); scored.append((sum(freq[w] for w in words)/max(len(words),1),i,s["text"]))
    return " ".join(x[2] for x in sorted(sorted(scored,reverse=True)[:min(5,len(scored))],key=lambda x:x[1]))

def extract_topics(text, limit=6):
    words=tokenize(text)
    return [w.capitalize() for w,_ in Counter(words).most_common(limit)] if words else []

def extract_agreements(segments):
    patterns=[r"\b(hay que|tenemos que|tengo que|debe|debemos|deberíamos|necesitamos|voy a|me encargo|acordamos|queda pendiente|vamos a)\b"]
    due_re=re.compile(r"\b(hoy|mañana|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|esta semana|próxima semana|proxima semana|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b",re.I)
    out=[]
    for s in segments:
        txt=s["text"].strip()
        if any(re.search(p,txt,re.I) for p in patterns):
            due=due_re.search(txt); out.append({"text":txt,"owner":s.get("speaker","Hablante 1"),"due":due.group(0) if due else "Sin fecha detectada"})
    return out[:12]

async def save_upload(file: UploadFile, target: Path, max_bytes: int):
    written=0
    try:
        with target.open("wb") as out:
            while True:
                chunk=await file.read(4*1024*1024)
                if not chunk: break
                written += len(chunk)
                if written > max_bytes: raise HTTPException(status_code=413, detail=f"Archivo demasiado grande. Máximo permitido: {max_bytes//1024//1024} MB")
                out.write(chunk)
        return written
    except Exception:
        target.unlink(missing_ok=True)
        raise

def transcribe_file(path: Path):
    model = get_model()
    raw_segments, info = model.transcribe(
        str(path),
        language="es",
        vad_filter=VAD_ENABLED,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    segments=[]
    for seg in raw_segments:
        text=seg.text.strip()
        if text:
            segments.append({"start":round(seg.start,2),"end":round(seg.end,2),"text":text,"speaker":"Hablante 1"})
    if DIARIZATION_ENABLED and segments:
        segments = apply_diarization(path, segments)
    return segments, info

def apply_diarization(path: Path, segments):
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return segments
    output = pipeline(str(path))
    annotation = getattr(output, "speaker_diarization", output)
    turns=[]
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    if not turns:
        return segments
    speaker_map={}
    for seg in segments:
        best_speaker=None
        best_overlap=0.0
        for start,end,speaker in turns:
            overlap=max(0.0,min(seg["end"],end)-max(seg["start"],start))
            if overlap>best_overlap:
                best_overlap=overlap
                best_speaker=speaker
        if best_speaker is not None:
            if best_speaker not in speaker_map:
                speaker_map[best_speaker]=f"Hablante {len(speaker_map)+1}"
            seg["speaker"]=speaker_map[best_speaker]
    return segments

def speaker_count(segments):
    return len({s.get("speaker") for s in segments if s.get("speaker")}) or 1

def set_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS: JOBS[job_id].update(values)

def process_job(job_id: str, path: Path, filename: str):
    try:
        set_job(job_id,status="processing",progress=8,message="Preparando motor de transcripción…")
        segments, info = transcribe_file(path)
        set_job(job_id,progress=88,message="Generando resumen y acuerdos…")
        full_text=" ".join(s["text"] for s in segments)
        result={"status":"completed","filename":filename,"language":getattr(info,"language","es"),"language_probability":round(getattr(info,"language_probability",0),4),"duration":round(getattr(info,"duration",0),2),"speakers":speaker_count(segments),"segments":segments,"transcript":full_text,"summary":build_summary(segments),"topics":extract_topics(full_text),"agreements":extract_agreements(segments)}
        set_job(job_id,status="completed",progress=100,message="Listo",result=result)
    except Exception as exc:
        set_job(job_id,status="failed",message=f"Error procesando audio: {exc}",error=str(exc))
    finally:
        path.unlink(missing_ok=True)

@app.post("/uploads/init")
async def upload_init(payload: dict):
    filename=str(payload.get("filename") or "audio.m4a")
    content_type=str(payload.get("content_type") or "")
    size=int(payload.get("size") or 0)
    if size <= 0: raise HTTPException(status_code=400, detail="Tamaño de archivo inválido")
    if size > MAX_UPLOAD_BYTES: raise HTTPException(status_code=413, detail=f"Archivo demasiado grande. Máximo: {MAX_UPLOAD_MB} MB")
    suffix=resolve_suffix_name(filename,content_type)
    if suffix not in ALLOWED_UPLOADS-{".webm"}: raise HTTPException(status_code=400, detail=f"Formato no soportado ({suffix})")
    job_id=uuid.uuid4().hex
    target=UPLOADS/f"job-{job_id}{suffix}"
    target.touch()
    with JOBS_LOCK:
        JOBS[job_id]={"id":job_id,"filename":filename,"path":str(target),"size":size,"received":0,"status":"uploading","progress":0,"message":"Subiendo archivo…","created":time.time()}
    return {"job_id":job_id,"chunk_size":MAX_CHUNK_BYTES,"max_upload_mb":MAX_UPLOAD_MB}

@app.post("/uploads/{job_id}/chunk")
async def upload_chunk(job_id: str, file: UploadFile=File(...)):
    with JOBS_LOCK: job=JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="Carga no encontrada")
    data=await file.read(MAX_CHUNK_BYTES+1)
    if len(data)>MAX_CHUNK_BYTES: raise HTTPException(status_code=413, detail=f"Bloque demasiado grande. Máximo: {MAX_CHUNK_MB} MB")
    target=Path(job["path"])
    with target.open("ab") as out: out.write(data)
    received=job.get("received",0)+len(data)
    if received>job["size"]+1024: raise HTTPException(status_code=400, detail="Se recibieron más datos que el tamaño declarado")
    pct=min(70, int((received/max(job["size"],1))*70))
    set_job(job_id,received=received,progress=pct,message=f"Subiendo… {pct}%")
    return {"ok":True,"received":received,"progress":pct}

@app.post("/uploads/{job_id}/complete")
async def upload_complete(job_id: str, background_tasks: BackgroundTasks):
    with JOBS_LOCK: job=JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="Carga no encontrada")
    target=Path(job["path"])
    actual=target.stat().st_size if target.exists() else 0
    if actual!=job["size"]: raise HTTPException(status_code=400, detail=f"Carga incompleta: {actual} de {job['size']} bytes")
    set_job(job_id,status="queued",progress=72,message="Archivo recibido. Iniciando transcripción…")
    background_tasks.add_task(process_job, job_id, target, job["filename"])
    return {"ok":True,"job_id":job_id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK: job=JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return {k:v for k,v in job.items() if k!="path"}

@app.post("/transcribe-live")
async def transcribe_live(file: UploadFile=File(...)):
    suffix=resolve_suffix(file,".webm")
    if suffix not in ALLOWED_UPLOADS: suffix=".webm"
    target=UPLOADS/f"live-{uuid.uuid4()}{suffix}"
    try:
        await save_upload(file,target,MAX_LIVE_CHUNK_BYTES)
        # En vivo prioriza latencia: no ejecuta diarización pesada por fragmento.
        global DIARIZATION_ENABLED
        diarization_state = DIARIZATION_ENABLED
        DIARIZATION_ENABLED = False
        try:
            segments,info=transcribe_file(target)
        finally:
            DIARIZATION_ENABLED = diarization_state
        return {"status":"completed","language":getattr(info,"language","es"),"text":" ".join(s["text"] for s in segments),"segments":segments}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=503,detail=f"Servicio de transcripción no disponible: {exc}")
    finally: target.unlink(missing_ok=True)

@app.post("/analyze")
async def analyze(file: UploadFile=File(...)):
    suffix=resolve_suffix(file)
    if suffix not in ALLOWED_UPLOADS-{".webm"}: raise HTTPException(status_code=400,detail=f"Formato no soportado ({suffix})")
    target=UPLOADS/f"{uuid.uuid4()}{suffix}"
    try:
        await save_upload(file,target,MAX_UPLOAD_BYTES)
        segments,info=transcribe_file(target)
        full_text=" ".join(s["text"] for s in segments)
        return {"status":"completed","filename":file.filename,"language":getattr(info,"language","es"),"language_probability":round(getattr(info,"language_probability",0),4),"duration":round(getattr(info,"duration",0),2),"speakers":speaker_count(segments),"segments":segments,"transcript":full_text,"summary":build_summary(segments),"topics":extract_topics(full_text),"agreements":extract_agreements(segments)}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=503,detail=f"Servicio de transcripción no disponible: {exc}")
    finally: target.unlink(missing_ok=True)
