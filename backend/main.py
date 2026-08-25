from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pathlib import Path
from collections import Counter
import os, re, uuid, time, threading

app = FastAPI(title="ActaAI Audio API", version="0.6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CHUNK_MB = int(os.getenv("MAX_CHUNK_MB", "12"))
MAX_CHUNK_BYTES = MAX_CHUNK_MB * 1024 * 1024
MAX_LIVE_CHUNK_MB = int(os.getenv("MAX_LIVE_CHUNK_MB", "40"))
MAX_LIVE_CHUNK_BYTES = MAX_LIVE_CHUNK_MB * 1024 * 1024
ALLOWED_UPLOADS = {".mp3", ".wav", ".m4a", ".m4b", ".mp4", ".mp4a", ".aac", ".caf", ".ogg", ".flac", ".webm", ".3gp", ".3g2", ".mov"}
MIME_SUFFIXES = {"audio/mp4":".m4a","audio/x-m4a":".m4a","audio/m4a":".m4a","audio/aac":".aac","audio/x-aac":".aac","audio/x-caf":".caf","audio/caf":".caf","video/mp4":".mp4","video/quicktime":".mov","audio/mpeg":".mp3","audio/wav":".wav","audio/x-wav":".wav","audio/ogg":".ogg","audio/flac":".flac","audio/webm":".webm"}

# Whisper is intentionally lazy-loaded. This lets FastAPI and /health start even
# when model download/initialization is slow or fails on a small Render instance.
_model = None
_model_error = None
_model_lock = threading.Lock()
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
            _model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
            return _model
        except Exception as exc:
            _model_error = str(exc)
            raise RuntimeError(f"No se pudo iniciar Whisper ({MODEL_NAME}): {exc}") from exc

@app.get("/health")
def health():
    return {"ok":True,"service":"actaai-audio","version":"0.6.0","max_upload_mb":MAX_UPLOAD_MB,"max_chunk_mb":MAX_CHUNK_MB,"live_transcription":True,"chunked_uploads":True,"apple_mpeg4_audio":True,"whisper":{"model":MODEL_NAME,"compute_type":COMPUTE_TYPE,"status":"ready" if _model is not None else ("error" if _model_error else "not_loaded"),"error":_model_error}}

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
    raw_segments, info = model.transcribe(str(path), language="es", vad_filter=True, beam_size=1)
    segments=[]
    for seg in raw_segments:
        text=seg.text.strip()
        if text: segments.append({"start":round(seg.start,2),"end":round(seg.end,2),"text":text,"speaker":"Hablante 1"})
    return segments, info

def set_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS: JOBS[job_id].update(values)

def process_job(job_id: str, path: Path, filename: str):
    try:
        set_job(job_id,status="processing",progress=8,message="Preparando motor de transcripción…")
        segments, info = transcribe_file(path)
        set_job(job_id,progress=88,message="Generando resumen y acuerdos…")
        full_text=" ".join(s["text"] for s in segments)
        result={"status":"completed","filename":filename,"language":getattr(info,"language","es"),"language_probability":round(getattr(info,"language_probability",0),4),"duration":round(getattr(info,"duration",0),2),"speakers":1,"segments":segments,"transcript":full_text,"summary":build_summary(segments),"topics":extract_topics(full_text),"agreements":extract_agreements(segments)}
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
        segments,info=transcribe_file(target)
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
        return {"status":"completed","filename":file.filename,"language":getattr(info,"language","es"),"language_probability":round(getattr(info,"language_probability",0),4),"duration":round(getattr(info,"duration",0),2),"speakers":1,"segments":segments,"transcript":full_text,"summary":build_summary(segments),"topics":extract_topics(full_text),"agreements":extract_agreements(segments)}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=503,detail=f"Servicio de transcripción no disponible: {exc}")
    finally: target.unlink(missing_ok=True)
