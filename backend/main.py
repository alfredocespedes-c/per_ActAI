from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pathlib import Path
from collections import Counter
import os
import re
import uuid
import math
import av
import numpy as np

app = FastAPI(title="ActaAI Audio API", version="0.4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_LIVE_CHUNK_MB = int(os.getenv("MAX_LIVE_CHUNK_MB", "40"))
MAX_LIVE_CHUNK_BYTES = MAX_LIVE_CHUNK_MB * 1024 * 1024
ALLOWED_UPLOADS = {
    ".mp3", ".wav", ".m4a", ".m4b", ".mp4", ".mp4a", ".aac", ".caf",
    ".ogg", ".flac", ".webm", ".3gp", ".3g2", ".mov"
}
MIME_SUFFIXES = {
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/x-aac": ".aac",
    "audio/x-caf": ".caf",
    "audio/caf": ".caf",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
}
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)

STOPWORDS = {
    "para","como","pero","porque","cuando","donde","desde","hasta","sobre","entre","este","esta","estos","estas",
    "esto","eso","esa","ese","aqui","ahi","muy","mas","menos","tambien","entonces","bueno","bien","vamos","tiene",
    "tener","hacer","hace","hecho","hay","que","del","las","los","una","uno","unos","unas","por","con","sin","y",
    "o","de","la","el","en","un","al","se","es","lo","le","me","te","nos","su","sus","mi","mis","ya","si",
}

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "actaai-audio",
        "version": "0.4.1",
        "max_upload_mb": MAX_UPLOAD_MB,
        "live_transcription": True,
        "apple_mpeg4_audio": True,
    }


def resolve_suffix(file: UploadFile, default: str = ".m4a"):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in ALLOWED_UPLOADS:
        return suffix
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    mapped = MIME_SUFFIXES.get(content_type)
    if mapped:
        return mapped
    return default if not suffix else suffix


def decode_audio(path: str, sample_rate: int = 16000):
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            arr = out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
            chunks.append(arr)
    for out in resampler.resample(None):
        arr = out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
        chunks.append(arr)
    return np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)


def voice_features(samples, sr=16000):
    if len(samples) < sr // 4:
        return np.zeros(12, dtype=np.float32)
    x = samples.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x) + 1e-8
    x = x / std
    n = min(len(x), sr * 8)
    x = x[:n]
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win)) + 1e-8
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    power = spec ** 2
    total = power.sum() + 1e-8
    centroid = float((freqs * power).sum() / total) / (sr / 2)
    bandwidth = math.sqrt(float((((freqs - centroid * (sr / 2)) ** 2) * power).sum() / total)) / (sr / 2)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    energy = float(np.log10(np.mean(x ** 2) + 1e-8))
    bands = np.array_split(power, 8)
    band_energy = [float(np.log10(b.sum() / total + 1e-8)) for b in bands]
    return np.array([centroid, bandwidth, zcr, energy, *band_energy], dtype=np.float32)


def kmeans(features, k, iters=40):
    x = np.asarray(features, dtype=np.float32)
    if len(x) <= k:
        return np.arange(len(x))
    seeds = np.linspace(0, len(x) - 1, k, dtype=int)
    centers = x[seeds].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            pts = x[labels == i]
            if len(pts):
                centers[i] = pts.mean(axis=0)
    return labels


def cluster_score(features, labels):
    x = np.asarray(features, dtype=np.float32)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1
    vals = []
    for i, p in enumerate(x):
        own = x[labels == labels[i]]
        a = float(np.mean(np.linalg.norm(own - p, axis=1))) if len(own) > 1 else 0.0
        b = min(float(np.mean(np.linalg.norm(x[labels == c] - p, axis=1))) for c in uniq if c != labels[i])
        vals.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(vals))


def estimate_speakers(path, segments):
    if len(segments) < 3:
        return [0] * len(segments)
    try:
        audio = decode_audio(path)
        feats = []
        for seg in segments:
            a = max(0, int(seg["start"] * 16000))
            b = min(len(audio), int(seg["end"] * 16000))
            feats.append(voice_features(audio[a:b]))
        x = np.asarray(feats)
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
        x = (x - mu) / sd
        best_labels = np.zeros(len(x), dtype=int)
        best_score = -1
        max_k = min(4, max(2, len(x) // 3))
        for k in range(2, max_k + 1):
            labels = kmeans(x, k)
            score = cluster_score(x, labels)
            if score > best_score:
                best_score, best_labels = score, labels
        if best_score < 0.08:
            return [0] * len(x)
        remap, nxt = {}, 0
        out = []
        for lab in best_labels.tolist():
            if lab not in remap:
                remap[lab] = nxt
                nxt += 1
            out.append(remap[lab])
        return out
    except Exception:
        return [0] * len(segments)


def tokenize(text):
    return [w for w in re.findall(r"[a-záéíóúñü]{4,}", text.lower()) if w not in STOPWORDS]


def build_summary(segments):
    if not segments:
        return "No se detectó contenido suficiente para generar un resumen."
    freq = Counter(tokenize(" ".join(s["text"] for s in segments)))
    scored = []
    for i, s in enumerate(segments):
        words = tokenize(s["text"])
        score = sum(freq[w] for w in words) / max(len(words), 1)
        scored.append((score, i, s["text"]))
    chosen = sorted(sorted(scored, reverse=True)[: min(5, len(scored))], key=lambda x: x[1])
    return " ".join(x[2] for x in chosen)


def extract_topics(text, limit=6):
    words = tokenize(text)
    if not words:
        return []
    counts = Counter(words)
    return [w.capitalize() for w, _ in counts.most_common(limit)]


def extract_agreements(segments):
    patterns = [r"\b(hay que|tenemos que|tengo que|debe|debemos|deberíamos|necesitamos|voy a|me encargo|acordamos|queda pendiente|vamos a)\b"]
    due_re = re.compile(r"\b(hoy|mañana|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|esta semana|próxima semana|proxima semana|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", re.I)
    out = []
    for s in segments:
        txt = s["text"].strip()
        if any(re.search(p, txt, re.I) for p in patterns):
            due = due_re.search(txt)
            out.append({"text": txt, "owner": s["speaker"], "due": due.group(0) if due else "Sin fecha detectada"})
    return out[:12]


async def save_upload(file: UploadFile, target: Path, max_bytes: int):
    written = 0
    with target.open("wb") as out:
        while True:
            chunk = await file.read(8 * 1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(status_code=413, detail=f"Archivo demasiado grande. Máximo permitido: {max_bytes // 1024 // 1024} MB")
            out.write(chunk)
    return written


def transcribe_file(path: Path, with_speakers: bool = False):
    raw_segments, info = model.transcribe(str(path), language="es", vad_filter=True, beam_size=1)
    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
    labels = estimate_speakers(str(path), segments) if with_speakers else [0] * len(segments)
    for seg, label in zip(segments, labels):
        seg["speaker"] = f"Hablante {label + 1}"
    full_text = " ".join(s["text"] for s in segments)
    return segments, info, labels, full_text


@app.post("/transcribe-live")
async def transcribe_live(file: UploadFile = File(...)):
    suffix = resolve_suffix(file, ".webm")
    if suffix not in ALLOWED_UPLOADS:
        suffix = ".webm"
    target = UPLOADS / f"live-{uuid.uuid4()}{suffix}"
    try:
        await save_upload(file, target, MAX_LIVE_CHUNK_BYTES)
        segments, info, _, full_text = transcribe_file(target, with_speakers=False)
        return {
            "status": "completed",
            "language": info.language,
            "text": full_text,
            "segments": segments,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error transcribiendo audio en vivo: {exc}")
    finally:
        target.unlink(missing_ok=True)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = resolve_suffix(file)
    if suffix not in ALLOWED_UPLOADS - {".webm"}:
        content_type = file.content_type or "desconocido"
        raise HTTPException(status_code=400, detail=f"Formato no soportado ({suffix or 'sin extensión'}, {content_type})")
    target = UPLOADS / f"{uuid.uuid4()}{suffix}"
    try:
        await save_upload(file, target, MAX_UPLOAD_BYTES)
        segments, info, labels, full_text = transcribe_file(target, with_speakers=True)
        return {
            "status": "completed",
            "filename": file.filename,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
            "speakers": len(set(labels)) if labels else 0,
            "segments": segments,
            "transcript": full_text,
            "summary": build_summary(segments),
            "topics": extract_topics(full_text),
            "agreements": extract_agreements(segments),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {exc}")
    finally:
        target.unlink(missing_ok=True)
