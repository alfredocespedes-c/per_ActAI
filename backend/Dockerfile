FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY backend /app
RUN mkdir -p /app/uploads /models/huggingface /models/cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000","--workers","1"]
