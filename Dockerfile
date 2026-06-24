# ── Stage 1: Builder ──────────────────────────────────────────────────
# Downloads ML models during BUILD TIME, not runtime.
# Models baked into image = zero download time on startup.
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (Docker layer caching)
# If requirements.txt doesn't change, this layer is cached
# and pip install is skipped on subsequent builds
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Pre-download models at BUILD time ────────────────────────────────
# This runs ONCE during docker build, not on every container start.
# Models are stored in /root/.cache/huggingface inside the image.
# Result: cold start goes from 4 minutes → 10 seconds.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import os

print('Downloading embedding model...')
SentenceTransformer('all-mpnet-base-v2')
print('Embedding model downloaded.')

print('Downloading reranker model...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Reranker model downloaded.')

print('All models cached successfully.')
PY

# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy pre-downloaded models from builder
# This is the key line — models travel with the image
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY . .

# Environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace offline mode — prevents any runtime downloads
# Models are already in /root/.cache/huggingface from builder
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface/sentence_transformers

# Port (Railway sets $PORT dynamically)
EXPOSE 8000

# Health check — Railway uses this to know when app is ready
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start command
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}