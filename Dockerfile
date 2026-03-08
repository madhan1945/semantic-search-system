# ============================================================
# Semantic Search System — Dockerfile
#
# Build:  docker build -t semantic-search .
# Run:    docker run -p 8000:8000 semantic-search
#
# For development with auto-reload:
#   docker run -p 8000:8000 -e MAX_DOCS=500 semantic-search
# ============================================================

FROM python:3.11-slim

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ───────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ─────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── NLTK data ───────────────────────────────────────────────
RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True)"

# ── Application source ──────────────────────────────────────
COPY . .

# ── Runtime configuration ───────────────────────────────────
# MAX_DOCS: limit dataset size for faster startup (remove for full corpus)
# SIMILARITY_THRESHOLD: cache hit sensitivity (default 0.85)
# REBUILD_INDEX: set to 'true' to force re-indexing on startup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAX_DOCS="" \
    SIMILARITY_THRESHOLD=0.85 \
    REBUILD_INDEX=false

# ── Expose API port ─────────────────────────────────────────
EXPOSE 8000

# ── Health check ────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ──────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
