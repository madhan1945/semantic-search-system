#!/usr/bin/env bash
# ============================================================
# run.sh — Start the Semantic Search API
#
# Usage:
#   ./run.sh                    # default settings
#   MAX_DOCS=500 ./run.sh       # dev mode (small corpus)
#   REBUILD_INDEX=true ./run.sh # force re-indexing
# ============================================================

set -euo pipefail

# ── Colour helpers ───────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Python version check ─────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_MAJOR=3; REQUIRED_MINOR=10

IFS='.' read -r MAJOR MINOR PATCH <<< "$PYTHON_VERSION"
if [ "$MAJOR" -lt "$REQUIRED_MAJOR" ] || \
   ([ "$MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$MINOR" -lt "$REQUIRED_MINOR" ]); then
    err "Python ${REQUIRED_MAJOR}.${REQUIRED_MINOR}+ required (found ${PYTHON_VERSION})"
fi
log "Python ${PYTHON_VERSION} ✓"

# ── Dependency check ─────────────────────────────────────────
if ! python3 -c "import fastapi" 2>/dev/null; then
    warn "Dependencies not installed. Running pip install…"
    pip install -r requirements.txt
fi
log "Dependencies ✓"

# ── NLTK data ────────────────────────────────────────────────
python3 -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
" && log "NLTK data ✓"

# ── Environment defaults ─────────────────────────────────────
export SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.85}"
export TOP_K="${TOP_K:-10}"
export REBUILD_INDEX="${REBUILD_INDEX:-false}"
export MAX_DOCS="${MAX_DOCS:-}"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Semantic Search System${NC}"
echo -e "${BLUE}  http://localhost:8000${NC}"
echo -e "${BLUE}  Docs: http://localhost:8000/docs${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo ""
log "SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD}"
log "MAX_DOCS=${MAX_DOCS:-'(full corpus)'}"
log "REBUILD_INDEX=${REBUILD_INDEX}"
echo ""

# ── Launch ───────────────────────────────────────────────────
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
