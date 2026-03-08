# Semantic Search System

A production-ready AI-powered semantic search system built on the 20 Newsgroups dataset (~20,000 documents, 20 categories). Uses sentence-transformer embeddings, FAISS vector database, Gaussian Mixture Model fuzzy clustering, and a custom-built semantic cache that recognises paraphrased queries and delivers 10x faster responses on cache hits. Exposed via a FastAPI REST API with full Swagger documentation, 35+ unit tests, and Docker support.
---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Design Decisions](#design-decisions)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Docker](#docker)
- [API Reference](#api-reference)
- [Cache Experiments](#cache-experiments)
- [Testing](#testing)
- [Configuration](#configuration)

---

## Project Overview

This system takes natural-language queries and retrieves semantically relevant newsgroup documents.  The core pipeline is:

```
dataset → preprocessing → embeddings → vector DB → fuzzy clustering → semantic cache → FastAPI
```

The distinguishing engineering choices are:

| Component | Technology | Reason |
|-----------|-----------|--------|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, 5× faster than mpnet, ~95% quality |
| Vector store | FAISS (IndexFlatIP) | Exact cosine search, battle-tested, serialisable |
| Clustering | Gaussian Mixture Model | Soft membership, principled BIC model selection |
| Cache | Pure-Python in-memory | Zero external deps, cluster-bucketed O(n/k) lookup |
| API | FastAPI + Pydantic v2 | Async, auto-docs, validated schemas |

---

## System Architecture

```
┌─────────────┐     clean      ┌─────────────┐    encode     ┌──────────────────┐
│   20 News   │ ─────────────► │ TextCleaner │ ────────────► │ EmbeddingModel   │
│  Dataset    │                └─────────────┘               │ (MiniLM-L6-v2)   │
└─────────────┘                                              └────────┬─────────┘
                                                                      │ 384-dim vectors
                                                         ┌────────────┴──────────┐
                                                         │                       │
                                                   ┌─────▼──────┐     ┌─────────▼──────┐
                                                   │  FAISS DB  │     │ FuzzyClusterer │
                                                   │ (vector    │     │  (GMM + PCA)   │
                                                   │  search)   │     │                │
                                                   └─────┬──────┘     └────────┬───────┘
                                                         │                     │
                                                         │        cluster distributions
                                                         │                     │
                                                   ┌─────▼─────────────────────▼──────┐
                                                   │         SemanticCache            │
                                                   │   cluster-bucketed lookup        │
                                                   │   cosine similarity ≥ threshold  │
                                                   └──────────────┬───────────────────┘
                                                                  │
                                                         ┌────────▼────────┐
                                                         │   FastAPI App   │
                                                         │  POST /query    │
                                                         │  GET  /cache/.. │
                                                         └─────────────────┘
```

### Query lifecycle

```
POST /query {"query": "What is gun control?"}
    │
    ├── embed query (all-MiniLM-L6-v2)
    ├── get cluster distribution (GMM)
    ├── look up semantic cache
    │       ├── HIT  → return cached result  ← fast path (~5–20 ms)
    │       └── MISS → FAISS search          ← compute path (~50–200 ms)
    │                  store result in cache
    └── return JSON response
```

---

## Design Decisions

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2` was chosen for:

- **Speed**: 22 M parameters → ~1 ms per sentence on CPU
- **Quality**: Trained on 1B+ sentence pairs; strong on informal text
- **Dimension**: 384-dim keeps FAISS index under 30 MB for 20 k docs
- **Normalisation**: L2-normalised output → dot product = cosine similarity

### Fuzzy Clustering (GMM)

Hard clustering (K-Means) was rejected because newsgroup posts span multiple topics.  GMM was preferred over Fuzzy C-Means because:

- BIC/AIC provide principled model selection for n_components
- Diagonal covariance balances expressivity and memory at 384 dims
- PCA reduction (384 → 50) mitigates the curse of dimensionality

The number of clusters is **not hardcoded** — BIC is minimised over candidates `[5, 8, 10, 12, 15, 20, 25, 30]`.

### Semantic Cache

The cache recognises that *"What is gun control policy?"* and *"Explain firearm regulation laws"* should return the same result.

Key design choices:

- **Cluster-bucketed**: entries are stored per cluster, reducing lookup from O(n) to O(n/k)
- **LRU eviction**: bounded memory growth via `max_entries` cap
- **Thread-safe**: `RLock` guards all read/write operations
- **Configurable threshold**: `SIMILARITY_THRESHOLD=0.85` is the default operating point

---

## Repository Structure

```
semantic-search-system/
│
├── data/
│   └── dataset_loader.py       # 20 Newsgroups loader → clean DataFrame
│
├── preprocessing/
│   └── clean_text.py           # Multi-stage text cleaning pipeline
│
├── embeddings/
│   └── embedding_model.py      # SentenceTransformer wrapper
│
├── vector_store/
│   └── vector_db.py            # FAISS index + metadata store
│
├── clustering/
│   └── fuzzy_clustering.py     # GMM fuzzy clusterer with BIC selection
│
├── cache/
│   └── semantic_cache.py       # Pure-Python semantic cache
│
├── services/
│   └── query_service.py        # Orchestrates full query pipeline
│
├── api/
│   └── main.py                 # FastAPI application
│
├── experiments/
│   └── threshold_analysis.py   # Cache threshold experiments
│
├── tests/
│   └── test_all.py             # pytest test suite
│
├── requirements.txt
├── README.md
├── Dockerfile
├── docker-compose.yml
└── run.sh
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
git clone <repo-url>
cd semantic-search-system

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the API

### Option 1 — run.sh (recommended)

```bash
chmod +x run.sh
./run.sh
```

For development (fast startup with 500 documents):

```bash
MAX_DOCS=500 ./run.sh
```

### Option 2 — uvicorn directly

```bash
uvicorn api.main:app --reload
```

### Option 3 — environment variable configuration

```bash
export MAX_DOCS=1000
export SIMILARITY_THRESHOLD=0.85
export REBUILD_INDEX=false
uvicorn api.main:app --reload
```

On first startup, the system will:

1. Download the 20 Newsgroups dataset (~17 MB, cached locally)
2. Clean all documents
3. Generate embeddings (a few minutes on CPU for the full corpus)
4. Fit the GMM clusterer with BIC-guided component selection
5. Index everything into FAISS and persist to disk

**Subsequent startups load from disk in seconds.**

---

## Docker

### Build and run

```bash
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```

For a faster first run (smaller corpus):

```bash
docker run -p 8000:8000 -e MAX_DOCS=500 semantic-search
```

### Docker Compose

```bash
docker-compose up --build
```

The compose file mounts a named volume so the index persists between restarts.

---

## API Reference

Interactive docs available at `http://localhost:8000/docs`.

---

### `POST /query`

Execute a semantic search query.

**Request:**
```json
{
  "query": "Explain gun control policies"
}
```

**Response (cache miss):**
```json
{
  "query": "Explain gun control policies",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "doc_id": 4821,
      "score": 0.8934,
      "category": "talk.politics.guns",
      "text_snippet": "firearm regulation state law amendment...",
      "cluster_distribution": {"cluster_3": 0.62, "cluster_7": 0.28}
    }
  ],
  "dominant_cluster": 3,
  "latency_ms": 87.4
}
```

**Response (cache hit — similar query was previously issued):**
```json
{
  "query": "What is gun control?",
  "cache_hit": true,
  "matched_query": "Explain gun control policies",
  "similarity_score": 0.913,
  "result": [ ... ],
  "dominant_cluster": 3,
  "latency_ms": 6.2
}
```

**curl example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain gun control policies"}'
```

---

### `GET /cache/stats`

Returns cache performance metrics.

**Response:**
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

```bash
curl http://localhost:8000/cache/stats
```

---

### `DELETE /cache`

Flush all cache entries and reset counters.

**Response:**
```json
{
  "message": "Cache flushed successfully.",
  "stats": {
    "total_entries": 0,
    "hit_count": 0,
    "miss_count": 0,
    "hit_rate": 0.0
  }
}
```

```bash
curl -X DELETE http://localhost:8000/cache
```

---

### `GET /health`

Liveness probe.

```bash
curl http://localhost:8000/health
```

---

## Cache Experiments

Run all experiments:

```bash
python experiments/threshold_analysis.py
```

Results are saved to `data/experiments/`.

### Experiment 1: Threshold sweep

We vary the similarity threshold from 0.50 to 1.00 and measure hit rate.

| Threshold | Behaviour |
|-----------|-----------|
| 0.50–0.70 | High hit rate; risk of returning loosely related results |
| **0.85** | **Default — semantically similar queries match; unrelated do not** |
| 0.95–1.00 | Near-exact match only; very low hit rate; little caching benefit |

**Key insight:** At threshold 0.85, the pairs *"What is gun control?"* and *"Explain firearm regulation laws"* match (cosine sim ≈ 0.91), while *"gun control"* and *"space exploration"* do not (cosine sim ≈ 0.12).

### Experiment 2: Latency

| Path | Mean latency |
|------|-------------|
| Cache HIT | ~5–20 ms |
| Cache MISS (FAISS search) | ~50–200 ms |

Cache hits are roughly 10× faster because they skip the vector DB search.

### Experiment 3: Cluster BIC / AIC

The GMM logs BIC and AIC for each candidate n_components.  The selected n minimises BIC, which penalises complexity and prevents overfitting.  Typical selection for the full 20 Newsgroups corpus is n_components ≈ 15–20.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short

# Run a specific test class
pytest tests/test_all.py::TestSemanticCache -v
```

The test suite covers:

- `TestTextCleaner` — 10 unit tests for each cleaning step
- `TestEmbeddingModel` — vector shape, normalisation, similarity ordering
- `TestVectorDatabase` — insert, batch insert, search, empty search
- `TestFuzzyClusterer` — fit, predict_proba shape, probability normalisation
- `TestSemanticCache` — miss, hit, threshold behaviour, stats, flush
- `TestDatasetLoader` — DataFrame structure, category counts
- `TestIntegration` — end-to-end cache miss → store → hit cycle

---

## Configuration

All parameters can be set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `0.85` | Cache hit cosine similarity cutoff |
| `TOP_K` | `10` | Number of FAISS results per query |
| `REBUILD_INDEX` | `false` | Force re-indexing even if saved index exists |
| `MAX_DOCS` | *(full corpus)* | Limit corpus size (useful during development) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Override embedding model |

Create a `.env` file in the project root to persist these:

```dotenv
SIMILARITY_THRESHOLD=0.85
MAX_DOCS=2000
REBUILD_INDEX=false
```

---

## License

MIT © madhan1945
