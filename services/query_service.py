"""
query_service.py
----------------
Orchestrates the full query pipeline:

  query text
    → embed (EmbeddingModel)
    → lookup cluster distribution (FuzzyClusterer)
    → check semantic cache (SemanticCache)
    → on miss: search vector DB (VectorDatabase)
    → store result in cache
    → return structured response

This module is the single integration point used by the FastAPI layer.
It also handles one-time system initialisation (build vs. load path).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cache.semantic_cache import SemanticCache
from clustering.fuzzy_clustering import FuzzyClusterer
from data.dataset_loader import DatasetLoader
from embeddings.embedding_model import EmbeddingModel
from preprocessing.clean_text import TextCleaner
from vector_store.vector_db import VectorDatabase

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


class QueryService:
    """
    High-level service that coordinates all subsystems.

    Lifecycle
    ---------
    1. Call QueryService().
    2. Call .initialise() — loads or builds all components.
    3. Call .query(text) for search requests.

    Parameters
    ----------
    similarity_threshold : float
        Passed to SemanticCache.  Default 0.85.
    top_k : int
        Number of vector DB results to return per query.
    rebuild : bool
        If True, re-index the dataset even if a saved index exists.
        Defaults to False (loads from disk if available).
    max_docs : int or None
        Limit dataset to first N documents (useful for dev/testing).
        None → use full dataset.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        top_k: int = 10,
        rebuild: bool = False,
        max_docs: Optional[int] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.rebuild = rebuild
        self.max_docs = max_docs

        # Components — initialised lazily
        self._encoder: Optional[EmbeddingModel] = None
        self._cleaner: Optional[TextCleaner] = None
        self._vector_db: Optional[VectorDatabase] = None
        self._clusterer: Optional[FuzzyClusterer] = None
        self._cache: Optional[SemanticCache] = None
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """
        Load or build all subsystems.

        If persist files exist and rebuild=False, subsystems are loaded
        from disk (fast).  Otherwise the full pipeline is executed:
          load dataset → clean → embed → cluster → index → save.
        """
        logger.info("Initialising QueryService…")

        # Always instantiate lightweight components
        self._encoder = EmbeddingModel()
        self._cleaner = TextCleaner()
        self._cache = SemanticCache(similarity_threshold=self.similarity_threshold)
        self._vector_db = VectorDatabase(dim=self._encoder.dim)
        self._clusterer = FuzzyClusterer()

        # Try loading from disk first
        if not self.rebuild:
            db_loaded = self._vector_db.load()
            cl_loaded = self._clusterer.load()
            self._cache.load()

            if db_loaded and cl_loaded:
                logger.info("All components loaded from disk. System ready.")
                self._ready = True
                return

        # Full build path
        logger.info("Building index from scratch (this may take a few minutes)…")
        self._build_pipeline()
        self._ready = True
        logger.info("System initialisation complete.")

    def _build_pipeline(self) -> None:
        """Execute the full data → embed → cluster → index pipeline."""
        # 1. Load dataset
        logger.info("Loading dataset…")
        loader = DatasetLoader()
        df = loader.load()
        if self.max_docs:
            df = df.head(self.max_docs).copy()
            logger.info("Limiting to %d documents for development.", self.max_docs)

        # 2. Clean text
        logger.info("Cleaning %d documents…", len(df))
        df["clean_text"] = self._cleaner.clean_batch(
            df["text"].tolist(), show_progress=True
        )
        # Drop empty documents (too short after cleaning)
        original_count = len(df)
        df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
        logger.info(
            "Retained %d / %d documents after cleaning.", len(df), original_count
        )

        # 3. Generate embeddings
        logger.info("Generating embeddings…")
        embeddings = self._encoder.encode(
            df["clean_text"].tolist(), show_progress=True
        )
        logger.info("Embeddings shape: %s", embeddings.shape)

        # 4. Fuzzy clustering
        logger.info("Fitting fuzzy clusterer…")
        self._clusterer.fit(embeddings)
        self._clusterer.save()

        cluster_dists = self._clusterer.get_cluster_distributions(embeddings)

        # 5. Index into vector DB
        logger.info("Indexing documents into vector DB…")
        self._vector_db.add_batch(
            doc_ids=df["doc_id"].tolist(),
            texts=df["clean_text"].tolist(),
            embeddings=embeddings,
            categories=df["category"].tolist(),
            cluster_dists=cluster_dists,
        )
        self._vector_db.save()
        logger.info("Pipeline complete. %d documents indexed.", self._vector_db.size)

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural-language query.

        Returns
        -------
        dict with keys:
            query, cache_hit, matched_query, similarity_score,
            result, dominant_cluster, latency_ms
        """
        self._assert_ready()
        start = time.perf_counter()

        # Step 1: Embed the query
        query_embedding = self._encoder.encode_query(query_text)

        # Step 2: Get cluster distribution for the query
        cluster_dist = self._clusterer.get_document_distribution(query_embedding)
        dominant_cluster = self._clusterer.dominant_cluster(query_embedding)

        # Step 3: Check semantic cache
        cache_hit, cached_entry = self._cache.lookup(query_embedding, cluster_dist)

        if cache_hit:
            elapsed = (time.perf_counter() - start) * 1000
            sim = float(
                np.dot(query_embedding, cached_entry.query_embedding)
            )
            return {
                "query": query_text,
                "cache_hit": True,
                "matched_query": cached_entry.query_text,
                "similarity_score": round(sim, 4),
                "result": cached_entry.result,
                "dominant_cluster": cached_entry.dominant_cluster,
                "latency_ms": round(elapsed, 2),
            }

        # Step 4: Vector DB search (cache miss path)
        raw_results = self._vector_db.search(query_embedding, top_k=self.top_k)
        result = self._format_results(raw_results)

        # Step 5: Store in cache
        self._cache.store(
            query_text=query_text,
            query_embedding=query_embedding,
            result=result,
            cluster_membership=cluster_dist,
        )

        elapsed = (time.perf_counter() - start) * 1000
        return {
            "query": query_text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result,
            "dominant_cluster": int(dominant_cluster),
            "latency_ms": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        """Return current cache performance statistics."""
        self._assert_ready()
        return self._cache.stats.to_dict()

    def flush_cache(self) -> dict:
        """Clear the cache and return updated stats."""
        self._assert_ready()
        self._cache.flush()
        return self._cache.stats.to_dict()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_results(
        self, raw_results: List[tuple]
    ) -> List[Dict[str, Any]]:
        """
        Convert raw (score, metadata) tuples from the vector DB into a
        clean list of result dicts suitable for JSON serialisation.
        """
        formatted = []
        for score, meta in raw_results:
            formatted.append(
                {
                    "doc_id": meta["doc_id"],
                    "score": round(score, 4),
                    "category": meta["category"],
                    "text_snippet": meta["text"][:300],
                    "cluster_distribution": meta.get("cluster_dist", {}),
                }
            )
        return formatted

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "QueryService is not initialised.  Call .initialise() first."
            )

    @property
    def is_ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# CLI convenience — useful for smoke testing the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    svc = QueryService(max_docs=500)  # small subset for quick testing
    svc.initialise()

    queries = [
        "What is gun control policy?",
        "Explain firearm regulation laws",   # Should be a cache hit after 1st
        "How does NASA plan space missions?",
        "Tell me about space exploration",
    ]

    for q in queries:
        result = svc.query(q)
        print(
            f"\n{'HIT ' if result['cache_hit'] else 'MISS'} | "
            f"score={result['similarity_score']}  | "
            f"cluster={result['dominant_cluster']} | "
            f"{result['latency_ms']:.1f}ms | '{q}'"
        )

    print("\nCache stats:", svc.cache_stats())
