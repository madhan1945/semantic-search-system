"""
semantic_cache.py
-----------------
A pure-Python in-memory semantic cache with optional disk persistence.

No Redis, Memcached, or external caching libraries are used.
All data structures are standard Python dicts and lists.

Cache Architecture
------------------
The cache is organised as a two-level structure:

  Level 1 — Cluster buckets
      Cache entries are grouped by their dominant cluster.
      When a new query arrives, we only search entries in the same cluster
      bucket(s) as the query, reducing comparison work from O(n) to O(n/k)
      where k is the number of clusters.

  Level 2 — Similarity check
      Within a bucket, we compute cosine similarity between the incoming
      query embedding and each stored query embedding.  If similarity
      exceeds SIMILARITY_THRESHOLD, we return the cached result.

Why this design?
  - Simple: no external dependencies, fully introspectable.
  - Fast: cluster pre-filtering avoids full linear scans.
  - Tunable: SIMILARITY_THRESHOLD controls precision/recall tradeoff.

SIMILARITY_THRESHOLD (default 0.85):
  - 1.0 → exact match only (extremely low hit rate, useless cache)
  - 0.85 → semantically similar queries match (recommended operating point)
  - 0.70 → broad matching; may return results for loosely related queries
  - See experiments/threshold_analysis.py for empirical evidence.

Eviction Policy:
  LRU (Least Recently Used) with a configurable max_entries cap.
  When the cache is full, the oldest-accessed entry is evicted.
  This prevents unbounded memory growth in production.
"""

import json
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default similarity threshold — configurable via environment variable
import os
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

_DEFAULT_PERSIST = Path(__file__).parent.parent / "data" / "cache"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """
    A single cached query-result pair with associated metadata.

    Fields
    ------
    query_text        : original query string
    query_embedding   : float32 vector used for similarity lookup
    result            : the full search result to return on cache hit
    cluster_membership: fuzzy cluster distribution for the query
    timestamp         : unix timestamp of insertion
    last_accessed     : unix timestamp of most recent cache hit
    hit_count         : how many times this entry has been returned
    dominant_cluster  : cluster with highest membership probability
    """
    query_text: str
    query_embedding: np.ndarray
    result: Any
    cluster_membership: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    dominant_cluster: int = 0

    def touch(self) -> None:
        """Update access time and increment hit counter."""
        self.last_accessed = time.time()
        self.hit_count += 1

    def to_dict(self) -> dict:
        """Serialisable representation (excludes numpy array)."""
        return {
            "query_text": self.query_text,
            "result": self.result,
            "cluster_membership": self.cluster_membership,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "hit_count": self.hit_count,
            "dominant_cluster": self.dominant_cluster,
        }


@dataclass
class CacheStats:
    """Tracks cache performance metrics."""
    total_entries: int = 0
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return round(self.hit_count / total, 4) if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
        }


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Cluster-bucketed semantic cache with LRU eviction.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity above which a cache hit is declared.
    max_entries : int
        Maximum number of cache entries before LRU eviction kicks in.
    persist_dir : Path or None
        Directory for optional disk persistence.  None → in-memory only.
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_entries: int = 10_000,
        persist_dir: Optional[Path] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.persist_dir = Path(persist_dir or _DEFAULT_PERSIST)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Cluster-bucketed storage: cluster_id → OrderedDict(entry_id → CacheEntry)
        # OrderedDict preserves insertion order for LRU eviction
        self._buckets: Dict[int, OrderedDict] = {}

        # Global flat index: entry_id → CacheEntry (for LRU tracking across buckets)
        self._global_index: OrderedDict = OrderedDict()

        # Thread safety — RLock allows re-entrant locking from the same thread
        self._lock: RLock = RLock()

        self._stats: CacheStats = CacheStats()
        self._entry_counter: int = 0

    # ------------------------------------------------------------------
    # Cache lookup (read path)
    # ------------------------------------------------------------------

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_membership: Dict[str, float],
    ) -> Tuple[bool, Optional[CacheEntry]]:
        """
        Search the cache for a semantically similar prior query.

        Algorithm
        ---------
        1. Identify which cluster buckets to search (top clusters of query).
        2. Compute cosine similarity to each entry in those buckets.
        3. Return the highest-scoring entry if its similarity ≥ threshold.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape (dim,).  Must be L2-normalised for dot-product == cosine.
        cluster_membership : dict
            Fuzzy cluster distribution of the incoming query.

        Returns
        -------
        (hit: bool, entry: CacheEntry or None)
        """
        with self._lock:
            # Identify relevant buckets (clusters present in query distribution)
            bucket_ids = self._get_bucket_ids(cluster_membership)

            best_score: float = -1.0
            best_entry: Optional[CacheEntry] = None

            for bucket_id in bucket_ids:
                bucket = self._buckets.get(bucket_id, {})
                for entry_id, entry in bucket.items():
                    # Cosine similarity: dot product of L2-normalised vectors
                    score = float(np.dot(query_embedding, entry.query_embedding))
                    if score > best_score:
                        best_score = score
                        best_entry = (entry_id, entry)

            if best_score >= self.similarity_threshold and best_entry is not None:
                entry_id, entry = best_entry
                entry.touch()
                # Move to end of LRU order (most recently used)
                self._global_index.move_to_end(entry_id)
                self._stats.hit_count += 1
                logger.debug(
                    "Cache HIT  | score=%.4f  query='%s'",
                    best_score,
                    entry.query_text[:60],
                )
                return True, entry

            self._stats.miss_count += 1
            logger.debug("Cache MISS | best_score=%.4f", best_score)
            return False, None

    # ------------------------------------------------------------------
    # Cache insertion (write path)
    # ------------------------------------------------------------------

    def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: Any,
        cluster_membership: Dict[str, float],
    ) -> int:
        """
        Insert a new entry into the cache.

        Parameters
        ----------
        query_text        : original query string
        query_embedding   : L2-normalised float32 embedding
        result            : search result to cache
        cluster_membership: fuzzy cluster distribution

        Returns
        -------
        int  entry_id of the new entry
        """
        with self._lock:
            # Evict LRU entry if at capacity
            if len(self._global_index) >= self.max_entries:
                self._evict_lru()

            dominant_cluster = self._dominant_cluster(cluster_membership)
            entry = CacheEntry(
                query_text=query_text,
                query_embedding=query_embedding.astype(np.float32),
                result=result,
                cluster_membership=cluster_membership,
                dominant_cluster=dominant_cluster,
            )

            entry_id = self._entry_counter
            self._entry_counter += 1

            # Insert into global LRU index
            self._global_index[entry_id] = entry

            # Insert into cluster buckets
            bucket_ids = self._get_bucket_ids(cluster_membership)
            for bucket_id in bucket_ids:
                if bucket_id not in self._buckets:
                    self._buckets[bucket_id] = OrderedDict()
                self._buckets[bucket_id][entry_id] = entry

            self._stats.total_entries = len(self._global_index)
            logger.debug(
                "Cache STORE | entry_id=%d  cluster=%d  query='%s'",
                entry_id,
                dominant_cluster,
                query_text[:60],
            )
            return entry_id

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Clear all cache entries and reset statistics."""
        with self._lock:
            self._buckets.clear()
            self._global_index.clear()
            self._entry_counter = 0
            self._stats = CacheStats()
        logger.info("Cache flushed.")

    @property
    def stats(self) -> CacheStats:
        self._stats.total_entries = len(self._global_index)
        return self._stats

    def entries(self) -> List[dict]:
        """Return all cache entries as serialisable dicts."""
        with self._lock:
            return [e.to_dict() for e in self._global_index.values()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist cache to disk."""
        path = self.persist_dir / "cache.pkl"
        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "global_index": self._global_index,
                        "buckets": self._buckets,
                        "stats": self._stats,
                        "counter": self._entry_counter,
                    },
                    f,
                )
        logger.info("Cache saved to %s (%d entries).", path, len(self._global_index))

    def load(self) -> bool:
        """Load cache from disk.  Returns True on success."""
        path = self.persist_dir / "cache.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._global_index = data["global_index"]
        self._buckets = data["buckets"]
        self._stats = data["stats"]
        self._entry_counter = data["counter"]
        logger.info("Cache loaded from %s (%d entries).", path, len(self._global_index))
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_bucket_ids(self, cluster_membership: Dict[str, float]) -> List[int]:
        """
        Parse cluster membership dict and return a list of integer bucket ids.
        We search all clusters in the distribution (not just the dominant one)
        to handle boundary documents that span multiple clusters.
        """
        ids = []
        for key in cluster_membership:
            try:
                ids.append(int(key.split("_")[-1]))
            except (ValueError, IndexError):
                pass
        return ids if ids else [0]

    def _dominant_cluster(self, cluster_membership: Dict[str, float]) -> int:
        """Return cluster id with highest membership probability."""
        if not cluster_membership:
            return 0
        best_key = max(cluster_membership, key=cluster_membership.get)
        try:
            return int(best_key.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    def _evict_lru(self) -> None:
        """
        Remove the least recently accessed entry from all data structures.
        Called internally when max_entries is reached.
        """
        if not self._global_index:
            return
        # First item in OrderedDict is LRU
        lru_id, lru_entry = next(iter(self._global_index.items()))
        del self._global_index[lru_id]

        # Remove from all cluster buckets
        bucket_ids = self._get_bucket_ids(lru_entry.cluster_membership)
        for bucket_id in bucket_ids:
            if bucket_id in self._buckets:
                self._buckets[bucket_id].pop(lru_id, None)
                if not self._buckets[bucket_id]:
                    del self._buckets[bucket_id]

        logger.debug("Evicted LRU entry id=%d", lru_id)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cache = SemanticCache(similarity_threshold=0.85)

    # Simulate two semantically similar queries
    rng = np.random.default_rng(0)
    base_vec = rng.normal(size=384).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)

    # Slightly perturbed version (will be similar but not identical)
    perturbed = base_vec + rng.normal(scale=0.05, size=384).astype(np.float32)
    perturbed /= np.linalg.norm(perturbed)

    cluster_dist = {"cluster_3": 0.62, "cluster_7": 0.27, "cluster_1": 0.11}

    cache.store(
        query_text="What is gun control policy?",
        query_embedding=base_vec,
        result={"docs": ["doc1", "doc2"]},
        cluster_membership=cluster_dist,
    )

    hit, entry = cache.lookup(perturbed, cluster_dist)
    print(f"Cache hit: {hit}")
    if hit:
        print(f"Matched query: '{entry.query_text}'")

    print(f"Stats: {cache.stats.to_dict()}")
