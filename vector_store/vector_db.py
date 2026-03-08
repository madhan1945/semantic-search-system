"""
vector_db.py
------------
FAISS-backed vector store with metadata persistence.

Design Decisions:
    FAISS (Facebook AI Similarity Search) was chosen over Chroma/Annoy because:
    1. It is battle-tested at billion-scale and handles 20 k docs trivially.
    2. IndexFlatIP (inner-product) == cosine similarity for L2-normalised vecs,
       giving exact nearest-neighbour results without approximation error.
    3. Serialisation to disk is a single faiss.write_index() call.
    4. Pure C++ backend → faster than Python-native alternatives at query time.

    Metadata (text, category, cluster distribution) is stored separately in a
    Python dict keyed by integer position (FAISS internal id) because FAISS
    itself only stores float32 vectors.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Default persistence paths
_DEFAULT_DIR = Path(__file__).parent.parent / "data" / "vector_store"


class VectorDatabase:
    """
    Stores document embeddings in FAISS and metadata in memory (persisted
    to disk as pickle).

    Stored per document
    -------------------
    - document_id   (int)   : original row index from the dataset DataFrame
    - text          (str)   : cleaned document text
    - embedding     (array) : float32 vector
    - category      (str)   : newsgroup category label
    - cluster_dist  (dict)  : fuzzy cluster membership probabilities

    Parameters
    ----------
    dim : int
        Embedding dimensionality.  Must match the encoder output.
    persist_dir : Path or str
        Directory for index + metadata files.  Created automatically.
    """

    def __init__(self, dim: int = 384, persist_dir: Optional[Path] = None):
        self.dim = dim
        self.persist_dir = Path(persist_dir or _DEFAULT_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # FAISS index: IndexFlatIP → exact cosine (vectors must be L2-normalised)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)

        # Metadata store: int position → dict
        self._metadata: Dict[int, dict] = {}

        # Monotonic counter tracks number of vectors added
        self._count: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(
        self,
        doc_id: int,
        text: str,
        embedding: np.ndarray,
        category: str = "",
        cluster_dist: Optional[dict] = None,
    ) -> int:
        """
        Add a single document to the store.

        Returns
        -------
        int
            Internal FAISS position (0-indexed).
        """
        vec = embedding.astype(np.float32).reshape(1, -1)
        self._index.add(vec)

        position = self._count
        self._metadata[position] = {
            "doc_id": doc_id,
            "text": text,
            "category": category,
            "cluster_dist": cluster_dist or {},
        }
        self._count += 1
        return position

    def add_batch(
        self,
        doc_ids: List[int],
        texts: List[str],
        embeddings: np.ndarray,
        categories: Optional[List[str]] = None,
        cluster_dists: Optional[List[dict]] = None,
    ) -> None:
        """
        Batch insert — much faster than calling add() in a loop because
        FAISS processes the whole matrix in one C++ call.
        """
        n = len(doc_ids)
        if embeddings.shape != (n, self.dim):
            raise ValueError(
                f"embeddings shape {embeddings.shape} does not match "
                f"expected ({n}, {self.dim})"
            )

        vecs = embeddings.astype(np.float32)
        self._index.add(vecs)

        for i in range(n):
            position = self._count + i
            self._metadata[position] = {
                "doc_id": doc_ids[i],
                "text": texts[i],
                "category": categories[i] if categories else "",
                "cluster_dist": cluster_dists[i] if cluster_dists else {},
            }

        self._count += n
        logger.info("Added %d documents. Total: %d", n, self._count)

    def update_cluster_distributions(
        self, cluster_dists: List[dict]
    ) -> None:
        """
        Update cluster membership distributions for all stored documents.
        Called after fuzzy clustering is complete.

        Parameters
        ----------
        cluster_dists : list of dict
            Must be in the same order as documents were inserted.
        """
        if len(cluster_dists) != self._count:
            raise ValueError(
                f"Expected {self._count} distributions, got {len(cluster_dists)}"
            )
        for i, dist in enumerate(cluster_dists):
            self._metadata[i]["cluster_dist"] = dist
        logger.info("Updated cluster distributions for %d documents.", self._count)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[float, dict]]:
        """
        Return top-k most similar documents.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape (dim,) or (1, dim).
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (score, metadata_dict) tuples, sorted descending by score.
        """
        if self._count == 0:
            return []

        vec = query_embedding.astype(np.float32).reshape(1, -1)
        k = min(top_k, self._count)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self._metadata[idx])
            results.append((float(score), meta))

        return results

    def get_all_embeddings(self) -> np.ndarray:
        """
        Reconstruct all stored vectors from the FAISS index.
        Used by the clustering module to run GMM on the full corpus.

        Returns
        -------
        np.ndarray of shape (n_docs, dim)
        """
        if self._count == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        # faiss.reconstruct_batch retrieves vectors by position
        all_vecs = np.zeros((self._count, self.dim), dtype=np.float32)
        for i in range(self._count):
            all_vecs[i] = self._index.reconstruct(i)
        return all_vecs

    def get_metadata_list(self) -> List[dict]:
        """Return all metadata dicts in insertion order."""
        return [self._metadata[i] for i in range(self._count)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        index_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"metadata": self._metadata, "count": self._count}, f
            )
        logger.info("Saved vector store to %s", self.persist_dir)

    def load(self) -> bool:
        """
        Load a previously saved index and metadata from disk.

        Returns
        -------
        bool
            True if loaded successfully, False if files not found.
        """
        index_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            logger.warning("No saved vector store found at %s.", self.persist_dir)
            return False

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            data = pickle.load(f)

        self._metadata = data["metadata"]
        self._count = data["count"]
        logger.info(
            "Loaded vector store: %d documents from %s",
            self._count,
            self.persist_dir,
        )
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return self._count

    def __repr__(self) -> str:
        return f"VectorDatabase(dim={self.dim}, size={self._count})"


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = VectorDatabase(dim=4)

    # Insert dummy documents
    for i in range(5):
        vec = np.random.rand(4).astype(np.float32)
        vec /= np.linalg.norm(vec)
        db.add(doc_id=i, text=f"Document {i}", embedding=vec, category="test")

    print(db)

    # Search
    query = np.random.rand(4).astype(np.float32)
    query /= np.linalg.norm(query)
    results = db.search(query, top_k=3)
    for score, meta in results:
        print(f"  score={score:.4f}  doc_id={meta['doc_id']}")
