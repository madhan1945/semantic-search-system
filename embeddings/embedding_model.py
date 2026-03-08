"""
embedding_model.py
------------------
Wraps sentence-transformers to produce document/query embeddings.

Model Choice — sentence-transformers/all-MiniLM-L6-v2
------------------------------------------------------
Speed vs Accuracy tradeoff:
  • 'all-MiniLM-L6-v2' has 22 M parameters and produces 384-dimensional
    embeddings.  It is 5× faster than larger models (e.g., all-mpnet-base-v2)
    while retaining ~95% of their semantic quality on STS benchmarks.
  • For a search system where latency matters at query time and we have
    ~20 k documents to embed offline, this is the right operating point.
  • The model was fine-tuned on 1B+ sentence pairs, so it generalises well
    to newsgroup-style informal text.
  • 384 dimensions keeps FAISS index memory manageable (≈ 30 MB for 20 k docs).

Alternative models considered:
  • all-mpnet-base-v2   → better accuracy, 3× slower, 768-dim
  • paraphrase-MiniLM-L3-v2 → fastest, but noticeably weaker on rare topics
"""

import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model identifier (overridable via environment variable)
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Local cache directory so the model is only downloaded once
_CACHE_DIR = Path(__file__).parent.parent / ".model_cache"


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or local path.
    batch_size : int
        Sentences per forward pass.  Larger values use more RAM but are
        faster on CPU because of better vectorisation.
    normalize : bool
        L2-normalise embeddings so dot-product == cosine similarity.
        Required for FAISS IndexFlatIP (inner product) to behave as cosine.
    device : str
        'cpu' or 'cuda'.  Auto-detected if not specified.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        normalize: bool = True,
        device: str = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(
            model_name,
            cache_folder=str(_CACHE_DIR),
            device=device,
        )
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded. Embedding dimension: %d", self.embedding_dim
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into embedding vectors.

        Parameters
        ----------
        texts : str or list of str
            Input text(s).
        show_progress : bool
            Display tqdm progress bar during batch encoding.

        Returns
        -------
        np.ndarray
            Shape (n, embedding_dim) for a list, or (embedding_dim,) for a
            single string.  Always float32.
        """
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True
        else:
            squeeze = False

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)
        return embeddings[0] if squeeze else embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single search query.
        Convenience wrapper — identical to encode() but clearly signals
        intent at call sites.
        """
        return self.encode(query)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1-D embedding vectors.
        When normalize=True the vectors are already unit-norm, so
        cosine similarity == dot product.
        """
        if self.normalize:
            return float(np.dot(a, b))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self.embedding_dim


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = EmbeddingModel()

    sentences = [
        "What is gun control policy?",
        "Explain firearm regulation laws",
        "How does NASA plan Mars missions?",
    ]

    vecs = model.encode(sentences)
    print(f"Embeddings shape: {vecs.shape}")

    sim_01 = model.cosine_similarity(vecs[0], vecs[1])
    sim_02 = model.cosine_similarity(vecs[0], vecs[2])
    print(f"Similarity (gun q1 vs q2): {sim_01:.4f}")   # should be high
    print(f"Similarity (gun vs NASA):  {sim_02:.4f}")   # should be low
