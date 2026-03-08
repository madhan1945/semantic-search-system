"""
fuzzy_clustering.py
-------------------
Gaussian Mixture Model (GMM) based fuzzy clustering.

Why GMM over hard clustering (K-Means)?
    Documents naturally belong to multiple topics simultaneously.
    A post in alt.guns might discuss politics, law, and technology.
    Hard clustering forces exactly one label; GMM assigns a probability
    distribution over all clusters, which is semantically richer.

Why GMM over Fuzzy C-Means (FCM)?
    GMM has principled model selection via AIC/BIC, which we use to
    determine the number of clusters automatically.  FCM requires manual
    tuning of the fuzzification parameter m and lacks a native model
    selection criterion.

Cluster count selection (n_components):
    We fit GMMs for n in [5, 10, 15, 20, 25, 30] and choose the n that
    minimises BIC (Bayesian Information Criterion).
    BIC penalises model complexity, preventing over-fitting to noise.
    AIC is also computed for comparison.
    See experiments/threshold_analysis.py for visualisation.

Covariance type:
    'diag' — each component has its own diagonal covariance matrix.
    Chosen because:
    - 'full' is too expensive at 384 dims (covariance matrices are 384×384).
    - 'spherical' is too rigid for high-dimensional text embeddings.
    - 'diag' captures per-dimension variance at manageable cost.

Dimensionality reduction before clustering:
    Clustering in 384 dimensions suffers from the curse of dimensionality.
    We reduce to 50 dims with PCA (retaining ~95% variance) before fitting
    the GMM.  The original 384-dim embeddings are still stored in FAISS
    for retrieval; PCA is only used inside this module.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST = Path(__file__).parent.parent / "data" / "clustering"

# Candidate cluster counts to evaluate
_CANDIDATE_COMPONENTS = [5, 8, 10, 12, 15, 20, 25, 30]

# PCA target dimensions
_PCA_DIMS = 50


class FuzzyClusterer:
    """
    Fits a GMM on document embeddings and returns per-document cluster
    membership probability distributions.

    Parameters
    ----------
    n_components : int or None
        Number of mixture components.  If None, automatically selected
        by minimising BIC over _CANDIDATE_COMPONENTS.
    covariance_type : str
        GMM covariance structure ('diag' recommended).
    pca_dims : int
        Number of PCA components to reduce to before clustering.
    random_state : int
        Reproducibility seed.
    persist_dir : Path or str
        Where to save/load fitted models.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        covariance_type: str = "diag",
        pca_dims: int = _PCA_DIMS,
        random_state: int = 42,
        persist_dir: Optional[Path] = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.pca_dims = pca_dims
        self.random_state = random_state
        self.persist_dir = Path(persist_dir or _DEFAULT_PERSIST)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._pca: Optional[PCA] = None
        self._gmm: Optional[GaussianMixture] = None
        self._is_fitted: bool = False
        self._bic_scores: Dict[int, float] = {}
        self._aic_scores: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> "FuzzyClusterer":
        """
        Fit PCA + GMM on the provided embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_docs, embedding_dim).  Should be L2-normalised.

        Returns
        -------
        self
        """
        logger.info(
            "Fitting clusterer on %d documents (dim=%d)…",
            embeddings.shape[0],
            embeddings.shape[1],
        )

        # Step 1: PCA dimensionality reduction
        effective_pca_dims = min(self.pca_dims, embeddings.shape[1], embeddings.shape[0] - 1)
        logger.info("Running PCA: %d → %d dims", embeddings.shape[1], effective_pca_dims)
        self._pca = PCA(n_components=effective_pca_dims, random_state=self.random_state)
        reduced = self._pca.fit_transform(embeddings)
        variance_explained = self._pca.explained_variance_ratio_.sum()
        logger.info("PCA variance explained: %.1f%%", variance_explained * 100)

        # Step 2: Determine optimal number of components via BIC
        if self.n_components is None:
            self.n_components = self._select_n_components(reduced)

        # Step 3: Fit final GMM
        logger.info(
            "Fitting GMM with n_components=%d, covariance=%s",
            self.n_components,
            self.covariance_type,
        )
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            n_init=3,          # Multiple random initialisations → more stable solution
            verbose=0,
        )
        self._gmm.fit(reduced)
        self._is_fitted = True
        logger.info(
            "GMM converged: %s  |  lower_bound=%.4f",
            self._gmm.converged_,
            self._gmm.lower_bound_,
        )
        return self

    def _select_n_components(self, reduced: np.ndarray) -> int:
        """
        Evaluate BIC for each candidate n_components and return the best.
        Also computes AIC and silhouette for logging/inspection.
        """
        logger.info(
            "Selecting n_components via BIC over candidates: %s",
            _CANDIDATE_COMPONENTS,
        )
        best_n = _CANDIDATE_COMPONENTS[0]
        best_bic = np.inf

        # Limit candidates to those that are valid given sample count
        max_valid = min(reduced.shape[0] - 1, max(_CANDIDATE_COMPONENTS))
        candidates = [c for c in _CANDIDATE_COMPONENTS if c <= max_valid]

        for n in candidates:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                max_iter=100,
                n_init=2,
            )
            gmm.fit(reduced)
            bic = gmm.bic(reduced)
            aic = gmm.aic(reduced)
            self._bic_scores[n] = bic
            self._aic_scores[n] = aic

            # Silhouette score requires hard labels — use predicted labels
            labels = gmm.predict(reduced)
            try:
                sil = silhouette_score(reduced, labels, sample_size=2000, random_state=42)
            except Exception:
                sil = float("nan")

            logger.info(
                "  n=%2d  BIC=%.1f  AIC=%.1f  Silhouette=%.3f", n, bic, aic, sil
            )

            if bic < best_bic:
                best_bic = bic
                best_n = n

        logger.info("Selected n_components=%d (lowest BIC=%.1f)", best_n, best_bic)
        return best_n

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return cluster membership probability matrix.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n, dim) — same space as training embeddings.

        Returns
        -------
        np.ndarray of shape (n, n_components) where each row sums to 1.
        """
        self._check_fitted()
        reduced = self._pca.transform(embeddings)
        return self._gmm.predict_proba(reduced)

    def get_cluster_distributions(
        self, embeddings: np.ndarray, top_k: int = 3, threshold: float = 0.05
    ) -> List[dict]:
        """
        Return per-document cluster distributions in the required format:

            {"cluster_0": 0.62, "cluster_3": 0.27, "cluster_7": 0.11}

        Only clusters above `threshold` probability are included, and at
        most `top_k` clusters are returned.  This keeps distributions sparse
        and interpretable.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n, dim).
        top_k : int
            Maximum clusters to include per document.
        threshold : float
            Minimum probability to include a cluster.

        Returns
        -------
        list of dict, length n.
        """
        proba_matrix = self.predict_proba(embeddings)
        distributions = []

        for proba_row in proba_matrix:
            # Sort by probability descending
            ranked = np.argsort(proba_row)[::-1]
            dist = {}
            for idx in ranked[:top_k]:
                p = float(proba_row[idx])
                if p >= threshold:
                    dist[f"cluster_{idx}"] = round(p, 4)
            distributions.append(dist)

        return distributions

    def get_document_distribution(
        self, doc_embedding: np.ndarray, top_k: int = 3
    ) -> dict:
        """
        Get cluster distribution for a single query/document embedding.
        Used by the semantic cache to identify relevant cluster buckets.
        """
        proba = self.predict_proba(doc_embedding.reshape(1, -1))[0]
        ranked = np.argsort(proba)[::-1]
        return {
            f"cluster_{i}": round(float(proba[i]), 4)
            for i in ranked[:top_k]
            if proba[i] >= 0.05
        }

    def dominant_cluster(self, embedding: np.ndarray) -> int:
        """Return the index of the cluster with highest membership probability."""
        self._check_fitted()
        reduced = self._pca.transform(embedding.reshape(1, -1))
        return int(self._gmm.predict(reduced)[0])

    # ------------------------------------------------------------------
    # Cluster analysis helpers
    # ------------------------------------------------------------------

    def cluster_weights(self) -> np.ndarray:
        """Return the mixing weights of each Gaussian component."""
        self._check_fitted()
        return self._gmm.weights_

    def most_uncertain_indices(
        self, embeddings: np.ndarray, n: int = 10
    ) -> List[int]:
        """
        Return indices of documents with highest entropy cluster membership.
        These are 'boundary cases' — documents that straddle multiple topics.
        Entropy is maximised when probabilities are uniform across clusters.
        """
        proba = self.predict_proba(embeddings)
        # Entropy: -sum(p * log(p))
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        return list(np.argsort(entropy)[::-1][:n])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist fitted models to disk."""
        self._check_fitted()
        path = self.persist_dir / "clusterer.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pca": self._pca,
                    "gmm": self._gmm,
                    "n_components": self.n_components,
                    "bic_scores": self._bic_scores,
                    "aic_scores": self._aic_scores,
                },
                f,
            )
        logger.info("Saved clusterer to %s", path)

    def load(self) -> bool:
        """Load a previously saved clusterer.  Returns True on success."""
        path = self.persist_dir / "clusterer.pkl"
        if not path.exists():
            logger.warning("No saved clusterer found at %s.", path)
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._pca = data["pca"]
        self._gmm = data["gmm"]
        self.n_components = data["n_components"]
        self._bic_scores = data.get("bic_scores", {})
        self._aic_scores = data.get("aic_scores", {})
        self._is_fitted = True
        logger.info(
            "Loaded clusterer (n_components=%d) from %s", self.n_components, path
        )
        return True

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Clusterer is not fitted.  Call .fit() or .load() first."
            )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    # Simulate 200 embeddings in 384 dims
    fake_embeddings = rng.normal(size=(200, 384)).astype(np.float32)
    norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
    fake_embeddings /= norms

    clusterer = FuzzyClusterer()
    clusterer.fit(fake_embeddings)

    dists = clusterer.get_cluster_distributions(fake_embeddings[:5])
    for i, d in enumerate(dists):
        print(f"Doc {i}: {d}")

    uncertain = clusterer.most_uncertain_indices(fake_embeddings, n=3)
    print(f"\nMost uncertain docs: {uncertain}")
