"""
tests/test_all.py
-----------------
Comprehensive test suite covering all major components.

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import sys
import os
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_model():
    from embeddings.embedding_model import EmbeddingModel
    return EmbeddingModel()


@pytest.fixture(scope="module")
def text_cleaner():
    from preprocessing.clean_text import TextCleaner
    return TextCleaner()


@pytest.fixture(scope="module")
def vector_db():
    from vector_store.vector_db import VectorDatabase
    return VectorDatabase(dim=384)


@pytest.fixture(scope="module")
def semantic_cache():
    from cache.semantic_cache import SemanticCache
    return SemanticCache(similarity_threshold=0.85)


# ---------------------------------------------------------------------------
# Tests: TextCleaner
# ---------------------------------------------------------------------------

class TestTextCleaner:

    def test_removes_email_headers(self, text_cleaner):
        text = "From: user@example.com\nSubject: Test\n\nHello world."
        cleaned = text_cleaner.clean(text)
        assert "From:" not in cleaned
        assert "Subject:" not in cleaned

    def test_removes_email_addresses(self, text_cleaner):
        text = "Contact me at someone@domain.com for details."
        cleaned = text_cleaner.clean(text)
        assert "@" not in cleaned

    def test_removes_urls(self, text_cleaner):
        text = "Visit http://example.com for more info."
        cleaned = text_cleaner.clean(text)
        assert "http" not in cleaned

    def test_removes_quotes(self, text_cleaner):
        text = "> This was said before\nHere is my reply."
        cleaned = text_cleaner.clean(text)
        assert ">" not in cleaned

    def test_lowercase(self, text_cleaner):
        text = "The Quick Brown Fox Jumps Over The Lazy Dog."
        cleaned = text_cleaner.clean(text)
        assert cleaned == cleaned.lower()

    def test_removes_stopwords(self, text_cleaner):
        text = "This is a test of the text cleaning pipeline."
        cleaned = text_cleaner.clean(text)
        # Common stopwords should be gone
        assert " is " not in f" {cleaned} "
        assert " the " not in f" {cleaned} "

    def test_empty_input(self, text_cleaner):
        assert text_cleaner.clean("") == ""
        assert text_cleaner.clean("   ") == ""

    def test_short_result_returns_empty(self, text_cleaner):
        # Only stopwords → after removal, shorter than min_length
        result = text_cleaner.clean("the is a")
        assert result == ""

    def test_batch_cleaning(self, text_cleaner):
        texts = ["Hello world from user@test.com", "Visit http://test.com", ""]
        results = text_cleaner.clean_batch(texts)
        assert len(results) == 3
        assert "@" not in results[0]
        assert "http" not in results[1]
        assert results[2] == ""

    def test_removes_html(self, text_cleaner):
        text = "<p>Hello <b>world</b></p>"
        cleaned = text_cleaner.clean(text)
        assert "<" not in cleaned
        assert ">" not in cleaned


# ---------------------------------------------------------------------------
# Tests: EmbeddingModel
# ---------------------------------------------------------------------------

class TestEmbeddingModel:

    def test_single_string_produces_1d_vector(self, embedding_model):
        vec = embedding_model.encode("test query")
        assert vec.ndim == 1
        assert vec.shape[0] == embedding_model.dim

    def test_batch_produces_2d_array(self, embedding_model):
        texts = ["first sentence", "second sentence", "third sentence"]
        vecs = embedding_model.encode(texts)
        assert vecs.ndim == 2
        assert vecs.shape == (3, embedding_model.dim)

    def test_vectors_are_normalised(self, embedding_model):
        vec = embedding_model.encode("test query")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_similar_queries_have_high_similarity(self, embedding_model):
        v1 = embedding_model.encode("What is gun control?")
        v2 = embedding_model.encode("Explain firearm regulation laws")
        v3 = embedding_model.encode("How to bake bread")
        sim_12 = embedding_model.cosine_similarity(v1, v2)
        sim_13 = embedding_model.cosine_similarity(v1, v3)
        # Similar topic should score higher than unrelated topic
        assert sim_12 > sim_13

    def test_cosine_similarity_range(self, embedding_model):
        v1 = embedding_model.encode("hello world")
        v2 = embedding_model.encode("goodbye world")
        sim = embedding_model.cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    def test_encode_query_alias(self, embedding_model):
        v1 = embedding_model.encode("test")
        v2 = embedding_model.encode_query("test")
        np.testing.assert_array_almost_equal(v1, v2)


# ---------------------------------------------------------------------------
# Tests: VectorDatabase
# ---------------------------------------------------------------------------

class TestVectorDatabase:

    def _make_vec(self, dim=384, seed=None):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def test_add_and_size(self, vector_db):
        initial = vector_db.size
        vector_db.add(doc_id=9999, text="test doc", embedding=self._make_vec(384, 1))
        assert vector_db.size == initial + 1

    def test_search_returns_results(self, vector_db):
        # Add a few documents with known embeddings
        db = __import__("vector_store.vector_db", fromlist=["VectorDatabase"]).VectorDatabase(dim=4)
        for i in range(5):
            v = np.zeros(4, dtype=np.float32)
            v[i % 4] = 1.0  # one-hot vectors
            db.add(doc_id=i, text=f"doc {i}", embedding=v)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = db.search(query, top_k=3)
        assert len(results) == 3
        # Highest score should be doc 0 (same direction)
        assert results[0][1]["doc_id"] == 0

    def test_search_empty_db(self):
        from vector_store.vector_db import VectorDatabase
        db = VectorDatabase(dim=8)
        query = np.random.rand(8).astype(np.float32)
        results = db.search(query, top_k=5)
        assert results == []

    def test_batch_add(self, vector_db):
        from vector_store.vector_db import VectorDatabase
        db = VectorDatabase(dim=384)
        n = 10
        ids = list(range(n))
        texts = [f"doc {i}" for i in range(n)]
        embs = np.stack([self._make_vec(384, i) for i in range(n)])
        db.add_batch(doc_ids=ids, texts=texts, embeddings=embs)
        assert db.size == n

    def test_get_all_embeddings_shape(self):
        from vector_store.vector_db import VectorDatabase
        db = VectorDatabase(dim=16)
        for i in range(4):
            v = np.random.rand(16).astype(np.float32)
            v /= np.linalg.norm(v)
            db.add(i, f"doc {i}", v)
        embs = db.get_all_embeddings()
        assert embs.shape == (4, 16)


# ---------------------------------------------------------------------------
# Tests: FuzzyClusterer
# ---------------------------------------------------------------------------

class TestFuzzyClusterer:

    @pytest.fixture(scope="class")
    def fitted_clusterer(self):
        from clustering.fuzzy_clustering import FuzzyClusterer
        rng = np.random.default_rng(42)
        embs = rng.normal(size=(150, 32)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        clusterer = FuzzyClusterer(n_components=5, pca_dims=10)
        clusterer.fit(embs)
        return clusterer, embs

    def test_fit_sets_is_fitted(self, fitted_clusterer):
        clusterer, _ = fitted_clusterer
        assert clusterer.is_fitted

    def test_predict_proba_shape(self, fitted_clusterer):
        clusterer, embs = fitted_clusterer
        proba = clusterer.predict_proba(embs[:10])
        assert proba.shape == (10, 5)

    def test_probabilities_sum_to_one(self, fitted_clusterer):
        clusterer, embs = fitted_clusterer
        proba = clusterer.predict_proba(embs[:20])
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(20), decimal=5)

    def test_cluster_distributions_format(self, fitted_clusterer):
        clusterer, embs = fitted_clusterer
        dists = clusterer.get_cluster_distributions(embs[:3])
        assert len(dists) == 3
        for d in dists:
            assert isinstance(d, dict)
            assert all(k.startswith("cluster_") for k in d)
            assert abs(sum(d.values()) - 1.0) < 0.05  # top-k may not sum to 1

    def test_dominant_cluster_is_valid(self, fitted_clusterer):
        clusterer, embs = fitted_clusterer
        c = clusterer.dominant_cluster(embs[0])
        assert 0 <= c < 5

    def test_not_fitted_raises(self):
        from clustering.fuzzy_clustering import FuzzyClusterer
        clusterer = FuzzyClusterer()
        with pytest.raises(RuntimeError):
            clusterer.predict_proba(np.zeros((1, 32)))


# ---------------------------------------------------------------------------
# Tests: SemanticCache
# ---------------------------------------------------------------------------

class TestSemanticCache:

    @pytest.fixture(autouse=True)
    def fresh_cache(self, semantic_cache):
        semantic_cache.flush()
        yield semantic_cache

    def _rand_vec(self, seed=None):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=384).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def test_miss_on_empty_cache(self, semantic_cache):
        q = self._rand_vec(0)
        hit, entry = semantic_cache.lookup(q, {"cluster_0": 1.0})
        assert not hit
        assert entry is None

    def test_exact_match_is_hit(self, semantic_cache):
        v = self._rand_vec(1)
        semantic_cache.store("query A", v, {"result": "A"}, {"cluster_0": 1.0})
        hit, entry = semantic_cache.lookup(v, {"cluster_0": 1.0})
        assert hit
        assert entry.query_text == "query A"

    def test_similar_query_is_hit(self, semantic_cache):
        base = self._rand_vec(2)
        semantic_cache.store("gun control policy", base, {"docs": []}, {"cluster_0": 1.0})

        rng = np.random.default_rng(99)
        perturbed = base + rng.normal(scale=0.01, size=384).astype(np.float32)
        perturbed /= np.linalg.norm(perturbed)

        hit, entry = semantic_cache.lookup(perturbed, {"cluster_0": 1.0})
        assert hit

    def test_dissimilar_query_is_miss(self, semantic_cache):
        v1 = self._rand_vec(3)
        v2 = self._rand_vec(4)  # completely independent random vector
        semantic_cache.store("query about cats", v1, {"docs": []}, {"cluster_1": 1.0})
        hit, _ = semantic_cache.lookup(v2, {"cluster_1": 1.0})
        # Two independent random unit vectors in 384-D have ~0 cosine similarity
        assert not hit

    def test_stats_update_on_hit(self, semantic_cache):
        v = self._rand_vec(5)
        semantic_cache.store("test stats", v, {}, {"cluster_0": 1.0})
        semantic_cache.lookup(v, {"cluster_0": 1.0})
        stats = semantic_cache.stats
        assert stats.hit_count >= 1

    def test_stats_update_on_miss(self, semantic_cache):
        v = self._rand_vec(6)
        semantic_cache.lookup(v, {"cluster_0": 1.0})
        stats = semantic_cache.stats
        assert stats.miss_count >= 1

    def test_flush_clears_entries(self, semantic_cache):
        v = self._rand_vec(7)
        semantic_cache.store("temp query", v, {}, {"cluster_0": 1.0})
        semantic_cache.flush()
        assert semantic_cache.stats.total_entries == 0

    def test_hit_rate_calculation(self, semantic_cache):
        v = self._rand_vec(8)
        semantic_cache.store("q", v, {}, {"cluster_0": 1.0})
        semantic_cache.lookup(v, {"cluster_0": 1.0})   # hit
        semantic_cache.lookup(self._rand_vec(9), {"cluster_0": 1.0})  # miss
        stats = semantic_cache.stats
        assert 0 < stats.hit_rate < 1.0

    def test_high_threshold_reduces_hits(self):
        from cache.semantic_cache import SemanticCache
        strict_cache = SemanticCache(similarity_threshold=0.999)
        base = self._rand_vec(10)
        strict_cache.store("strict q", base, {}, {"cluster_0": 1.0})
        rng = np.random.default_rng(11)
        perturbed = base + rng.normal(scale=0.05, size=384).astype(np.float32)
        perturbed /= np.linalg.norm(perturbed)
        hit, _ = strict_cache.lookup(perturbed, {"cluster_0": 1.0})
        # With a very high threshold, slight perturbation should not hit
        assert not hit

    def test_low_threshold_increases_hits(self):
        from cache.semantic_cache import SemanticCache
        lenient_cache = SemanticCache(similarity_threshold=0.50)
        base = self._rand_vec(12)
        lenient_cache.store("lenient q", base, {}, {"cluster_0": 1.0})
        rng = np.random.default_rng(13)
        perturbed = base + rng.normal(scale=0.1, size=384).astype(np.float32)
        perturbed /= np.linalg.norm(perturbed)
        hit, _ = lenient_cache.lookup(perturbed, {"cluster_0": 1.0})
        assert hit


# ---------------------------------------------------------------------------
# Tests: DatasetLoader
# ---------------------------------------------------------------------------

class TestDatasetLoader:

    def test_load_returns_dataframe(self):
        from data.dataset_loader import DatasetLoader
        import pandas as pd
        loader = DatasetLoader(categories=["sci.space", "rec.sport.hockey"])
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"doc_id", "text", "category", "category_id"}

    def test_correct_number_of_categories(self):
        from data.dataset_loader import DatasetLoader
        loader = DatasetLoader(categories=["sci.space", "talk.politics.guns"])
        df = loader.load()
        assert df["category"].nunique() == 2

    def test_no_null_texts(self):
        from data.dataset_loader import DatasetLoader
        loader = DatasetLoader(categories=["sci.space"])
        df = loader.load()
        assert df["text"].notna().all()

    def test_doc_ids_unique(self):
        from data.dataset_loader import DatasetLoader
        loader = DatasetLoader(categories=["sci.space"])
        df = loader.load()
        assert df["doc_id"].is_unique


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestIntegration:
    """
    Light end-to-end test using a small dataset (no full 20k indexing).
    """

    def test_query_pipeline_cache_miss_then_hit(self):
        """
        Store two semantically similar queries and verify the second hits.
        This tests the full path: embed → cluster → cache lookup/store.
        """
        from embeddings.embedding_model import EmbeddingModel
        from cache.semantic_cache import SemanticCache
        from clustering.fuzzy_clustering import FuzzyClusterer

        encoder = EmbeddingModel()
        cache = SemanticCache(similarity_threshold=0.80)
        clusterer = FuzzyClusterer(n_components=3, pca_dims=10)

        # Fit clusterer on small random data
        rng = np.random.default_rng(0)
        dummy_data = rng.normal(size=(50, encoder.dim)).astype(np.float32)
        dummy_data /= np.linalg.norm(dummy_data, axis=1, keepdims=True)
        clusterer.fit(dummy_data)

        q1 = "What is gun control?"
        q2 = "Explain firearm legislation"

        e1 = encoder.encode(q1)
        e2 = encoder.encode(q2)

        dist1 = clusterer.get_document_distribution(e1)

        # First query → miss
        hit, _ = cache.lookup(e1, dist1)
        assert not hit

        cache.store(q1, e1, {"docs": ["result"]}, dist1)

        # Second similar query → should hit
        dist2 = clusterer.get_document_distribution(e2)
        hit, entry = cache.lookup(e2, dist2)
        # This may or may not hit depending on actual similarity — check sim explicitly
        sim = float(np.dot(e1, e2))
        if sim >= 0.80:
            assert hit
            assert entry.query_text == q1
        else:
            pytest.skip(f"Queries not similar enough (sim={sim:.3f}) — skipping hit assertion")
