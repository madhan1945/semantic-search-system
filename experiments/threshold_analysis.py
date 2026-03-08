"""
experiments/threshold_analysis.py
----------------------------------
Empirical analysis of how the similarity threshold affects cache behaviour.

Experiments
-----------
1. Hit-rate curve:
   Vary threshold from 0.50 to 1.00 in 0.05 steps.
   For each threshold, run a fixed set of query pairs and record hit rate.

2. Precision / Recall tradeoff:
   - Low threshold → high hit rate but may return irrelevant cached results.
   - High threshold → precise matches but low hit rate (misses many valid hits).

3. Latency comparison:
   Cache hit path vs. vector DB search path.

4. Cluster BIC / AIC curves (GMM model selection):
   Plot BIC and AIC as a function of n_components to show how the optimal
   number of clusters is determined.

Run:
    python experiments/threshold_analysis.py

Output:
    data/experiments/*.json   — raw results
    data/experiments/*.png    — plots (requires matplotlib)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

_OUTPUT_DIR = _ROOT / "data" / "experiments"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Query pair definitions
# ---------------------------------------------------------------------------

# Each tuple: (query_a, query_b, expected_match: bool)
# We expect True pairs to hit the cache when query_a is stored first.
QUERY_PAIRS: List[Tuple[str, str, bool]] = [
    # High-similarity pairs (should hit)
    ("What is gun control policy?", "Explain firearm regulation laws", True),
    ("How does NASA plan Mars missions?", "What are NASA's plans for Mars exploration?", True),
    ("Tell me about electric vehicles", "What are the benefits of EVs?", True),
    ("How to treat a cold?", "What are home remedies for the common cold?", True),
    ("Explain quantum computing basics", "What is quantum computing?", True),
    # Medium-similarity pairs (outcome depends on threshold)
    ("What is gun control?", "Discuss government policies", False),
    ("Space exploration history", "How do rockets work?", False),
    # Low-similarity pairs (should miss)
    ("What is gun control policy?", "How to bake sourdough bread?", False),
    ("Space exploration", "Stock market trends", False),
    ("Electric vehicles", "Medieval history", False),
]


# ---------------------------------------------------------------------------
# Experiment 1: Hit-rate curve across thresholds
# ---------------------------------------------------------------------------

def run_threshold_experiment(
    n_docs: int = 200,
    thresholds: List[float] = None,
) -> Dict:
    """
    Sweep similarity threshold and measure cache hit rate.

    For each threshold:
    1. Build a small embedding model.
    2. Store the first query of each pair.
    3. Query with the second query.
    4. Record hit / miss.

    Returns dict of {threshold: hit_rate}.
    """
    from cache.semantic_cache import SemanticCache
    from embeddings.embedding_model import EmbeddingModel

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 1.01, 0.05)]

    logger.info("Loading embedding model for threshold experiment…")
    encoder = EmbeddingModel()

    # Pre-compute all embeddings
    logger.info("Embedding %d query pairs…", len(QUERY_PAIRS))
    all_queries_a = [p[0] for p in QUERY_PAIRS]
    all_queries_b = [p[1] for p in QUERY_PAIRS]
    embs_a = encoder.encode(all_queries_a)
    embs_b = encoder.encode(all_queries_b)

    # Dummy cluster distributions (all in cluster 0 for simplicity)
    dummy_dist = {"cluster_0": 1.0}

    results = {}
    for threshold in thresholds:
        cache = SemanticCache(similarity_threshold=threshold)
        # Store all "a" queries
        for i, (qa, emb_a) in enumerate(zip(all_queries_a, embs_a)):
            cache.store(
                query_text=qa,
                query_embedding=emb_a,
                result={"mock": f"result_{i}"},
                cluster_membership=dummy_dist,
            )
        # Query with "b" queries
        hits = 0
        for emb_b in embs_b:
            hit, _ = cache.lookup(emb_b, dummy_dist)
            if hit:
                hits += 1

        hit_rate = hits / len(QUERY_PAIRS)
        results[threshold] = {
            "hit_rate": round(hit_rate, 4),
            "hits": hits,
            "total": len(QUERY_PAIRS),
        }
        logger.info("  threshold=%.2f  hit_rate=%.4f  hits=%d/%d",
                    threshold, hit_rate, hits, len(QUERY_PAIRS))

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Per-pair analysis
# ---------------------------------------------------------------------------

def run_pair_analysis() -> List[dict]:
    """
    For each QUERY_PAIRS entry, compute the actual cosine similarity
    and assess whether each threshold would produce a hit.
    """
    from embeddings.embedding_model import EmbeddingModel

    encoder = EmbeddingModel()
    records = []

    for qa, qb, expected in QUERY_PAIRS:
        ea = encoder.encode(qa)
        eb = encoder.encode(qb)
        sim = float(np.dot(ea, eb))
        records.append(
            {
                "query_a": qa,
                "query_b": qb,
                "cosine_similarity": round(sim, 4),
                "expected_match": expected,
                "hit_at_085": sim >= 0.85,
                "hit_at_075": sim >= 0.75,
                "hit_at_070": sim >= 0.70,
            }
        )
        logger.info("  sim=%.4f  %s | %s", sim, qa[:40], qb[:40])

    return records


# ---------------------------------------------------------------------------
# Experiment 3: Latency comparison
# ---------------------------------------------------------------------------

def run_latency_experiment(n_warmup: int = 5, n_measure: int = 20) -> dict:
    """
    Measure mean latency for cache hits vs. cache misses.

    Cache hit path: embed → lookup → return  (~5–20 ms on CPU)
    Cache miss path: embed → lookup → FAISS search → store  (~50–200 ms)

    These are approximated without the full pipeline by timing each step.
    """
    from cache.semantic_cache import SemanticCache
    from embeddings.embedding_model import EmbeddingModel

    encoder = EmbeddingModel()
    cache = SemanticCache(similarity_threshold=0.85)

    # Generate random L2-normalised embeddings
    rng = np.random.default_rng(42)

    def rand_vec():
        v = rng.normal(size=encoder.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    # Warm-up
    for _ in range(n_warmup):
        encoder.encode("warm up query")

    # Measure encode + cache lookup (cache miss scenario — empty cache)
    miss_times = []
    for i in range(n_measure):
        start = time.perf_counter()
        v = encoder.encode(f"unique query number {i} {rng.integers(1e6)}")
        cache.lookup(v, {"cluster_0": 1.0})
        miss_times.append((time.perf_counter() - start) * 1000)

    # Store one entry then measure hit scenario
    anchor = encoder.encode("semantic caching is useful for search")
    cache.store("semantic caching is useful for search", anchor,
                {"doc": "result"}, {"cluster_0": 1.0})

    hit_times = []
    for _ in range(n_measure):
        # Slightly perturbed version — same direction, high similarity
        perturbed = anchor + rng.normal(scale=0.02, size=encoder.dim).astype(np.float32)
        perturbed /= np.linalg.norm(perturbed)
        start = time.perf_counter()
        cache.lookup(perturbed, {"cluster_0": 1.0})
        hit_times.append((time.perf_counter() - start) * 1000)

    results = {
        "miss_mean_ms": round(float(np.mean(miss_times)), 2),
        "miss_std_ms": round(float(np.std(miss_times)), 2),
        "hit_mean_ms": round(float(np.mean(hit_times)), 2),
        "hit_std_ms": round(float(np.std(hit_times)), 2),
        "speedup_factor": round(float(np.mean(miss_times)) / float(np.mean(hit_times)), 1),
    }
    logger.info("Latency: miss=%.1f ms  hit=%.1f ms  speedup=%.1fx",
                results["miss_mean_ms"], results["hit_mean_ms"], results["speedup_factor"])
    return results


# ---------------------------------------------------------------------------
# Optional: Plot results if matplotlib is available
# ---------------------------------------------------------------------------

def plot_threshold_curve(threshold_results: dict) -> None:
    """Plot hit-rate vs threshold curve and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    thresholds = sorted(threshold_results.keys())
    hit_rates = [threshold_results[t]["hit_rate"] for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, hit_rates, marker="o", linewidth=2, color="#2E86AB")
    ax.axvline(x=0.85, color="#E84855", linestyle="--", label="Default threshold (0.85)")
    ax.set_xlabel("Similarity Threshold", fontsize=12)
    ax.set_ylabel("Cache Hit Rate", fontsize=12)
    ax.set_title("Semantic Cache Hit Rate vs. Similarity Threshold", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = _OUTPUT_DIR / "threshold_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Plot saved to %s", path)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Threshold sweep")
    logger.info("=" * 60)
    threshold_results = run_threshold_experiment()
    path = _OUTPUT_DIR / "threshold_sweep.json"
    with open(path, "w") as f:
        json.dump(threshold_results, f, indent=2)
    logger.info("Saved to %s", path)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: Per-pair similarity analysis")
    logger.info("=" * 60)
    pair_results = run_pair_analysis()
    path = _OUTPUT_DIR / "pair_analysis.json"
    with open(path, "w") as f:
        json.dump(pair_results, f, indent=2)
    logger.info("Saved to %s", path)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3: Latency comparison")
    logger.info("=" * 60)
    latency_results = run_latency_experiment()
    path = _OUTPUT_DIR / "latency.json"
    with open(path, "w") as f:
        json.dump(latency_results, f, indent=2)
    logger.info("Saved to %s", path)

    # Generate plot
    plot_threshold_curve(threshold_results)

    logger.info("\nAll experiments complete. Results in: %s", _OUTPUT_DIR)


if __name__ == "__main__":
    main()
