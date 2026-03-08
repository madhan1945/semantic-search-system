"""
dataset_loader.py
-----------------
Responsible for loading the 20 Newsgroups dataset from scikit-learn's built-in
fetcher (which mirrors the UCI archive) and returning a clean DataFrame.

Design Decision:
    We use sklearn's fetch_20newsgroups instead of a raw HTTP download because:
    1. It handles caching transparently (~17 MB, downloaded once).
    2. It provides structured access to text + category metadata.
    3. It is reproducible across environments without custom downloader logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class NewsDocument:
    """Typed container for a single newsgroup document."""
    doc_id: int
    text: str
    category: str
    category_id: int


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Loads the 20 Newsgroups dataset and exposes it as a pandas DataFrame
    or a list of NewsDocument instances.

    Parameters
    ----------
    subset : str
        One of 'train', 'test', or 'all'.  Default is 'all' so we maximise
        the corpus available for embedding and clustering.
    remove : tuple
        Parts of each post to strip before any further preprocessing.
        We remove headers, footers, and quote sections here because they
        contain metadata that would pollute semantic embeddings.
    categories : Optional[list]
        Limit to a subset of the 20 categories.  None → all 20.
    """

    def __init__(
        self,
        subset: str = "all",
        remove: tuple = ("headers", "footers", "quotes"),
        categories: Optional[list] = None,
    ):
        self.subset = subset
        self.remove = remove
        self.categories = categories
        self._raw = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Fetch the dataset and return a tidy DataFrame with columns:
            doc_id | text | category | category_id
        """
        logger.info("Fetching 20 Newsgroups dataset (subset=%s)…", self.subset)
        self._raw = fetch_20newsgroups(
            subset=self.subset,
            remove=self.remove,
            categories=self.categories,
            shuffle=True,
            random_state=42,
        )

        target_names = self._raw.target_names  # list of category strings

        records = []
        for idx, (text, target_id) in enumerate(
            zip(self._raw.data, self._raw.target)
        ):
            records.append(
                {
                    "doc_id": idx,
                    "text": text,
                    "category": target_names[target_id],
                    "category_id": int(target_id),
                }
            )

        df = pd.DataFrame(records)
        logger.info(
            "Loaded %d documents across %d categories.",
            len(df),
            df["category_id"].nunique(),
        )
        return df

    def category_names(self) -> list:
        """Return the ordered list of category label strings."""
        if self._raw is None:
            raise RuntimeError("Call .load() before accessing category_names().")
        return self._raw.target_names

    def sample(self, n: int = 5, seed: int = 42) -> pd.DataFrame:
        """Convenience: return n random rows for quick inspection."""
        df = self.load()
        return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DatasetLoader()
    df = loader.load()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nCategories:\n{df['category'].value_counts()}")
