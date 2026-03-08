"""
clean_text.py
-------------
Multi-stage text cleaning pipeline for newsgroup posts.

Design Decisions:
    Each cleaning step is implemented as a small, focused function so that:
    - Individual steps can be toggled without rewriting the pipeline.
    - Unit tests target each step in isolation.
    - Comments explain *why* the step exists, not just *what* it does.

Pipeline order matters:
    1. Lowercase first so regex patterns are case-insensitive by default.
    2. Remove structural noise (headers, emails, URLs) before token-level ops.
    3. Collapse whitespace last so prior steps can be messy.
"""

import logging
import re
import string
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regexes (compiled once at import time for performance)
# ---------------------------------------------------------------------------

# Email addresses: user@domain.tld (newsgroups contain many)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")

# URLs: http(s):// or www. prefixed links
_RE_URL = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)

# Lines that look like email/news headers:  "From:", "Subject:", "Lines:" etc.
_RE_HEADER_LINE = re.compile(
    r"^(from|subject|newsgroups|organization|lines|message-id|"
    r"references|nntp-posting-host|x-newsreader|date|path|"
    r"reply-to|sender|distribution|mime-version|content-type|"
    r"content-transfer-encoding)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Quoted reply lines: lines starting with ">" (common in newsgroups)
_RE_QUOTE_LINE = re.compile(r"^>.*$", re.MULTILINE)

# HTML/XML tags
_RE_HTML_TAG = re.compile(r"<[^>]+>")

# HTML entities: &amp; &lt; &gt; &nbsp; etc.
_RE_HTML_ENTITY = re.compile(r"&[a-z]+;", re.IGNORECASE)

# Non-alphabetic characters (keeps spaces; punctuation removed separately)
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")

# Multiple whitespace characters → single space
_RE_WHITESPACE = re.compile(r"\s+")

# Signature blocks: lines after "-- " (email/news convention)
_RE_SIGNATURE = re.compile(r"\n--\s*\n.*", re.DOTALL)

# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------


def to_lowercase(text: str) -> str:
    """
    Lowercase all characters.
    Reason: Ensures 'Gun' and 'gun' map to the same token, reducing vocabulary
    size and improving embedding consistency.
    """
    return text.lower()


def remove_email_headers(text: str) -> str:
    """
    Strip structural newsgroup/email header lines (From:, Subject:, etc.).
    Reason: These lines carry routing metadata, not semantic content.
    Leaving them in would cause the model to learn spurious patterns
    (e.g., associating 'organization:' with a topic).
    """
    return _RE_HEADER_LINE.sub("", text)


def remove_signatures(text: str) -> str:
    """
    Remove email signature blocks that follow the standard '-- \\n' delimiter.
    Reason: Signatures are boilerplate personal/organisational info and
    introduce noise that hurts topic coherence.
    """
    return _RE_SIGNATURE.sub("", text)


def remove_quoted_lines(text: str) -> str:
    """
    Remove '>' prefixed quote lines common in newsgroup replies.
    Reason: Quoted text is someone else's content and would duplicate signal
    from their original post, biasing cluster assignments.
    """
    return _RE_QUOTE_LINE.sub("", text)


def remove_html(text: str) -> str:
    """
    Strip HTML/XML tags and decode common HTML entities.
    Reason: Some newsgroup messages were composed in HTML-capable clients,
    leaving markup that adds no semantic content.
    """
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_HTML_ENTITY.sub(" ", text)
    return text


def remove_emails(text: str) -> str:
    """
    Remove email addresses.
    Reason: Email addresses are identifiers, not content.  Keeping them
    could cause the model to cluster by author rather than topic.
    """
    return _RE_EMAIL.sub(" ", text)


def remove_urls(text: str) -> str:
    """
    Remove URLs.
    Reason: URLs are often machine-generated or domain-specific and carry
    little transferable semantic meaning for our embedding model.
    """
    return _RE_URL.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation characters.
    Reason: For bag-of-words style downstream processing (stopword removal,
    keyword extraction) punctuation is noise.  The sentence-transformer
    handles its own sub-word tokenisation so pre-removing punctuation
    is a mild form of normalisation rather than a strict requirement.
    """
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_non_alpha(text: str) -> str:
    """
    Remove any remaining non-alphabetic characters (digits, special symbols).
    Reason: After the previous steps, residual numbers and symbols are
    unlikely to carry topic-discriminating signal.
    """
    return _RE_NON_ALPHA.sub(" ", text)


def remove_stopwords(text: str, language: str = "english") -> str:
    """
    Remove common English stopwords using NLTK's curated list.
    Reason: Stopwords (the, is, at, …) occur uniformly across all topics
    and inflate the vocabulary without aiding similarity computation.
    Note: We tokenise→filter→rejoin to avoid regex boundary issues.
    """
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(text)
    filtered = [tok for tok in tokens if tok not in stop_words and len(tok) > 1]
    return " ".join(filtered)


def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace runs (spaces, tabs, newlines) to a single space
    and strip leading/trailing whitespace.
    Reason: Prior steps leave irregular whitespace; normalising here ensures
    a clean string is passed to the embedding model.
    """
    return _RE_WHITESPACE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class TextCleaner:
    """
    Applies the cleaning pipeline in a fixed, sensible order.

    Parameters
    ----------
    remove_stops : bool
        Whether to apply stopword removal.  Default True.
        Set to False if you want to preserve full grammatical sentences
        (e.g., for models that benefit from context like BERT-style encoders).
    min_length : int
        Documents shorter than this (in characters) after cleaning are
        returned as empty strings and should be filtered downstream.
    """

    def __init__(self, remove_stops: bool = True, min_length: int = 20):
        self.remove_stops = remove_stops
        self.min_length = min_length

    def clean(self, text: str) -> str:
        """Run the full pipeline on a single document string."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = to_lowercase(text)
        text = remove_email_headers(text)
        text = remove_signatures(text)
        text = remove_quoted_lines(text)
        text = remove_html(text)
        text = remove_emails(text)
        text = remove_urls(text)
        text = remove_punctuation(text)
        text = remove_non_alpha(text)

        if self.remove_stops:
            text = remove_stopwords(text)

        text = normalize_whitespace(text)

        if len(text) < self.min_length:
            return ""

        return text

    def clean_batch(self, texts: list, show_progress: bool = False) -> list:
        """
        Clean a list of documents.  Returns a list of the same length;
        short/empty documents are represented as empty strings.
        """
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Cleaning")
            except ImportError:
                pass

        return [self.clean(t) for t in texts]


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = """From: user@example.com (John)
Subject: Re: Gun control debate
Organization: University of Example
Lines: 12

> Original poster said something interesting.

I agree with the above point.  Here is my take on firearm regulation:
The laws are quite complex and vary by state.  See http://example.com/laws
for more info.

-- 
John Doe | user@example.com
"""
    cleaner = TextCleaner()
    print("Raw:\n", sample)
    print("\nCleaned:\n", cleaner.clean(sample))
