#!/usr/bin/env python3
"""
utils.py — Shared utility functions.

Covers:
  - Corpus loading from data/Wikipedia Entries/
  - Artifact path constants (single source of truth)
  - Simple timing context manager
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


# ---------------------------------------------------------------------------
# Artifact paths — single source of truth used by build and run
# ---------------------------------------------------------------------------
ARTIFACTS_DIR      = Path("artifacts")
FAISS_INDEX_PATH   = str(ARTIFACTS_DIR / "faiss.index")
CHUNK_META_PATH    = str(ARTIFACTS_DIR / "chunk_meta.json")
CHUNK_VECTORS_PATH = str(ARTIFACTS_DIR / "chunk_vectors.npy")
BM25_PREFIX        = str(ARTIFACTS_DIR / "bm25")


CORPUS_DIR         = Path("data") / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = Path("data") / "public_queries.json"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: str | Path = CORPUS_DIR) -> list[dict]:
    """
    Load all Wikipedia page JSON files from corpus_dir.

    Returns:
        List of dicts: [{page_id: int, title: str, content: str}, ...]
    """
    corpus_dir = Path(corpus_dir)
    pages = []
    json_files = sorted(corpus_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {corpus_dir}")

    for fpath in json_files:
        with open(fpath, encoding="utf-8") as f:
            page = json.load(f)
        pages.append(page)

    print(f"Loaded {len(pages)} pages from {corpus_dir}")
    return pages


def load_public_queries(path: str | Path = PUBLIC_QUERIES_PATH) -> list[dict]:
    """Load public queries JSON (list of {query_id, query, relevant_page_ids})."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str):
    """
    Context manager that prints elapsed time for a block.

    Usage:
        with timer("Building FAISS index"):
            build_faiss_index(...)
    """
    t0 = time.perf_counter()
    print(f"[{label}] starting ...")
    yield
    elapsed = time.perf_counter() - t0
    print(f"[{label}] done in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Eval constants — imported by eval.py
# ---------------------------------------------------------------------------
K_EVAL = 10   # NDCG@K cutoff used by the autograder


def normalize_page_id(pid) -> int:
    """
    Coerce a page_id to int.
    eval.py calls this to normalise IDs coming from both run() and the
    ground-truth JSON, so string/int mismatches don't hurt scoring.
    """
    return int(pid)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def ensure_artifacts_dir() -> None:
    """Create the artifacts/ directory if it doesn't exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def check_artifacts_present() -> bool:
    """Return True if all required artifact files exist."""
    required = [
        FAISS_INDEX_PATH,
        CHUNK_META_PATH,
        BM25_PREFIX + "_vocab.json",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing artifacts:", missing)
        return False
    return True