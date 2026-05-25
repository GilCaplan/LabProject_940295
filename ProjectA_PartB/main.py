#!/usr/bin/env python3
"""
main.py — Autograder entry point + offline index builder.

Autograder calls:   from main import run; run(queries)
Build script calls: from main import build_offline_index; build_offline_index()
"""

from utils import (
    FAISS_INDEX_PATH,
    CHUNK_META_PATH,
    CHUNK_VECTORS_PATH,
    BM25_PREFIX,
    ensure_artifacts_dir,
    load_corpus,
    timer,
)
from retrieve import load_indexes, retrieve_batch


def build_offline_index() -> None:
    """Build and save all artifacts from the corpus."""
    from chunk import chunk_corpus
    from embed import embed_chunks
    from index import (
        build_faiss_index, save_faiss_index,
        build_bm25_index,  save_bm25_index,
        save_chunk_meta,   save_vectors,
    )

    ensure_artifacts_dir()

    with timer("Loading corpus"):
        pages = load_corpus()

    with timer("Chunking"):
        chunks = chunk_corpus(pages)
        print(f"  {len(chunks)} total chunks from {len(pages)} pages.")

    with timer("Embedding chunks"):
        vectors = embed_chunks(chunks, show_progress=True)

    with timer("Saving metadata and vectors"):
        save_chunk_meta(chunks, CHUNK_META_PATH)
        save_vectors(vectors, CHUNK_VECTORS_PATH)

    with timer("Building FAISS index"):
        faiss_index = build_faiss_index(vectors)
        save_faiss_index(faiss_index, FAISS_INDEX_PATH)

    with timer("Building BM25 numpy index"):
        bm25 = build_bm25_index(chunks)
        save_bm25_index(bm25, BM25_PREFIX)

    print("\nAll artifacts saved. Ready to commit.")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    import os
    missing = [p for p in [FAISS_INDEX_PATH, CHUNK_META_PATH, BM25_PREFIX + "_vocab.json"]
               if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}. Run python3 scripts/build_index.py first.")
    load_indexes(
        faiss_path  = FAISS_INDEX_PATH,
        meta_path   = CHUNK_META_PATH,
        bm25_prefix = BM25_PREFIX,
    )
    _loaded = True


def run(queries: list) -> list:
    """Return top-10 page IDs per query. Called by autograder."""
    _ensure_loaded()
    return retrieve_batch(queries)