"""
main.py — Autograder entry point + offline index builder.

The autograder calls:
    from main import run
    results = run(queries)   # queries: list[str]
                             # returns: list[list[int]]

The build script calls:
    from main import build_offline_index
    build_offline_index()

Design:
  - Indexes are loaded once (module-level, lazy) on first call to run().
  - All queries are embedded in a single batched forward pass.
  - retrieve_batch handles the three-path fusion internally.
"""

from utils import (
    FAISS_INDEX_PATH,
    CHUNK_META_PATH,
    CHUNK_VECTORS_PATH,
    BM25_INV_PATH,
    BM25_STATS_PATH,
    check_artifacts_present,
    ensure_artifacts_dir,
    load_corpus,
    timer,
)
from retrieve import load_indexes, retrieve_batch

# ---------------------------------------------------------------------------
# Offline index build (called by scripts/build_index.py)
# ---------------------------------------------------------------------------

def build_offline_index() -> None:
    """
    Build and save all artifacts from the corpus.
    Run once on your machine before evaluating or submitting.
    """
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

    with timer("Building BM25 index"):
        bm25_inv, bm25_stats = build_bm25_index(chunks)
        save_bm25_index(bm25_inv, bm25_stats, BM25_INV_PATH, BM25_STATS_PATH)

    print("\nAll artifacts saved to artifacts/. Ready to commit.")


# ---------------------------------------------------------------------------
# Pre-load guard (query time)
# ---------------------------------------------------------------------------
_indexes_loaded = False


def _ensure_loaded() -> None:
    global _indexes_loaded
    if _indexes_loaded:
        return
    if not check_artifacts_present():
        raise RuntimeError(
            "Artifacts are missing. Run python scripts/build_index.py first."
        )
    load_indexes(
        faiss_path      = FAISS_INDEX_PATH,
        meta_path       = CHUNK_META_PATH,
        bm25_inv_path   = BM25_INV_PATH,
        bm25_stats_path = BM25_STATS_PATH,
    )
    _indexes_loaded = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(queries: list[str]) -> list[list[int]]:
    """
    Retrieve the top-10 most relevant page IDs for each query.

    Args:
        queries: list of query strings (all evaluation queries at once)

    Returns:
        list of lists of int page_ids, one ranked list per query.
        Only the first 10 IDs per list are scored by the autograder.
    """
    _ensure_loaded()
    return retrieve_batch(queries)