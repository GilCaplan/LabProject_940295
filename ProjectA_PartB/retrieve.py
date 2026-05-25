#!/usr/bin/env python3
"""
retrieve.py — Query-time retrieval with three-path fusion.

Three retrieval paths, fused via Reciprocal Rank Fusion (RRF):

  Path 1 — Dense ANN
    Embed the raw query, search the FAISS index for the top-K nearest chunks.

  Path 2 — BM25 sparse
    Tokenise the query, score every chunk using the prebuilt BM25 index.
    Handles exact keyword matches (entity names, numbers) that MiniLM may
    miss due to vocabulary mismatch.

  Path 3 — HyDE (Hypothetical Document Embedding)  ★ creative
    Construct a short "hypothetical answer" sentence for the query, embed it,
    and run a second dense search.  The hypothesis bridges the style gap
    between a terse query and long Wikipedia prose.

Fusion:
  RRF score = Σ  1 / (RRF_K + rank_i)   over all retriever lists.
  No learnt weights needed; robust across query types.

Aggregation:
  Chunk scores are rolled up to page scores by taking the MAX chunk score
  per page (a page is as relevant as its best-matching chunk).
"""

import re
import math
from collections import defaultdict

import numpy as np
import faiss

from embed import embed_queries, embed_texts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAISS_TOP_K  = 100   # candidates retrieved per dense pass
BM25_TOP_K   = 100   # candidates from BM25
RRF_K        = 60    # RRF smoothing constant (standard value)
FINAL_TOP_N  = 10    # page IDs to return per query

# Multiple HyDE templates — different phrasings improve recall
_HYDE_TEMPLATES = [
    "This Wikipedia article answers the query '{query}'. It discusses the relevant facts, background, and key details.",
    "A Wikipedia page about {query} would describe the specific person, place, event, or concept involved.",
    "The answer to '{query}' can be found in a Wikipedia article covering the topic in detail.",
]


# ---------------------------------------------------------------------------
# Singleton index store (loaded once, reused across all queries)
# ---------------------------------------------------------------------------
class _IndexStore:
    faiss_index:   faiss.Index | None = None
    chunk_meta:    list[dict]  | None = None   # [{chunk_id, page_id, chunk_type}]
    bm25_inverted: dict        | None = None   # {term: {cid_str: score}}
    bm25_stats:    dict        | None = None   # {avgdl, N}

_store = _IndexStore()


def load_indexes(
    faiss_path: str,
    meta_path:  str,
    bm25_inv_path: str,
    bm25_stats_path: str,
    nprobe: int = 32,
) -> None:
    """
    Load all prebuilt indexes into the module-level singleton.
    Call this once at the top of run() (it no-ops on repeated calls).
    """
    if _store.faiss_index is not None:
        return  # already loaded

    from index import load_faiss_index, load_chunk_meta, load_bm25_index

    print("Loading indexes ...")
    _store.faiss_index   = load_faiss_index(faiss_path, nprobe=nprobe)
    _store.chunk_meta    = load_chunk_meta(meta_path)
    _store.bm25_inverted, _store.bm25_stats = load_bm25_index(bm25_inv_path, bm25_stats_path)
    print("Indexes loaded.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bm25_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def _dense_retrieve(query_vecs: np.ndarray) -> list[list[tuple[int, float]]]:
    """
    Run FAISS ANN for a batch of query vectors.

    Returns:
        For each query: list of (chunk_id, score) sorted descending.
    """
    index = _store.faiss_index
    scores_mat, ids_mat = index.search(query_vecs, FAISS_TOP_K)

    results = []
    for scores_row, ids_row in zip(scores_mat, ids_mat):
        hits = []
        for score, cid in zip(scores_row, ids_row):
            if cid >= 0:  # FAISS uses -1 for padding
                hits.append((int(cid), float(score)))
        results.append(hits)
    return results


# Terms appearing in more than this fraction of chunks are skipped (too common)
_BM25_DF_THRESHOLD = 0.05

def _bm25_retrieve(query_tokens: list[str]) -> list[tuple[int, float]]:
    """
    Score chunks against query tokens using the prebuilt BM25 index.
    Skips high-df terms (appear in >5% of chunks) — they are slow and noisy.

    Returns:
        List of (chunk_id, bm25_score) sorted descending, top BM25_TOP_K.
    """
    inverted = _store.bm25_inverted
    N = _store.bm25_stats["N"]
    max_df = int(N * _BM25_DF_THRESHOLD)
    acc: dict[int, float] = defaultdict(float)

    for token in set(query_tokens):
        if token not in inverted:
            continue
        postings = inverted[token]
        if len(postings) > max_df:
            continue   # skip high-frequency terms
        for cid_str, score in postings.items():
            acc[int(cid_str)] += score

    ranked = sorted(acc.items(), key=lambda x: x[1], reverse=True)
    return ranked[:BM25_TOP_K]


def _hyde_texts(query: str) -> list[str]:
    """Return all HyDE hypothesis strings for a query."""
    return [t.format(query=query) for t in _HYDE_TEMPLATES]


def _rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]],
    weights: list[float] | None = None,
) -> dict[int, float]:
    """
    Weighted Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        ranked_lists: each list is [(chunk_id, score), ...] sorted best-first
        weights:      per-list multipliers (default: all 1.0)

    Returns:
        dict mapping chunk_id → RRF score
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, (cid, _) in enumerate(ranked):
            rrf_scores[cid] += w / (RRF_K + rank + 1)
    return rrf_scores


def _chunk_scores_to_page_scores(
    chunk_scores: dict[int, float],
) -> list[tuple[int, float]]:
    """
    Aggregate chunk-level RRF scores to page-level scores.

    Strategy: max-pool — a page score = its best chunk score.
    Summary chunks get a 1.2× boost (they are more topic-representative).
    """
    meta = _store.chunk_meta  # [{chunk_id, page_id, chunk_type}]
    page_scores: dict[int, float] = defaultdict(float)

    for cid, score in chunk_scores.items():
        if cid >= len(meta):
            continue
        m = meta[cid]
        boosted = score * (1.5 if m["chunk_type"] == "summary" else 1.0)
        pid = m["page_id"]
        if boosted > page_scores[pid]:
            page_scores[pid] = boosted

    ranked = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_batch(queries: list[str]) -> list[list[int]]:
    """
    Retrieve top-10 page IDs for each query in the batch.

    Single batched embed pass for all queries + all HyDE hypotheses.
    Weighted RRF: BM25 gets 2x weight (outperforms dense on fictional corpus).
    Multiple HyDE templates improve recall for paraphrastic queries.

    Args:
        queries: list of query strings

    Returns:
        list of lists of page_id ints, one per query, most relevant first
    """
    Q = len(queries)
    H = len(_HYDE_TEMPLATES)

    # --- Build all HyDE hypotheses for all queries ---
    # Layout: [q0_h0, q0_h1, q0_h2, q1_h0, q1_h1, q1_h2, ...]
    hyde_texts = []
    for q in queries:
        hyde_texts.extend(_hyde_texts(q))

    # --- Single batched embed: queries + all hypotheses ---
    all_texts = list(queries) + hyde_texts
    all_vecs  = embed_queries(all_texts)           # (Q + Q*H, 384)
    query_vecs = all_vecs[:Q]                      # (Q, 384)
    hyde_vecs  = all_vecs[Q:].reshape(Q, H, -1)   # (Q, H, 384)

    # --- Batch FAISS search for queries ---
    dense_results = _dense_retrieve(query_vecs)    # list of Q results

    all_page_results = []

    for i, query in enumerate(queries):
        # Path 1: dense hits (weight 1.0)
        dense_hits = dense_results[i]

        # Path 2: BM25 sparse hits (weight 2.0 — stronger on fictional corpus)
        tokens = _bm25_tokenize(query)
        sparse_hits = _bm25_retrieve(tokens)

        # Path 3: multiple HyDE hits (weight 1.0 each)
        hyde_hit_lists = []
        for h in range(H):
            h_vec = hyde_vecs[i, h:h+1, :]        # (1, 384)
            h_results = _dense_retrieve(h_vec)
            hyde_hit_lists.append(h_results[0])

        # Weighted RRF: dense=1, bm25=2, each hyde=1
        all_lists  = [dense_hits, sparse_hits] + hyde_hit_lists
        all_weights = [1.0, 4.0] + [1.0] * H
        chunk_scores = _rrf_fuse(all_lists, all_weights)

        # Roll up to page level
        page_ranked = _chunk_scores_to_page_scores(chunk_scores)

        # Take top-10 page IDs
        top_pages = [int(pid) for pid, _ in page_ranked[:FINAL_TOP_N]]
        all_page_results.append(top_pages)

    return all_page_results