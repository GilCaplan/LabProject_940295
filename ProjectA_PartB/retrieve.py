#!/usr/bin/env python3
"""
retrieve.py — BM25S retrieval with query-time optimizations.

Query-time improvements (no reindex needed):
  1. Snowball stemmer (PyStemmer) — "modernized" → "modern"
  2. Aggressive stopword removal — removes question words and common words
     that match every chunk equally
  3. Token repetition boosting — repeat numbers and long words to increase
     their BM25S weight (simulates field boosting)
  4. Decade expansion — "1820s" → ["1820","1821",...,"1829"]
  5. Year stream — separate BM25S search on year tokens only (high precision)

Research basis:
  - BM25S paper: stemmer + stopwords improves NDCG@10 from 38.4 → 39.7
  - BM25 field boosting: title tokens 5x more important than body
  - Query augmentation: token repetition boosts IDF weight at query time
"""

import re
from collections import defaultdict
import numpy as np
import faiss
import bm25s

BM25S_TOP_K = 200
RRF_K       = 60
FINAL_TOP_N = 10
BM25_WEIGHT = 8.0
YEAR_WEIGHT = 4.0

# Question words and stopwords to strip from BM25S query
STOPWORDS = {
    "who","what","where","when","which","whose","whom","how",
    "was","is","are","were","be","been","has","have","had",
    "the","a","an","of","in","on","at","to","for","with",
    "by","from","that","this","it","its","and","or","but",
    "did","do","does","their","there","about","during","after",
    "before","between","into","through","over","under",
}

# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------
class _Store:
    faiss_index = None
    bm25s_index = None
    chunk_meta  = None

_store = _Store()


def load_indexes(faiss_path, meta_path, bm25s_path, **kwargs):
    if _store.faiss_index is not None:
        return
    from index import load_faiss_index, load_chunk_meta, load_bm25s_index
    print("Loading indexes ...")
    _store.faiss_index = load_faiss_index(faiss_path)
    _store.chunk_meta  = load_chunk_meta(meta_path)
    _store.bm25s_index = load_bm25s_index(bm25s_path)
    print("Indexes loaded.")


# ---------------------------------------------------------------------------
# Stemmer (lazy loaded)
# ---------------------------------------------------------------------------
_stemmer = None

def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            import Stemmer
            _stemmer = Stemmer.Stemmer("english")
        except ImportError:
            _stemmer = False  # unavailable
    return _stemmer if _stemmer else None


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------

def _expand_decades(t: str) -> list:
    """'1820s' → ['1820','1821',...,'1829']"""
    m = re.match(r"(\d{3})0s", t)
    if m:
        base = int(m.group(1)) * 10
        return [str(base + i) for i in range(10)]
    m = re.match(r"(\d{2})(\d{2})s", t)
    if m:
        base = int(m.group(1) + m.group(2))
        return [str(base + i) for i in range(10)]
    return [t]


def _build_query_tokens(query: str) -> tuple[list, list]:
    """
    Build two token lists from query:
      main_tokens: stopword-filtered, stemmed, with boosted repetitions
      year_tokens: decade-expanded year tokens only

    Token boosting (simulates BM25F field weighting):
      - Numbers (years, stats): repeat 3x — very discriminative
      - Long words (>6 chars, likely names/entities): repeat 2x
      - Regular content words: appear once
    """
    text = re.sub(r"[^a-z0-9\s]", " ", query.lower())
    raw_tokens = text.split()

    stemmer = _get_stemmer()
    main_tokens = []
    year_tokens = []

    for t in raw_tokens:
        # Expand decades
        expanded = _expand_decades(t)

        if len(expanded) > 1 or (len(expanded) == 1 and re.match(r"\d{4}$", expanded[0])):
            # Year tokens — add to year stream, also add to main with boost
            year_tokens.extend(expanded)
            main_tokens.extend(expanded * 3)   # 3x boost for years
        elif t in STOPWORDS:
            pass  # skip
        elif re.match(r"\d+", t):
            # Other numbers — boost 3x
            main_tokens.extend([t] * 3)
        else:
            # Content word — stem and optionally boost
            word = stemmer.stemWord(t) if stemmer else t
            if len(t) > 6:
                main_tokens.extend([word, word, t])  # 2x stemmed + original
            else:
                main_tokens.append(word)

    return main_tokens, year_tokens


def _bm25s_retrieve(tokens: list, k: int = 200) -> list:
    """BM25S sparse retrieval — returns (chunk_id, score) list."""
    if not tokens:
        return []
    stemmer = _get_stemmer()
    tokenized = bm25s.tokenize(
        [" ".join(tokens)],
        stemmer=stemmer,
        stopwords="en",
        show_progress=False,
    )
    k = min(k, _store.bm25s_index.scores["data"].shape[0])
    results, scores = _store.bm25s_index.retrieve(tokenized, k=k)
    return [(int(results[0][i]), float(scores[0][i])) for i in range(len(results[0]))]


def _rrf_fuse(ranked_lists: list, weights: list = None) -> dict:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, (cid, _) in enumerate(ranked):
            scores[cid] += w / (RRF_K + rank + 1)
    return scores


def _to_pages(chunk_scores: dict) -> list:
    """Max-pool chunk scores → page scores. Summary chunks get 1.5x boost."""
    meta = _store.chunk_meta
    page_scores = defaultdict(float)
    for cid, score in chunk_scores.items():
        if cid >= len(meta):
            continue
        m = meta[cid]
        boosted = score * (1.5 if m["chunk_type"] == "summary" else 1.0)
        pid = int(m["page_id"])
        if boosted > page_scores[pid]:
            page_scores[pid] = boosted
    return sorted(page_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_batch(queries: list) -> list:
    """
    Retrieve top-10 page IDs per query using BM25S with query decomposition.

    For multi-hop queries ("What links X, Y, and Z"), decomposes into
    sub-queries and searches each independently, then fuses results.
    This ensures pages about X are not penalized by Y and Z terms.
    """
    all_page_results = []
    for query in queries:
        sub_queries = _decompose_query(query)
        is_multihop = len(sub_queries) > 1

        all_lists   = []
        all_weights = []

        if is_multihop:
            # Search each sub-query independently
            for sub_q in sub_queries:
                tokens = sub_q.lower().split()
                hits = _bm25s_retrieve(tokens, k=BM25S_TOP_K)
                if hits:
                    all_lists.append(hits)
                    all_weights.append(BM25_WEIGHT)
            # Also search full query for cross-cutting terms
            full_hits = _bm25s_retrieve(query.split(), k=BM25S_TOP_K)
            if full_hits:
                all_lists.append(full_hits)
                all_weights.append(BM25_WEIGHT / 2)

        else:
            # Single-hop: use full query
            hits = _bm25s_retrieve(query.split(), k=BM25S_TOP_K)
            if hits:
                all_lists.append(hits)
                all_weights.append(BM25_WEIGHT)

        # Year stream
        _, year_tokens = _build_query_tokens(query)
        if year_tokens:
            year_hits = _bm25s_retrieve(year_tokens, k=200)
            if year_hits:
                all_lists.append(year_hits)
                all_weights.append(YEAR_WEIGHT)

        if not all_lists:
            all_page_results.append([])
            continue

        chunk_scores = _rrf_fuse(all_lists, all_weights)
        page_ranked  = _to_pages(chunk_scores)
        top_pages    = [int(pid) for pid, _ in page_ranked[:FINAL_TOP_N]]
        all_page_results.append(top_pages)

    return all_page_results


# ---------------------------------------------------------------------------
# Query decomposition for multi-hop queries
# ---------------------------------------------------------------------------

def _decompose_query(query: str) -> list[str]:
    """
    Split multi-hop queries into sub-queries.

    Patterns detected:
      "What links X, Y, and Z" → ["X", "Y", "Z"]
      "How do A, B, and C" → ["A", "B", "C"]
      "Which X combines A with B" → ["A", "B"]
    
    Returns list of sub-queries, or [query] if no decomposition found.
    """
    q = query.strip()

    # Pattern: "What links A, B, and C" / "How do A, B, and C"
    m = re.match(
        r"(?:what links|how do|how does|what connects)\s+(.+)",
        q, re.IGNORECASE
    )
    if m:
        parts_str = m.group(1)
        # Split on ", " and " and "
        parts = re.split(r",\s+(?:and\s+)?|\s+and\s+", parts_str)
        parts = [p.strip().rstrip("?.,") for p in parts if len(p.strip()) > 5]
        if len(parts) >= 2:
            return parts

    # Pattern: "Which X combines A with B"
    m = re.match(r"which .+ combines (.+?) with (.+)", q, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip().rstrip("?.,")]

    return [query]