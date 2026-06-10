#!/usr/bin/env python3
"""
retrieve.py — Query-time scoring over the synthetic-page index.

Score per (query, page) — three normalized signals plus a number gate,
weights validated on the public query set:

  1. dense_max : best chunk cosine (MiniLM)  — multi-fact context match
  2. dense_sum : summary-chunk cosine        — lead sentence carries the
                 decisive template facts
  3. year      : idf-weighted exact year match, with decade expansion
                 ("1820s" -> 1820..1829) — years are generator slot values
                 and MiniLM is weak on digits
  4. nummatch  : exact non-year number overlap (populations, point totals)

  score = norm(dense_max) + norm(dense_sum) + norm(year) + 0.3 * nummatch
"""

import math
import re

import numpy as np

NUM_WEIGHT  = 0.3
FINAL_TOP_N = 10


# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------
class _Store:
    index = None

_store = _Store()


def load_indexes(npz_path: str, meta_path: str, **kwargs) -> None:
    if _store.index is not None:
        return
    from index import load_synth_index
    print("Loading index ...")
    _store.index = load_synth_index(npz_path, meta_path)
    # precompute per-page number sets
    _store.index["page_num_sets"] = [set(ns) for ns in _store.index["page_nums"]]
    _store.index["chunk_year_sets"] = [set(ys) for ys in _store.index["chunk_years"]]
    print("Index loaded.")


# ---------------------------------------------------------------------------
# Query slot extraction
# ---------------------------------------------------------------------------

def _expand_decades(t: str) -> list:
    m = re.match(r"(\d{3})0s$", t)
    if m:
        base = int(m.group(1)) * 10
        return [str(base + i) for i in range(10)]
    m = re.match(r"(\d{2})(\d{2})s$", t)
    if m:
        base = int(m.group(1) + m.group(2))
        return [str(base + i) for i in range(10)]
    return [t]


def _query_years(query: str) -> set:
    years = set()
    for t in re.findall(r"\b\d{3,4}0s\b|\b\d{4}\b", query.replace(",", "")):
        years.update(_expand_decades(t))
    return years


def _query_nums(query: str) -> set:
    nums = {m for m in re.findall(r"\b\d+(?:\.\d+)?\b", query.replace(",", ""))
            if not re.match(r"^\d{4}$", m)}
    return nums - {"10"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    mx = v.max()
    return v / mx if mx > 0 else v


def _score_query(query: str, qvec: np.ndarray) -> np.ndarray:
    idx = _store.index
    vectors    = idx["vectors"]
    chunk_page = idx["chunk_page"]
    is_summary = idx["is_summary"]
    n_pages    = len(idx["page_ids"])
    n_chunks   = len(vectors)

    sims = vectors @ qvec

    dense_max = np.zeros(n_pages, dtype=np.float32)
    np.maximum.at(dense_max, chunk_page, sims)

    dense_sum = np.zeros(n_pages, dtype=np.float32)
    np.maximum.at(dense_sum, chunk_page, np.where(is_summary, sims, -1.0))

    year_feat = np.zeros(n_pages, dtype=np.float32)
    q_years = _query_years(query)
    if q_years:
        year_df = idx["year_df"]
        chunk_scores = np.zeros(n_chunks, dtype=np.float32)
        for ci, ys in enumerate(idx["chunk_year_sets"]):
            hit = q_years & ys
            if hit:
                chunk_scores[ci] = sum(
                    math.log(1 + n_chunks / (1 + year_df.get(y, 0))) for y in hit
                )
        np.maximum.at(year_feat, chunk_page, chunk_scores)

    score = _norm(dense_max) + _norm(dense_sum) + _norm(year_feat)

    q_nums = _query_nums(query)
    if q_nums:
        nummatch = np.array(
            [len(q_nums & pn) / len(q_nums) for pn in idx["page_num_sets"]],
            dtype=np.float32,
        )
        score = score + NUM_WEIGHT * nummatch

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_batch(queries: list) -> list:
    """Return top-10 page IDs per query (batched query embedding)."""
    from embed import embed_queries
    qvecs = embed_queries(queries)

    page_ids = _store.index["page_ids"]
    results = []
    for qi, query in enumerate(queries):
        scores = _score_query(query, qvecs[qi])
        order = np.argsort(-scores)[:FINAL_TOP_N]
        results.append([int(page_ids[i]) for i in order])
    return results
