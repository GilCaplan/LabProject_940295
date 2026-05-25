#!/usr/bin/env python3
"""
retrieve.py — Fast BM25 retrieval with query decomposition.

Uses numpy CSR sparse BM25 (no scipy/bm25s) — compliant with import rules.
Speed comes from numpy vectorized scoring instead of Python loops.

Key techniques:
  - Query decomposition: "What links X, Y, Z" → 3 independent searches
  - Year expansion: "1820s" → ["1820".."1829"]
  - Porter stemmer (pure Python): "modernized" → "modern"
  - Weighted RRF fusion
  - Summary chunk 1.5x boost
"""

import re
from collections import defaultdict
import numpy as np
import faiss

BM25_TOP_K  = 200
RRF_K       = 60
FINAL_TOP_N = 10
BM25_WEIGHT = 8.0
YEAR_WEIGHT = 4.0


# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------
class _Store:
    faiss_index = None
    bm25_index  = None
    chunk_meta  = None

_store = _Store()


def load_indexes(faiss_path, meta_path, bm25_prefix, **kwargs):
    if _store.faiss_index is not None:
        return
    from index import load_faiss_index, load_chunk_meta, load_bm25_index
    print("Loading indexes ...")
    _store.faiss_index = load_faiss_index(faiss_path)
    _store.chunk_meta  = load_chunk_meta(meta_path)
    _store.bm25_index  = load_bm25_index(bm25_prefix)
    print("Indexes loaded.")


# ---------------------------------------------------------------------------
# Porter stemmer (pure Python, no deps)
# ---------------------------------------------------------------------------
class _PorterStemmer:
    def __init__(self):
        self._vowels = set("aeiou")

    def _cons(self, word, i):
        if word[i] in self._vowels: return False
        if word[i] == "y": return i == 0 or not self._cons(word, i - 1)
        return True

    def _m(self, word):
        n, i, L = 0, 0, len(word)
        while i < L and not self._cons(word, i): i += 1
        while i < L:
            while i < L and not self._cons(word, i): i += 1
            while i < L and self._cons(word, i): i += 1
            n += 1
        return n

    def _has_vowel(self, word):
        return any(not self._cons(word, i) for i in range(len(word)))

    def stem(self, word):
        if len(word) <= 2: return word
        w = word.lower()
        if w.endswith("sses"): w = w[:-2]
        elif w.endswith("ies"): w = w[:-2]
        elif w.endswith("ss"): pass
        elif w.endswith("s"): w = w[:-1]
        if w.endswith("eed"):
            if self._m(w[:-3]) > 0: w = w[:-1]
        elif w.endswith("ed"):
            s = w[:-2]
            if self._has_vowel(s):
                w = s
                if w.endswith(("at","bl","iz")): w += "e"
                elif len(w)>1 and self._cons(w,-1) and self._cons(w,-2) and w[-1]==w[-2] and w[-1] not in "lsz": w=w[:-1]
        elif w.endswith("ing"):
            s = w[:-3]
            if self._has_vowel(s):
                w = s
                if w.endswith(("at","bl","iz")): w += "e"
                elif len(w)>1 and self._cons(w,-1) and self._cons(w,-2) and w[-1]==w[-2] and w[-1] not in "lsz": w=w[:-1]
        if w.endswith("y") and len(w)>1 and self._has_vowel(w[:-1]): w=w[:-1]+"i"
        for suf,rep in [("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
                        ("izer","ize"),("bli","ble"),("alli","al"),("entli","ent"),("eli","e"),
                        ("ousli","ous"),("ization","ize"),("ation","ate"),("ator","ate"),
                        ("alism","al"),("iveness","ive"),("fulness","ful"),("ousness","ous"),
                        ("aliti","al"),("iviti","ive"),("biliti","ble")]:
            if w.endswith(suf) and self._m(w[:-len(suf)])>0: w=w[:-len(suf)]+rep; break
        for suf,rep in [("icate","ic"),("ative",""),("alize","al"),("iciti","ic"),("ical","ic"),("ful",""),("ness","")]:
            if w.endswith(suf) and self._m(w[:-len(suf)])>0: w=w[:-len(suf)]+rep; break
        for suf in ["al","ance","ence","er","ic","able","ible","ant","ement","ment","ent","ion","ou","ism","ate","iti","ous","ive","ize"]:
            stem=w[:-len(suf)]
            if w.endswith(suf) and self._m(stem)>1:
                if suf=="ion" and stem and stem[-1] in "st": w=stem
                elif suf!="ion": w=stem
                break
        if w.endswith("e"):
            s=w[:-1]
            if self._m(s)>1: w=s
            elif self._m(s)==1: w=s
        if len(w)>1 and w[-1]==w[-2]=="l" and self._cons(w,-1) and self._m(w)>1: w=w[:-1]
        return w

_stemmer = _PorterStemmer()


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------

def _expand_decades(t: str) -> list:
    m = re.match(r"(\d{3})0s", t)
    if m:
        base = int(m.group(1)) * 10
        return [str(base + i) for i in range(10)]
    m = re.match(r"(\d{2})(\d{2})s", t)
    if m:
        base = int(m.group(1) + m.group(2))
        return [str(base + i) for i in range(10)]
    return [t]


def _tokenize_query(query: str) -> tuple:
    """
    Returns (main_tokens, year_tokens).
    Main tokens are stemmed. Years are expanded from decades.
    """
    text = re.sub(r"[^a-z0-9\s]", " ", query.lower())
    main_tokens = []
    year_tokens = []

    for t in text.split():
        expanded = _expand_decades(t)
        if len(expanded) > 1 or (len(expanded)==1 and re.match(r"\d{4}$", expanded[0])):
            year_tokens.extend(expanded)
            main_tokens.extend(expanded)
        else:
            stemmed = _stemmer.stem(t)
            main_tokens.append(stemmed)
            if stemmed != t:
                main_tokens.append(t)

    return main_tokens, year_tokens


def _decompose_query(query: str) -> list:
    """Split multi-hop queries into sub-queries."""
    q = query.strip()
    m = re.match(r"(?:what links|how do|how does|what connects)\s+(.+)", q, re.IGNORECASE)
    if m:
        parts = re.split(r",\s+(?:and\s+)?|\s+and\s+", m.group(1))
        parts = [p.strip().rstrip("?.,") for p in parts if len(p.strip()) > 5]
        if len(parts) >= 2:
            return parts
    m = re.match(r"which .+ combines (.+?) with (.+)", q, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip().rstrip("?.,")]
    return [query]


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _bm25_retrieve(tokens: list, k: int = 200) -> list:
    """Fast numpy CSR BM25 retrieval."""
    from index import bm25_query
    return bm25_query(_store.bm25_index, tokens, k=k)


def _rrf_fuse(ranked_lists: list, weights: list = None) -> dict:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, (cid, _) in enumerate(ranked):
            scores[cid] += w / (RRF_K + rank + 1)
    return scores


def _to_pages(chunk_scores: dict) -> list:
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
    Retrieve top-10 page IDs per query.
    Multi-hop queries decomposed into sub-queries for better recall.
    Fast numpy BM25 scoring — no Python loops over chunks.
    """
    all_page_results = []
    for query in queries:
        sub_queries  = _decompose_query(query)
        is_multihop  = len(sub_queries) > 1
        all_lists    = []
        all_weights  = []

        if is_multihop:
            for sub_q in sub_queries:
                tokens, _ = _tokenize_query(sub_q)
                hits = _bm25_retrieve(tokens, k=BM25_TOP_K)
                if hits:
                    all_lists.append(hits)
                    all_weights.append(BM25_WEIGHT)
            # Full query search too
            full_tokens, _ = _tokenize_query(query)
            full_hits = _bm25_retrieve(full_tokens, k=BM25_TOP_K)
            if full_hits:
                all_lists.append(full_hits)
                all_weights.append(BM25_WEIGHT / 2)
        else:
            tokens, _ = _tokenize_query(query)
            hits = _bm25_retrieve(tokens, k=BM25_TOP_K)
            if hits:
                all_lists.append(hits)
                all_weights.append(BM25_WEIGHT)

        # Year stream
        _, year_tokens = _tokenize_query(query)
        if year_tokens:
            year_hits = _bm25_retrieve(year_tokens, k=200)
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