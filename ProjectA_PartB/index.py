#!/usr/bin/env python3
"""
index.py — Offline index construction.

Two indexes:

1. FAISS IVFFlat (dense ANN)
   - Saved as: artifacts/faiss.index

2. BM25 numpy sparse index (fast lexical search, numpy-only)
   - Precomputes BM25 scores at index time into CSR-style numpy arrays
   - Query time = numpy slice + sum — no Python loops, very fast
   - Saved as: artifacts/bm25_data.npy, bm25_indices.npy,
               bm25_indptr.npy, bm25_vocab.json, bm25_stats.json

Supporting files:
   - artifacts/chunk_meta.json   — chunk → page_id mapping
   - artifacts/chunk_vectors.npy — raw embeddings
"""

import json
import math
import re
import numpy as np
import faiss

ARTIFACTS_DIR = "artifacts"


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


def _tokenize(text: str) -> list:
    """Lowercase, strip punctuation, stem, split."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for t in text.split():
        stemmed = _stemmer.stem(t)
        tokens.append(stemmed)
        if stemmed != t:
            tokens.append(t)
    return tokens


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    N, dim = vectors.shape
    nlist = max(64, int(math.sqrt(N)))
    quantiser = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    print(f"  Training FAISS IVFFlat (N={N}, nlist={nlist}) ...")
    index.train(vectors)
    index.add(vectors)
    print(f"  Index built: {index.ntotal} vectors.")
    return index


def save_faiss_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)
    print(f"  Saved FAISS index → {path}")


def load_faiss_index(path: str, nprobe: int = 32) -> faiss.Index:
    import torch
    index = faiss.read_index(path)
    index.nprobe = nprobe
    if torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("  FAISS index moved to GPU.")
        except Exception as e:
            print(f"  FAISS GPU unavailable ({e}), using CPU.")
    return index


# ---------------------------------------------------------------------------
# BM25 numpy sparse index
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list, k1: float = 1.5, b: float = 0.75) -> dict:
    """
    Build BM25 index stored as CSR-style numpy arrays.
    Uses chunk-first approach: iterate chunks once, accumulate COO arrays,
    then sort by term to build CSR. Much faster than per-term lists.
    """
    print("  Building BM25 numpy index ...")
    N = len(chunks)

    # Pass 1: tokenize, compute df and avgdl
    print("  Pass 1: tokenizing ...")
    tokenized = []
    df = {}
    for i, chunk in enumerate(chunks):
        tokens = _tokenize(chunk["text"])
        tokenized.append(tokens)
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
        if i % 50000 == 0:
            print(f"    {i}/{N} chunks tokenized")

    avgdl = sum(len(t) for t in tokenized) / max(N, 1)
    max_df = int(N * 0.5)
    vocab = {term: idx for idx, term in enumerate(
        sorted(t for t, d in df.items() if d <= max_df)
    )}
    V = len(vocab)
    print(f"  Vocab: {V} terms, avgdl={avgdl:.1f}")

    # Pass 2: build COO arrays (term_idx, chunk_id, score)
    print("  Pass 2: scoring ...")
    coo_terms   = []
    coo_chunks  = []
    coo_scores  = []

    for cid, (chunk, tokens) in enumerate(zip(chunks, tokenized)):
        dl = len(tokens)
        tf_map = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        for term, tf in tf_map.items():
            if term not in vocab:
                continue
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            coo_terms.append(vocab[term])
            coo_chunks.append(cid)
            coo_scores.append(score)
        if cid % 50000 == 0:
            print(f"    {cid}/{N} chunks scored")

    # Convert to numpy and sort by term for CSR
    print("  Sorting and building CSR ...")
    coo_terms  = np.array(coo_terms,  dtype=np.int32)
    coo_chunks = np.array(coo_chunks, dtype=np.int32)
    coo_scores = np.array(coo_scores, dtype=np.float32)

    order   = np.argsort(coo_terms, kind="stable")
    indices = coo_chunks[order]
    data    = coo_scores[order]
    terms_sorted = coo_terms[order]

    # Build indptr
    indptr = np.zeros(V + 1, dtype=np.int32)
    np.add.at(indptr[1:], terms_sorted, 1)
    np.cumsum(indptr, out=indptr)

    print(f"  BM25 index: {V} terms, {N} chunks, {len(data)} entries.")
    return {"vocab": vocab, "data": data, "indices": indices, "indptr": indptr, "N": N}


def save_bm25_index(bm25: dict, prefix: str) -> None:
    np.save(prefix + "_data.npy",    bm25["data"])
    np.save(prefix + "_indices.npy", bm25["indices"])
    np.save(prefix + "_indptr.npy",  bm25["indptr"])
    with open(prefix + "_vocab.json", "w") as f:
        json.dump(bm25["vocab"], f)
    with open(prefix + "_stats.json", "w") as f:
        json.dump({"N": bm25["N"]}, f)
    print(f"  Saved BM25 index → {prefix}_*.npy/.json")


def load_bm25_index(prefix: str) -> dict:
    data    = np.load(prefix + "_data.npy")
    indices = np.load(prefix + "_indices.npy")
    indptr  = np.load(prefix + "_indptr.npy")
    with open(prefix + "_vocab.json") as f:
        vocab = json.load(f)
    with open(prefix + "_stats.json") as f:
        stats = json.load(f)
    print(f"  Loaded BM25 index: {len(vocab)} terms.")
    return {"vocab": vocab, "data": data, "indices": indices, "indptr": indptr, "N": stats["N"]}


def bm25_query(bm25: dict, tokens: list, k: int = 200) -> list:
    """
    Fast BM25 query using numpy CSR operations.
    No Python loops over chunks — pure numpy vectorized scoring.
    """
    vocab   = bm25["vocab"]
    data    = bm25["data"]
    indices = bm25["indices"]
    indptr  = bm25["indptr"]
    N       = bm25["N"]

    # Accumulate scores using numpy
    scores = np.zeros(N, dtype=np.float32)
    seen = set()
    for token in tokens:
        if token not in vocab or token in seen:
            continue
        seen.add(token)
        row = vocab[token]
        start, end = int(indptr[row]), int(indptr[row + 1])
        if end > start:
            scores[indices[start:end]] += data[start:end]

    # Top-k via argpartition (O(n) not O(n log n))
    if k >= N:
        top_k = np.argsort(scores)[::-1]
    else:
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]

    return [(int(cid), float(scores[cid])) for cid in top_k if scores[cid] > 0]


# ---------------------------------------------------------------------------
# Chunk metadata + vectors
# ---------------------------------------------------------------------------

def save_chunk_meta(chunks: list, path: str) -> None:
    meta = [{"chunk_id": c["chunk_id"], "page_id": c["page_id"], "chunk_type": c["chunk_type"]}
            for c in chunks]
    with open(path, "w") as f:
        json.dump(meta, f)
    print(f"  Saved chunk meta ({len(meta)} chunks) → {path}")


def load_chunk_meta(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def save_vectors(vectors: np.ndarray, path: str) -> None:
    np.save(path, vectors)
    print(f"  Saved vectors {vectors.shape} → {path}")


def load_vectors(path: str) -> np.ndarray:
    return np.load(path)