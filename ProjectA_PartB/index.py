#!/usr/bin/env python3
"""
index.py — Offline index construction.

Two indexes:

1. FAISS IVFFlat (dense ANN)
   - Inner-product metric (L2-normalised vectors → cosine via dot product)
   - Saved as: artifacts/faiss.index

2. BM25S sparse index (fast lexical search)
   - Uses bm25s library: precomputes scores at index time into scipy sparse matrices
   - 500x faster than JSON-dict BM25 at query time
   - Built with stemming (Snowball) for better recall on paraphrastic queries
   - Saved as: artifacts/bm25s_index/  (directory of bm25s internal files)

Supporting files:
   - artifacts/chunk_meta.json    — chunk → page_id mapping
   - artifacts/chunk_vectors.npy  — raw embeddings (for HyDE)
"""

import json
import math
import re
import numpy as np
import faiss
import bm25s

ARTIFACTS_DIR = "artifacts"


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Build IVFFlat index over L2-normalised vectors."""
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
    """Load FAISS index and move to GPU if available."""
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
# BM25S index
# ---------------------------------------------------------------------------

def build_bm25s_index(chunks: list[dict]) -> bm25s.BM25:
    """
    Build a BM25S index over chunk texts.

    BM25S precomputes all BM25 scores at index time and stores them
    as scipy sparse matrices — query time is a fast matrix slice.

    Uses Snowball stemmer for better recall on paraphrastic queries.
    """
    print("  Building BM25S index ...")
    corpus = [c["text"] for c in chunks]

    # Tokenize with stopword removal (stemmer API changed in bm25s 0.3+)
    try:
        import Stemmer
        stemmer = Stemmer.Stemmer("english")
    except ImportError:
        stemmer = None

    tokenized = bm25s.tokenize(
        corpus,
        stemmer=stemmer,
        stopwords="en",
        show_progress=True,
    )

    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(tokenized, show_progress=True)

    print(f"  BM25S index built: {len(corpus)} chunks.")
    return retriever


def save_bm25s_index(retriever: bm25s.BM25, path: str) -> None:
    """Save BM25S index to a directory."""
    import os
    os.makedirs(path, exist_ok=True)
    retriever.save(path)
    print(f"  Saved BM25S index → {path}/")


def load_bm25s_index(path: str) -> bm25s.BM25:
    """Load BM25S index from directory."""
    retriever = bm25s.BM25.load(path, load_corpus=False)
    print(f"  Loaded BM25S index from {path}/")
    return retriever


# ---------------------------------------------------------------------------
# Chunk metadata + vectors
# ---------------------------------------------------------------------------

def save_chunk_meta(chunks: list[dict], path: str) -> None:
    meta = [
        {"chunk_id": c["chunk_id"], "page_id": c["page_id"], "chunk_type": c["chunk_type"]}
        for c in chunks
    ]
    with open(path, "w") as f:
        json.dump(meta, f)
    print(f"  Saved chunk meta ({len(meta)} chunks) → {path}")


def load_chunk_meta(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_vectors(vectors: np.ndarray, path: str) -> None:
    np.save(path, vectors)
    print(f"  Saved vectors {vectors.shape} → {path}")


def load_vectors(path: str) -> np.ndarray:
    return np.load(path)