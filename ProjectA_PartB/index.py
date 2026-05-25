"""
index.py — Offline index construction.

Two indexes are built and saved to artifacts/:

1. FAISS IVFFlat (dense ANN)
   - Inner-product metric (vectors are L2-normalised, so IP == cosine).
   - nlist = max(64, sqrt(N)) centroids — good recall/speed trade-off.
   - Saved as: artifacts/faiss.index

2. BM25 inverted index (sparse keyword)
   - Pure Python, no extra dependencies.
   - Stores TF-IDF-style term weights per chunk.
   - Saved as: artifacts/bm25_index.json  (term → {chunk_id_str: score})
   - Saved as: artifacts/bm25_avgdl.json  ({"avgdl": float, "N": int})

Supporting files:
   - artifacts/chunk_meta.json   — list of chunk dicts (without embeddings)
   - artifacts/chunk_vectors.npy — raw float32 matrix (for HyDE)
"""

import json
import math
import re
import numpy as np
import faiss

ARTIFACTS_DIR = "artifacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build an IVFFlat index over the given L2-normalised vectors.

    Args:
        vectors: float32 array of shape (N, dim)

    Returns:
        Trained and populated faiss.Index
    """
    N, dim = vectors.shape
    nlist = max(64, int(math.sqrt(N)))

    # IVFFlat with inner-product metric
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


def load_faiss_index(path: str, nprobe: int = 64) -> faiss.Index:
    """
    Load index from disk, set nprobe, and move to GPU if CUDA is available.

    On GPU the IVF search runs ~10-30× faster than CPU for large corpora.
    Falls back to CPU silently if faiss-gpu is not installed or no GPU found.
    """
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
# BM25 inverted index
# ---------------------------------------------------------------------------

def build_bm25_index(
    chunks: list[dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[dict, dict]:
    """
    Build a BM25 inverted index over chunk texts.

    Args:
        chunks: list of chunk dicts (must have "chunk_id" and "text")
        k1, b:  BM25 hyperparameters

    Returns:
        (inverted_index, stats)
        inverted_index: {term: {chunk_id_str: bm25_score}}
        stats:          {"avgdl": float, "N": int}
    """
    print("  Building BM25 index ...")
    N = len(chunks)

    # Step 1: tokenise all chunks, compute doc lengths and df
    tokenised = []
    df: dict[str, int] = {}

    for chunk in chunks:
        tokens = _bm25_tokenize(chunk["text"])
        tokenised.append(tokens)
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    avgdl = sum(len(t) for t in tokenised) / max(N, 1)

    # Step 2: compute BM25 scores per (term, chunk) pair
    inverted: dict[str, dict[str, float]] = {}

    for idx, (chunk, tokens) in enumerate(zip(chunks, tokenised)):
        cid_str = str(chunk["chunk_id"])
        dl = len(tokens)
        tf_map: dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        for term, tf in tf_map.items():
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            score = idf * (numerator / denominator)

            if term not in inverted:
                inverted[term] = {}
            inverted[term][cid_str] = score

    stats = {"avgdl": avgdl, "N": N}
    print(f"  BM25 index built: {len(inverted)} unique terms, {N} chunks.")
    return inverted, stats


def save_bm25_index(inverted: dict, stats: dict, inv_path: str, stats_path: str) -> None:
    with open(inv_path, "w") as f:
        json.dump(inverted, f)
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"  Saved BM25 index → {inv_path}, {stats_path}")


def load_bm25_index(inv_path: str, stats_path: str) -> tuple[dict, dict]:
    with open(inv_path) as f:
        inverted = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)
    return inverted, stats


# ---------------------------------------------------------------------------
# Chunk metadata + vectors
# ---------------------------------------------------------------------------

def save_chunk_meta(chunks: list[dict], path: str) -> None:
    """Save chunk metadata (without embeddings) to JSON."""
    meta = [
        {
            "chunk_id":   c["chunk_id"],
            "page_id":    c["page_id"],
            "chunk_type": c["chunk_type"],
        }
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