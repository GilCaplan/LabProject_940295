"""
embed.py — Embedding with sentence-transformers/all-MiniLM-L6-v2.

Key design choices:
  - CUDA is used automatically when available; falls back to CPU silently.
  - Batch size is tuned per device: larger on GPU (avoids PCIe round-trips),
    smaller on CPU (avoids RAM pressure).
  - FAISS GPU index is also supported — see index.py.
  - L2-normalisation is applied so cosine similarity == dot product,
    which lets us use FAISS IndexFlatIP (inner product) as our metric.
  - Model is loaded once at module level (singleton) — calling embed_texts
    multiple times from run() does NOT reload weights.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Larger batches on GPU saturate the tensor cores; smaller on CPU avoids swap.
_BATCH_SIZE_GPU = 512
_BATCH_SIZE_CPU = 128

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_device() -> str:
    """Return the active compute device ('cuda' or 'cpu')."""
    return _DEVICE


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model on the right device."""
    global _model
    if _model is None:
        print(f"  Loading embedding model on {_DEVICE.upper()} ...")
        _model = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
        if _DEVICE == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")
    return _model


def _default_batch_size() -> int:
    return _BATCH_SIZE_GPU if _DEVICE == "cuda" else _BATCH_SIZE_CPU


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    batch_size: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of strings and return an L2-normalised float32 matrix.

    Args:
        texts:         list of strings to embed
        batch_size:    encoding batch size (auto-selected per device if None)
        show_progress: show tqdm progress bar (useful during offline build)

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32, L2-normalised
    """
    if batch_size is None:
        batch_size = _default_batch_size()

    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2 norm → cosine via dot product
        convert_to_numpy=True,
        device=_DEVICE,
    )
    return vectors.astype(np.float32)


def embed_chunks(chunks: list[dict], batch_size: int | None = None, show_progress: bool = True) -> np.ndarray:
    """
    Convenience wrapper: embed the "text" field of each chunk dict.

    Returns:
        np.ndarray of shape (len(chunks), 384)
    """
    texts = [c["text"] for c in chunks]
    return embed_texts(texts, batch_size=batch_size, show_progress=show_progress)


def embed_queries(queries: list[str]) -> np.ndarray:
    """
    Embed a batch of user queries.
    No progress bar — queries are few and fast.

    Returns:
        np.ndarray of shape (len(queries), 384)
    """
    return embed_texts(queries, show_progress=False)