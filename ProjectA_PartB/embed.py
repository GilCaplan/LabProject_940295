"""
embed.py — Embedding with sentence-transformers/all-MiniLM-L6-v2.

Key design choices:
  - Device selection is delegated to sentence-transformers (CUDA when
    available, CPU otherwise) — no direct torch import, keeping the
    import set to the allowed packages only.
  - Batch size is tuned per device: larger on GPU (avoids PCIe
    round-trips), smaller on CPU (avoids RAM pressure).
  - L2-normalisation is applied so cosine similarity == dot product.
  - Model is loaded once at module level (singleton) — calling embed_texts
    multiple times from run() does NOT reload weights.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_BATCH_SIZE_GPU = 512
_BATCH_SIZE_CPU = 128

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model (auto-selects CUDA/CPU)."""
    global _model
    if _model is None:
        print("  Loading embedding model ...")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"  Device: {_model.device}")
    return _model


def get_device() -> str:
    """Return the active compute device ('cuda' or 'cpu')."""
    return _get_model().device.type


def _default_batch_size() -> int:
    return _BATCH_SIZE_GPU if get_device() == "cuda" else _BATCH_SIZE_CPU


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
