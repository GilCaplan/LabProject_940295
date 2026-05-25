#!/usr/bin/env python3
"""
chunk.py — Corpus chunking for the retrieval pipeline.

Strategy:
  1. Summary chunk: title + first 2 sentences → high-signal, used for topic-level queries.
  2. Body chunks:   sliding window over remaining content (window=200 tokens, stride=150).

Title is prepended to every chunk so the embedding space "knows" the topic
even when a chunk is deep in the article.

Chunk dict schema:
  {
    "chunk_id":   int,         # global unique index
    "page_id":    int,
    "chunk_type": "summary" | "body",
    "text":       str          # what gets embedded
  }
"""

import re
from typing import Generator


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
WINDOW_TOKENS  = 100   # approximate tokens per body chunk (1 token ≈ 1 word)
STRIDE_TOKENS  = 75   # stride between windows (50-token overlap)
MIN_CHUNK_TOKENS = 15  # discard tiny tail chunks


def _tokenize(text: str) -> list[str]:
    """Whitespace-split tokenisation (fast, no external deps)."""
    return text.split()


def _sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation heuristics."""
    # Split on . ! ? followed by whitespace or end-of-string
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def _sliding_windows(tokens: list[str], window: int, stride: int) -> Generator[str, None, None]:
    """Yield overlapping token windows rejoined as strings."""
    start = 0
    while start < len(tokens):
        chunk_tokens = tokens[start : start + window]
        if len(chunk_tokens) >= MIN_CHUNK_TOKENS:
            yield " ".join(chunk_tokens)
        start += stride
        if start >= len(tokens):
            break


def chunk_page(page: dict, base_id: int = 0) -> list[dict]:
    """
    Chunk a single Wikipedia page into summary + body chunks.

    Args:
        page:    dict with keys "page_id", "title", "content"
        base_id: starting chunk_id for this page

    Returns:
        List of chunk dicts, first element is always the summary chunk.
    """
    page_id = int(page["page_id"])
    title   = page["title"].strip()
    content = page["content"].strip()

    chunks = []
    cid = base_id

    # ------------------------------------------------------------------
    # 1. Summary chunk — title + first 2 sentences
    # ------------------------------------------------------------------
    sentences = _sentences(content)
    summary_body = " ".join(sentences[:2]) if sentences else content[:300]
    summary_text = f"{title}. {summary_body}"

    chunks.append({
        "chunk_id":   cid,
        "page_id":    page_id,
        "chunk_type": "summary",
        "text":       summary_text,
    })
    cid += 1

    # ------------------------------------------------------------------
    # 2. Body chunks — sliding window over full content
    #    We prepend the title to each body chunk so the embedding
    #    retains topic context even for mid-article segments.
    # ------------------------------------------------------------------
    tokens = _tokenize(content)
    for window_text in _sliding_windows(tokens, WINDOW_TOKENS, STRIDE_TOKENS):
        chunks.append({
            "chunk_id":   cid,
            "page_id":    page_id,
            "chunk_type": "body",
            "text":       f"{title}. {window_text}",
        })
        cid += 1

    return chunks


def chunk_corpus(pages: list[dict]) -> list[dict]:
    """
    Chunk every page in the corpus and assign globally unique chunk_ids.

    Args:
        pages: list of page dicts (page_id, title, content)

    Returns:
        Flat list of all chunk dicts across the corpus.
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_page(page, base_id=len(all_chunks))
        all_chunks.extend(page_chunks)
    return all_chunks