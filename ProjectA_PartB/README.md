# Section B — Wikipedia Retrieval Pipeline

## Team
[Team member 1] · [Team member 2]

## Video presentation
[Link to video — max 3:00, at most 10 slides]

---

## Pipeline overview

```
Offline (your machine)           Online (autograder, ≤60 s)
─────────────────────────        ─────────────────────────────
corpus JSON                      queries
    │                                │
chunk.py                         embed.py (batch)
  sliding window +                   │
  summary chunks                 ┌───┴──────────────┐
    │                          dense             HyDE
embed.py                       FAISS           embed hypo-
  all-MiniLM-L6-v2             search          thetical doc
    │                            │                  │
index.py                      BM25 sparse      dense FAISS
  FAISS IVFFlat                 search              │
  BM25 inverted                   └───────┬──────────┘
    │                                  RRF fusion
artifacts/                              │
  faiss.index                     page aggregation
  chunk_meta.json                  (max-pool + summary boost)
  bm25_index.json                       │
  bm25_stats.json               top-10 page IDs
  chunk_vectors.npy
```

### Key creative contributions

| Stage | Technique | Why |
|-------|-----------|-----|
| Chunking | Summary chunk (title + first 2 sentences) | High-signal anchors for topic-level queries |
| Chunking | Title-prefixed body chunks | Every chunk retains article context |
| Retrieval | BM25 sparse path | Exact keyword matching (names, numbers) that MiniLM misses |
| Retrieval | **HyDE** — hypothetical document embedding | Bridges query/document vocabulary gap |
| Fusion | Reciprocal Rank Fusion (RRF) | No tuned weights; robust across query types |
| Aggregation | Summary chunk 1.2× boost | Rewards pages whose summary matches well |

---

## Artifacts

All files live under `artifacts/` and are committed to this repo (`.npy` and `.index` via Git LFS).

| File | Format | Description |
|------|--------|-------------|
| `faiss.index` | FAISS binary | IVFFlat index over all chunk embeddings (inner product) |
| `chunk_meta.json` | JSON list | `[{chunk_id, page_id, chunk_type}]` — maps FAISS row → page |
| `chunk_vectors.npy` | float32 numpy | Raw L2-normalised embeddings, shape `(N_chunks, 384)` |
| `bm25_index.json` | JSON dict | `{term: {chunk_id_str: bm25_score}}` |
| `bm25_stats.json` | JSON dict | `{avgdl: float, N: int}` |

---

## Setup

```bash
pip install -r requirements.txt
```

## Build index (once, on your machine)

```bash
python scripts/build_index.py
```

Reads from `data/Wikipedia Entries/`, writes all files to `artifacts/`.
Takes ~5–15 minutes depending on corpus size and hardware.

## Evaluate on public queries

```bash
python scripts/eval_public.py
```

Prints mean NDCG@10 on the 50 public queries.

---

## File guide

| File | Purpose |
|------|---------|
| `main.py` | `run(queries)` — autograder entry point |
| `chunk.py` | Sliding-window chunker with summary chunks |
| `embed.py` | MiniLM embedding, batched, L2-normalised |
| `index.py` | FAISS and BM25 index build + load utilities |
| `retrieve.py` | Three-path retrieval + RRF fusion |
| `utils.py` | Corpus loading, path constants, timing |
| `eval.py` | NDCG@10 utilities (read-only) |
| `scripts/build_index.py` | Offline index build (read-only) |
| `scripts/eval_public.py` | Public query self-evaluation (read-only) |

---

## Results

| System | Mean NDCG@10 (public) |
|--------|-----------------------|
| Dense only (baseline) | — |
| + BM25 fusion | — |
| + HyDE | — |
| + Summary boost (final) | — |

*(Fill in after running ablations)*