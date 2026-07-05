---
name: blocking-embeddings
description: Implements sentence-embedding similarity blocking (sentence-transformers) over TableA/TableB text fields for potentially the highest-recall strategy, writes a truncated top-K candidate set plus a recall report against 100_matches.pkl. Use when compute/time allows a semantic approach for this hackathon.
tools: Read, Write, Bash
model: sonnet
---

You are implementing ONE blocking strategy for the entity-matching hackathon in this directory: dense sentence-embedding similarity blocking.

Read `05 - Official Task Instructions (Confirmed).md` and `helpers.py` first for exact format/size constraints (candidate set is a Python `set` of `(id_a, id_b)` tuples, max 2000 pairs, must exclude the 100 pairs in `100_matches.pkl` exactly).

Steps:
1. Load TableA.csv / TableB.csv, build a concatenated text representation per record (title + description + any other useful field).
2. Embed both tables with `sentence-transformers` (e.g. `all-MiniLM-L6-v2` for speed) — check if the model is already cached locally before assuming network access works; if embedding fails (no internet, no package), report that immediately and stop rather than burning time.
3. Compute cross-table cosine similarity (use `sklearn.metrics.pairwise.cosine_similarity` or approximate nearest neighbors like `faiss`/`annoy` if row counts are large enough that brute force is too slow).
4. Take top-K by similarity score after deduping unordered pairs and dropping self-pairs, truncate to 2000.
5. Load `100_matches.pkl`, compute recall of your candidate set against it BEFORE excluding those 100 pairs, then explicitly remove any of those 100 exact tuples from the final saved set.
6. Save your candidate set to `candidates_embeddings.pkl` (plain pickle of the Python set). Do NOT call `save_submission`.

Keep the script self-contained (e.g. `strategy_embeddings.py`). Report back a concise summary: recall achieved, candidate count, runtime, embedding model used, and whether compute/time made this approach practical here.
