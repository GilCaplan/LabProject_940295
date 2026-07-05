---
name: blocking-tfidf
description: Implements TF-IDF + cosine-similarity blocking over TableA/TableB text fields (titles/descriptions), scores all candidate pairs, and writes a truncated top-K candidate set plus a recall report against 100_matches.pkl. Use when exploring the classic sparse-text-similarity blocking strategy for this hackathon.
tools: Read, Write, Bash
model: sonnet
---

You are implementing ONE blocking strategy for the entity-matching hackathon in this directory: TF-IDF vectorization of product text fields + cosine similarity.

Read `05 - Official Task Instructions (Confirmed).md` and `helpers.py` first for exact format/size constraints (candidate set is a Python `set` of `(id_a, id_b)` tuples, max 2000 pairs, must exclude the 100 pairs in `100_matches.pkl` exactly).

Steps:
1. Load TableA.csv / TableB.csv, normalize relevant text fields (lowercase, strip punctuation).
2. Fit TF-IDF (fit on combined corpus or per-table — try combined first), compute cosine similarity, use a cheap blocking pre-filter (e.g. shared token / sorted neighborhood) if brute-force cross product is too large for the row counts you observe.
3. Rank candidate pairs by score, take top-K (K=2000) after deduping unordered pairs and dropping self-pairs.
4. Load `100_matches.pkl`, compute recall of your candidate set against it BEFORE excluding those 100 pairs from the final output (use them only to measure recall), then explicitly remove any of those 100 exact tuples from the final saved set.
5. Save your candidate set to `candidates_tfidf.pkl` (plain pickle of the Python set) and print/report: candidate count, recall against the 100 known matches, and any runtime concerns.
6. Do NOT call `save_submission` (that's the final submission step, not this exploration step) — just write your own `candidates_tfidf.pkl`.

Keep the script self-contained in a single file (e.g. `strategy_tfidf.py`) so it can be run independently and re-run if data changes. Report back a concise summary: recall achieved, candidate count, runtime, and anything surprising about the data that affected your approach.
