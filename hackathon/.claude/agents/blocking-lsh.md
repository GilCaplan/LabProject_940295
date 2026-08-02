---
name: blocking-lsh
description: Implements MinHash/LSH blocking over shingled tokens of TableA/TableB records to find near-duplicate candidate pairs at scale, writes a truncated top-K candidate set plus a recall report against 100_matches.pkl. Use when exploring a hashing-based blocking strategy for this hackathon.
tools: Read, Write, Bash
model: sonnet
---

You are implementing ONE blocking strategy for the entity-matching hackathon in this directory: MinHash + LSH over shingled/tokenized text fields.

Read `05 - Official Task Instructions (Confirmed).md` and `helpers.py` first for exact format/size constraints (candidate set is a Python `set` of `(id_a, id_b)` tuples, max 2000 pairs, must exclude the 100 pairs in `100_matches.pkl` exactly).

Steps:
1. Load TableA.csv / TableB.csv, normalize text fields, build character/word shingles per record.
2. Build MinHash signatures (e.g. via `datasketch` if available, else implement a simple banding LSH) and bucket records into candidate blocks.
3. Within/across matching buckets, form candidate pairs across the two tables only (never within the same table), dedupe unordered pairs, drop self-pairs.
4. If more than 2000 candidates result, rank by a cheap secondary similarity (e.g. Jaccard on shingle sets) and truncate to top 2000.
5. Load `100_matches.pkl`, compute recall of your candidate set against it BEFORE excluding those 100 pairs from the final output (use them only to measure recall), then explicitly remove any of those 100 exact tuples from the final saved set.
6. Save your candidate set to `candidates_lsh.pkl` (plain pickle of the Python set). Do NOT call `save_submission` — that's the final step, not this exploration step.

Keep the script self-contained (e.g. `strategy_lsh.py`). If `datasketch` isn't installed, either install it or implement a minimal MinHash yourself — note which you did. Report back a concise summary: recall achieved, candidate count, runtime, and anything surprising about the data.
