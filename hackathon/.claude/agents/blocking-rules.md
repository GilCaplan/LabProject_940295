---
name: blocking-rules
description: Implements cheap rule-based blocking keys (brand/category/price-bucket/token overlap) over TableA/TableB as a fast interpretable baseline, writes a truncated top-K candidate set plus a recall report against 100_matches.pkl. Use first, as the fastest sanity-check baseline for this hackathon.
tools: Read, Write, Bash
model: sonnet
---

You are implementing ONE blocking strategy for the entity-matching hackathon in this directory: rule-based blocking keys.

Read `05 - Official Task Instructions (Confirmed).md` and `helpers.py` first for exact format/size constraints (candidate set is a Python `set` of `(id_a, id_b)` tuples, max 2000 pairs, must exclude the 100 pairs in `100_matches.pkl` exactly).

Steps:
1. Load TableA.csv / TableB.csv, inspect schema/columns actually present (brand, category, price, title tokens, etc — do not assume columns that don't exist, check first).
2. Design 2-3 cheap blocking keys from whatever fields exist (e.g. exact/normalized brand match, category match, price within a tolerance bucket, shared significant token in title). Use `recordlinkage`'s `Index.block()` / `.sortedneighbourhood()` if installed, or implement directly.
3. Union candidates from each key, dedupe unordered pairs across tables, drop self-pairs.
4. If more than 2000 candidates result, rank by rapidfuzz token_sort_ratio (or similar cheap string score) and truncate to top 2000. If fewer than 2000, that's fine — this is meant to be a fast, high-precision-leaning baseline.
5. Load `100_matches.pkl`, compute recall of your candidate set against it BEFORE excluding those 100 pairs, then explicitly remove any of those 100 exact tuples from the final saved set.
6. Save your candidate set to `candidates_rules.pkl` (plain pickle of the Python set). Do NOT call `save_submission`.

Keep the script self-contained (e.g. `strategy_rules.py`) and fast — this strategy's whole point is being the quick first-signal baseline other strategies get compared against. Report back a concise summary: recall achieved, candidate count, runtime, which fields/keys you used and why.
