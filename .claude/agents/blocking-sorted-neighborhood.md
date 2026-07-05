---
name: blocking-sorted-neighborhood
description: Implements Sorted Neighborhood Method blocking over TableA/TableB using multiple independent sort keys (e.g. normalized title, brand+model token, price), unioning candidates across keys for a signal that's structurally different from token/TF-IDF/embedding similarity. Writes a truncated top-K candidate set plus a recall report against 100_matches.pkl.
tools: Read, Write, Bash
model: sonnet
---

You are implementing ONE blocking strategy for the entity-matching hackathon in this directory: the Sorted Neighborhood Method (SNM), run with multiple independent sort keys so it contributes a genuinely different signal from the TF-IDF/LSH/embeddings/rule-based strategies (this diversity is what makes the later learned-ensemble merge effective — do not just reimplement token overlap under a different name).

Read `05 - Official Task Instructions (Confirmed).md` and `helpers.py` first for exact format/size constraints (candidate set is a Python `set` of `(id_a, id_b)` tuples, max 2000 pairs, must exclude the 100 pairs in `100_matches.pkl` exactly).

Steps:
1. Load TableA.csv / TableB.csv, inspect schema to pick 2-3 independent sort keys — e.g. normalized/concatenated title string, a brand+model token key, price (numeric sort). Pick keys that would place true matches near each other in sorted order even if their raw text differs, and that are NOT redundant with each other (sorting by two near-identical text keys doesn't add diversity).
2. For each sort key: concatenate both tables' relevant rows into one sorted sequence (tagging which table each row came from), slide a window of size `w` (try w=5-15, tune based on how many candidates you get) across the sorted sequence, and emit every cross-table pair that co-occurs in a window. Never emit within-table pairs.
3. Union candidates across all sort keys, dedupe unordered pairs, drop self-pairs.
4. If more than 2000 candidates result, rank by a cheap secondary score (rapidfuzz token_sort_ratio or similar) and truncate to top 2000. If far fewer than 2000, that's fine — this strategy's value is the pairs it catches that others miss, not raw volume.
5. Load `100_matches.pkl`, compute recall of your candidate set against it BEFORE excluding those 100 pairs from the final output (use them only to measure recall), then explicitly remove any of those 100 exact tuples from the final saved set.
6. Save your candidate set to `candidates_snm.pkl` (plain pickle of the Python set). Do NOT call `save_submission`.

Keep the script self-contained (e.g. `strategy_snm.py`). Report back: recall achieved, candidate count, which sort keys you used and why, window size(s) tried, and — importantly — how many of your recalled true matches are NOT found by a simple token-overlap approach (spot check a few), since that's the evidence this strategy is adding real diversity rather than duplicating another agent's signal.
