"""
Entity Matching Blocking — Solution Starter Pipeline
Mini Hackathon: given TableA.csv, TableB.csv, and 100_matches.pkl (known true
matches), produce a set of at most 2000 (id_a, id_b) candidate pairs maximizing
recall against a hidden ground-truth set.

This is copy-paste-adapt starter code, not a finished submission. See:
  - "05 - Official Task Instructions (Confirmed).md" for the rules/gotchas
  - "01 - Blocking Strategies Cheat Sheet.md" for technique tradeoffs
  - "03 - Python Blocking Toolkit.md" for the individual building blocks this
    script assembles

Before running: inspect TableA.csv / TableB.csv columns and update TEXT_COLS
and ID_COL below to match the real schema.
"""

import pickle
import re
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

from helpers import validate_candidate_set, save_submission

# ---------------------------------------------------------------------------
# Config — adjust after inspecting the real CSVs
# ---------------------------------------------------------------------------
SID = "<sid>"          # fill in right before final submission
ID_COL = "id"          # id column name in both tables
TEXT_COLS = ["title"]  # text columns to concatenate for blocking/scoring;
                       # add more (brand, category, modelno, price...) once you
                       # see the real schema — more discriminative fields help
MAX_PAIRS = 2000
STOPWORDS = {"inc", "the", "a", "an", "and", "of", "with", "for",
             "edition", "version", "co", "corp", "ltd"}

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
tableA = pd.read_csv("TableA.csv")
tableB = pd.read_csv("TableB.csv")

with open("100_matches.pkl", "rb") as f:
    known_matches = pickle.load(f)  # set of (id_a, id_b), the 100 given true matches

tableA_ids = set(tableA[ID_COL])
tableB_ids = set(tableB[ID_COL])

# Gotcha check: if ids collide across tables, never sort((id_a, id_b)) when
# deduping below — always build pairs as (id_from_A, id_from_B) explicitly.
shared_ids = tableA_ids & tableB_ids
if shared_ids:
    print(f"NOTE: {len(shared_ids)} ids shared between tables — do not sort pair tuples.")

# ---------------------------------------------------------------------------
# 2. Normalize
# ---------------------------------------------------------------------------
def normalize(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_tokens(s) -> list:
    return [t for t in normalize(s).split() if t not in STOPWORDS]


def row_text(row) -> str:
    return " ".join(normalize(row[c]) for c in TEXT_COLS if c in row)


id_to_text = {}
for _, row in tableA.iterrows():
    id_to_text[row[ID_COL]] = row_text(row)
for _, row in tableB.iterrows():
    id_to_text[row[ID_COL]] = row_text(row)

# ---------------------------------------------------------------------------
# 3. Candidate generation — multiple cheap high-recall passes, unioned
# ---------------------------------------------------------------------------
def token_blocking(dfA, dfB, max_block_size=500):
    """Inverted index on normalized tokens. Returns set of (id_a, id_b)."""
    inverted = defaultdict(lambda: [[], []])
    for _, row in dfA.iterrows():
        for tok in set(normalize_tokens(row_text(row))):
            inverted[tok][0].append(row[ID_COL])
    for _, row in dfB.iterrows():
        for tok in set(normalize_tokens(row_text(row))):
            inverted[tok][1].append(row[ID_COL])

    pairs = set()
    for tok, (ids_a, ids_b) in inverted.items():
        if not ids_a or not ids_b:
            continue
        if len(ids_a) * len(ids_b) > max_block_size * max_block_size:
            continue  # drop overly-common tokens
        for a in ids_a:
            for b in ids_b:
                pairs.add((a, b))
    return pairs


def sorted_neighborhood(dfA, dfB, key_fn, window=5):
    """Concat both tables, sort by key_fn, slide a window, keep cross-table pairs."""
    combined = []
    for _, row in dfA.iterrows():
        combined.append((key_fn(row), row[ID_COL], "A"))
    for _, row in dfB.iterrows():
        combined.append((key_fn(row), row[ID_COL], "B"))
    combined.sort(key=lambda x: x[0])

    pairs = set()
    n = len(combined)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            _, id_i, src_i = combined[i]
            _, id_j, src_j = combined[j]
            if src_i == src_j:
                continue
            id_a, id_b = (id_i, id_j) if src_i == "A" else (id_j, id_i)
            pairs.add((id_a, id_b))
    return pairs


candidates = set()
candidates |= token_blocking(tableA, tableB)
candidates |= sorted_neighborhood(tableA, tableB, key_fn=row_text, window=5)

# Optional: semantic/embedding pass (uncomment if sentence-transformers/faiss
# are installed and there's time budget — usually the single best recall boost)
#
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
#
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embA = model.encode(tableA.apply(row_text, axis=1).tolist(), normalize_embeddings=True)
# embB = model.encode(tableB.apply(row_text, axis=1).tolist(), normalize_embeddings=True)
# index = faiss.IndexFlatIP(embB.shape[1])
# index.add(np.array(embB, dtype="float32"))
# _, neighbors = index.search(np.array(embA, dtype="float32"), 15)  # top_k=15 per row
# ids_a_list = tableA[ID_COL].tolist()
# ids_b_list = tableB[ID_COL].tolist()
# for i, nbrs in enumerate(neighbors):
#     for j in nbrs:
#         if j != -1:
#             candidates.add((ids_a_list[i], ids_b_list[j]))

print(f"candidate pairs before scoring: {len(candidates)}")

# ---------------------------------------------------------------------------
# 4. Use the 100 known matches to sanity-check recall so far (proxy metric)
# ---------------------------------------------------------------------------
def proxy_recall(pair_set):
    return len(pair_set & known_matches) / len(known_matches)


print(f"proxy recall of raw candidate pool: {proxy_recall(candidates):.3f}")
# If this is well below 1.0, your blocking passes are too narrow — add more
# passes (different key fields) or loosen max_block_size / window before
# worrying about scoring/truncation.

# ---------------------------------------------------------------------------
# 5. Score every candidate with a similarity signal
# ---------------------------------------------------------------------------
scored = []
for a, b in candidates:
    score = fuzz.token_sort_ratio(id_to_text.get(a, ""), id_to_text.get(b, ""))
    scored.append((score, a, b))
scored.sort(reverse=True)

# ---------------------------------------------------------------------------
# 6. Dedupe + truncate to budget (global top-K / Cardinality Edge Pruning)
# ---------------------------------------------------------------------------
def finalize(scored_pairs, k):
    seen = set()
    out = []
    for score, a, b in scored_pairs:
        if a == b:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
        if len(out) >= k:
            break
    return out


final_list = finalize(scored, MAX_PAIRS)
candidate_set = set(final_list)

print(f"proxy recall after scoring/truncation to {MAX_PAIRS}: {proxy_recall(candidate_set):.3f}")

# ---------------------------------------------------------------------------
# 7. Required: strip the 100 given matches from the final output, then top up
#    from the remaining ranked pool to stay close to the 2000 budget.
# ---------------------------------------------------------------------------
removed = candidate_set & known_matches
candidate_set -= known_matches

if removed:
    for _, a, b in scored:  # scored is already sorted best-first
        if len(candidate_set) >= MAX_PAIRS:
            break
        pair = (a, b)
        if pair in known_matches or pair in candidate_set:
            continue
        candidate_set.add(pair)

print(f"final candidate set size: {len(candidate_set)}")
assert candidate_set.isdisjoint(known_matches), "Known matches leaked into final set!"

# ---------------------------------------------------------------------------
# 8. Validate + save
# ---------------------------------------------------------------------------
validate_candidate_set(candidate_set, tableA_ids, tableB_ids)
save_submission(candidate_set, SID)
