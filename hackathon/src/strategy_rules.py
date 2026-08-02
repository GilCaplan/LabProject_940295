"""
Rule-based blocking strategy (strategy_rules.py)
=================================================

Fast, cheap "first signal" baseline for the entity-matching hackathon.
Builds candidate (id_a, id_b) pairs between TableA (software titles, 1363
rows) and TableB (3226 rows) using three inexpensive blocking keys built
directly from the columns that actually exist (title, manufacturer, price):

  1. Normalized-manufacturer exact match.
     manufacturer is populated in both tables (A: fully populated, B: only
     356/3226 non-null) so this key alone can't carry recall, but where it
     IS present it's a very strong, cheap signal (punctuation/hyphen
     normalized, e.g. "sibelius-software-ltd ." -> "sibelius software ltd").

  2. Shared significant title token (inverted-index blocking).
     The dominant signal. Titles are short product names, so two records
     sharing a distinctive token (length >= 4, not a stopword, not a
     token that is near-universal in either table) are good match
     candidates. Tokens that are too common (appear in a large fraction of
     either table, e.g. "software", "edition") are dropped from the index
     before pairing to avoid combinatorial blow-up and low-precision noise.

  3. Close price match (tight relative tolerance) AND >=1 shared title
     token. Price alone is a weak/noisy signal here (heavy-tailed,
     multiple products can coincidentally share a price), so it's only
     used as a *conjunctive* tie-breaker key with token overlap, not
     standalone -- this catches a few pairs where the shared token barely
     missed key 2's frequency filter but price nails it down.

Candidates from all keys are unioned (dedup via a plain Python `set`,
built by always inserting (id_from_tableA, id_from_tableB) explicitly --
never `sorted()`, since TableA/TableB ids share the same 0..N integer
namespace and sorting would silently swap table provenance).

If the union exceeds the target cap, pairs are ranked by a composite score
-- rapidfuzz token_set_ratio on normalized titles (robust to TableB's much
longer, padded product descriptions), plus a manufacturer-exact-match
bonus, plus the shared-token-overlap count from key 2 -- and truncated to
the top N. (token_set_ratio was chosen over token_sort_ratio/WRatio, and
this composite over a hard "key-vote" tiering scheme, after empirically
comparing proxy recall against 100_matches.pkl -- see the truncation code
comment below for the numbers.)

Per this task's instructions, the 100 known matches in 100_matches.pkl
are used ONLY to measure a proxy recall number here -- they are NOT
excluded from this strategy's output (candidates_rules.pkl). Exclusion of
the 100 known matches happens later, once, at the final ensemble/
submission step.

Output: candidates_rules.pkl -- a plain pickle of the Python `set` of
(id_a, id_b) tuples.
"""

import pickle
import re
import time
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_CAP = 2000          # final candidate set size cap for this strategy
MAX_TOKEN_DF_FRAC = 0.03   # drop title tokens present in >3% of either table
MIN_TOKEN_LEN = 4
PRICE_REL_TOL = 0.01       # 1% relative tolerance for the price+token key
STOPWORDS = {
    "the", "and", "for", "with", "from", "your", "this", "that", "edition",
    "software", "version", "pro", "plus", "new", "deluxe", "full", "mac",
    "win", "windows", "cd", "dvd", "rom", "upgrade", "retail", "box",
}

DATA_DIR = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"


def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s):
    return [t for t in normalize_text(s).split() if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]


def main():
    t0 = time.time()

    tableA = pd.read_csv(f"{DATA_DIR}/TableA.csv")
    tableB = pd.read_csv(f"{DATA_DIR}/TableB.csv")

    with open(f"{DATA_DIR}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)

    print(f"TableA: {tableA.shape}, TableB: {tableB.shape}")
    print(f"Known matches (proxy labels): {len(known_matches)}")

    # -----------------------------------------------------------------
    # Precompute normalized fields
    # -----------------------------------------------------------------
    tableA = tableA.copy()
    tableB = tableB.copy()
    tableA["manu_norm"] = tableA["manufacturer"].apply(normalize_text)
    tableB["manu_norm"] = tableB["manufacturer"].apply(normalize_text)
    tableA["title_norm"] = tableA["title"].apply(normalize_text)
    tableB["title_norm"] = tableB["title"].apply(normalize_text)
    tableA["tokens"] = tableA["title"].apply(tokenize)
    tableB["tokens"] = tableB["title"].apply(tokenize)

    candidates = set()  # set of (id_a, id_b) -- id_a from TableA, id_b from TableB, never sorted
    manu_match_pairs = set()      # pairs where Key 1 fired
    token_overlap_count = defaultdict(int)  # pair -> number of shared blocking tokens (Key 2)
    price_match_pairs = set()     # pairs where Key 3 fired

    # -----------------------------------------------------------------
    # Key 1: normalized manufacturer exact match
    # -----------------------------------------------------------------
    manu_map_b = defaultdict(list)
    for _, row in tableB.iterrows():
        if row["manu_norm"]:
            manu_map_b[row["manu_norm"]].append(row["id"])

    k1_count = 0
    for _, rowA in tableA.iterrows():
        m = rowA["manu_norm"]
        if not m:
            continue
        for idb in manu_map_b.get(m, []):
            pair = (rowA["id"], idb)
            candidates.add(pair)
            manu_match_pairs.add(pair)
            k1_count += 1
    print(f"Key 1 (manufacturer exact match): {k1_count} raw pairs")

    # -----------------------------------------------------------------
    # Key 2: shared significant title token (inverted index blocking)
    # -----------------------------------------------------------------
    inv_a = defaultdict(list)
    for _, row in tableA.iterrows():
        for tok in set(row["tokens"]):
            inv_a[tok].append(row["id"])

    inv_b = defaultdict(list)
    for _, row in tableB.iterrows():
        for tok in set(row["tokens"]):
            inv_b[tok].append(row["id"])

    max_df_a = MAX_TOKEN_DF_FRAC * len(tableA)
    max_df_b = MAX_TOKEN_DF_FRAC * len(tableB)

    k2_count = 0
    shared_tokens = set(inv_a.keys()) & set(inv_b.keys())
    for tok in shared_tokens:
        ids_a = inv_a[tok]
        ids_b = inv_b[tok]
        if len(ids_a) > max_df_a or len(ids_b) > max_df_b:
            continue  # too common in either table -> low precision, skip
        for ida in ids_a:
            for idb in ids_b:
                pair = (ida, idb)
                candidates.add(pair)
                token_overlap_count[pair] += 1
                k2_count += 1
    print(f"Key 2 (shared title token, {len(shared_tokens)} tokens considered): {k2_count} raw pairs")
    print(f"Union size after key 1+2: {len(candidates)}")

    # -----------------------------------------------------------------
    # Key 3: close price match (tight relative tolerance) + >=1 shared token
    # -----------------------------------------------------------------
    priceB = tableB[["id", "price"]].dropna(subset=["price"])
    priceB_sorted = priceB.sort_values("price").reset_index(drop=True)
    prices_b = priceB_sorted["price"].values
    ids_b_sorted = priceB_sorted["id"].values

    import bisect

    tokens_b_map = dict(zip(tableB["id"], tableB["tokens"]))

    k3_count = 0
    for _, rowA in tableA.iterrows():
        pa = rowA["price"]
        if pd.isna(pa) or pa <= 0:
            continue
        lo = pa * (1 - PRICE_REL_TOL)
        hi = pa * (1 + PRICE_REL_TOL)
        left = bisect.bisect_left(prices_b, lo)
        right = bisect.bisect_right(prices_b, hi)
        if left >= right:
            continue
        tokens_a = set(rowA["tokens"])
        if not tokens_a:
            continue
        for idb in ids_b_sorted[left:right]:
            rowB_tokens = set(tokens_b_map[idb])
            if tokens_a & rowB_tokens:
                pair = (rowA["id"], idb)
                candidates.add(pair)
                price_match_pairs.add(pair)
                k3_count += 1
    print(f"Key 3 (price match + shared token): {k3_count} raw pairs")
    print(f"Union size after key 1+2+3: {len(candidates)}")

    # -----------------------------------------------------------------
    # Proxy recall BEFORE any truncation
    # -----------------------------------------------------------------
    recall_before_trunc = len(candidates & known_matches) / len(known_matches)
    print(f"Proxy recall vs 100_matches.pkl (before truncation): {recall_before_trunc:.3f}")

    # -----------------------------------------------------------------
    # Rank + truncate to TARGET_CAP if needed.
    #
    # Several ranking functions were tried empirically against the proxy
    # labels (100_matches.pkl) before settling on this one:
    #   - rapidfuzz token_sort_ratio alone:            proxy recall 0.54
    #   - a "key-vote" tiering scheme (# keys agreeing
    #     dominates, fuzzy score only a tie-breaker):   proxy recall 0.31
    #   - rapidfuzz token_set_ratio alone:              proxy recall 0.66
    #   - token_set_ratio + manufacturer-match bonus
    #     + shared-token-overlap count (this formula):  proxy recall 0.71
    #
    # token_set_ratio (not token_sort_ratio/WRatio) matters a lot here
    # because TableB titles are frequently long, padded product descriptions
    # ("... software full version for windows", "... win 98 me nt 2000 xp")
    # while TableA titles are short -- token_set_ratio ignores the extra
    # words in the longer string instead of penalizing the length mismatch.
    # The manufacturer-match and token-overlap-count terms then nudge
    # otherwise-similarly-scored pairs that also agree on manufacturer or
    # share more distinctive tokens higher, without letting them completely
    # dominate/override the fuzzy title similarity the way a hard vote-tier
    # scheme did.
    # -----------------------------------------------------------------
    if len(candidates) > TARGET_CAP:
        titleA_map = dict(zip(tableA["id"], tableA["title_norm"]))
        titleB_map = dict(zip(tableB["id"], tableB["title_norm"]))

        scored = []
        for pair in candidates:
            ida, idb = pair
            fuzzy = fuzz.token_set_ratio(titleA_map[ida], titleB_map[idb])
            manu_bonus = 25 if pair in manu_match_pairs else 0
            overlap = token_overlap_count.get(pair, 0)
            composite = fuzzy + manu_bonus + overlap
            scored.append((composite, ida, idb))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = {(ida, idb) for _, ida, idb in scored[:TARGET_CAP]}
        print(f"Truncated to top {TARGET_CAP} pairs by composite score "
              f"(token_set_ratio + manufacturer-match bonus + token overlap count)")
    else:
        print("No truncation needed -- union already within target cap")

    # -----------------------------------------------------------------
    # Final proxy recall (this strategy's output does NOT exclude the
    # 100 known matches -- per task instructions, exclusion happens once
    # at the final ensemble/submission step)
    # -----------------------------------------------------------------
    recall_final = len(candidates & known_matches) / len(known_matches)
    print(f"Final candidate set size: {len(candidates)}")
    print(f"Proxy recall vs 100_matches.pkl (final, pre-exclusion output): {recall_final:.3f}")

    with open(f"{DATA_DIR}/candidates_rules.pkl", "wb") as f:
        pickle.dump(candidates, f)
    print(f"Saved {len(candidates)} candidate pairs to candidates_rules.pkl")

    elapsed = time.time() - t0
    print(f"Runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
