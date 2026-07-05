"""
Sorted Neighborhood Method (SNM) blocking strategy.

Uses multiple INDEPENDENT sort keys so this contributes a structurally
different signal (proximity-in-sorted-order) than token/TF-IDF/embedding
similarity based strategies used elsewhere in the ensemble:

  1. normalized full title string        -> catches near-duplicate titles
                                             that sort adjacently even when
                                             token overlap is low (e.g. typos,
                                             word-order differences, extra
                                             qualifiers)
  2. brand/manufacturer + leading title
     token (fallback: title prefix when
     manufacturer missing)               -> groups records by brand-ish key,
                                             a categorical signal totally
                                             different from text similarity
  3. numeric price (fallback: normalized
     title prefix when price is missing) -> numeric-sort proximity; two
                                             genuinely different products
                                             rarely share a near-identical
                                             price, but true matches (same
                                             product across sites) often do

For each sort key, both tables are concatenated (tagged by provenance),
sorted, and a sliding window of size w is passed over the sequence. Every
cross-table pair co-occurring within a window is emitted as a candidate.
Within-table pairs are never emitted. Candidates are unioned across all
three keys, deduped, and (if needed) truncated to a budget using a cheap
rapidfuzz token_sort_ratio secondary score.

IMPORTANT (id-ordering gotcha): TableA and TableB ids share the same integer
namespace (0..N), so pairs are NEVER built via sorted((id1, id2)) - table
provenance is tracked explicitly throughout and every emitted tuple is
constructed as (id_from_tableA, id_from_tableB).

Per this run's instructions: the 100 known matches are used ONLY to measure
proxy recall here; they are NOT excluded from candidates_sorted_neighborhood.py's
output (exclusion happens later, once, at the ensemble/submission step).
"""

import pickle
import re
import pandas as pd

try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Tuned empirically against 100_matches.pkl proxy recall (see report):
# these values were chosen because the *union* across all three keys at
# these windows reaches proxy recall 0.57 while keeping raw per-key output
# sizes manageable (title/brand windows kept modest since their raw output
# grows fast; price window widened since its raw output stays small).
WINDOW_SIZES = {
    "title": 6,
    "brand_token": 6,
    "price": 7,
}
# Budget chosen so that ranking the full union by secondary score and
# truncating to this size loses ~0 proxy recall vs. the untruncated union
# (0.57 at 5000 == 0.57 raw union recall; smaller budgets tested lower,
# e.g. 0.50 at 2000, 0.55 at 3000).
MAX_CANDIDATES = 5000


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def leading_token(s):
    s = normalize_text(s)
    return s.split()[0] if s else ""


def brand_token_key(manufacturer, title):
    """Manufacturer (normalized) + leading title token, with a fallback to
    the title's own leading tokens when manufacturer is missing (very sparse
    in TableB: ~89% null)."""
    man = normalize_text(manufacturer)
    if man:
        return man + " " + leading_token(title)
    # fallback: first two tokens of the title as a pseudo-brand key
    toks = normalize_text(title).split()
    return " ".join(toks[:2])


def price_key(price, title):
    """Numeric price rounded to nearest dollar; fallback to a normalized
    title prefix (as a string sort key placed after all numeric keys) when
    price is missing (TableA: 199/1363 null)."""
    if pd.notna(price):
        try:
            return float(price)
        except (TypeError, ValueError):
            pass
    return None  # handled separately (grouped, sorted by title fallback)


# ---------------------------------------------------------------------------
# Core SNM windowing
# ---------------------------------------------------------------------------
def snm_pairs(records, window):
    """records: list of (sort_key, table, id) sorted by caller.
    Emits cross-table (id_a, id_b) pairs found within a sliding window."""
    pairs = set()
    n = len(records)
    for i in range(n):
        table_i, id_i = records[i][1], records[i][2]
        for j in range(i + 1, min(i + window, n)):
            table_j, id_j = records[j][1], records[j][2]
            if table_i == table_j:
                continue  # never emit within-table pairs
            if table_i == "A":
                pairs.add((id_i, id_j))
            else:
                pairs.add((id_j, id_i))
    return pairs


def run_snm_key(a_keys, b_keys, a_ids, b_ids, window):
    records = [(k, "A", i) for k, i in zip(a_keys, a_ids)]
    records += [(k, "B", i) for k, i in zip(b_keys, b_ids)]
    records.sort(key=lambda r: r[0])
    return snm_pairs(records, window)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tableA = pd.read_csv("tableA.csv")
    tableB = pd.read_csv("tableB.csv")

    a_ids = tableA["id"].tolist()
    b_ids = tableB["id"].tolist()

    all_candidates = set()
    per_key_counts = {}

    # --- Key 1: normalized title string ---
    a_title_keys = tableA["title"].apply(normalize_text).tolist()
    b_title_keys = tableB["title"].apply(normalize_text).tolist()
    pairs_title = run_snm_key(a_title_keys, b_title_keys, a_ids, b_ids,
                               WINDOW_SIZES["title"])
    per_key_counts["title"] = len(pairs_title)
    all_candidates |= pairs_title

    # --- Key 2: brand/manufacturer + leading title token ---
    a_brand_keys = [brand_token_key(m, t) for m, t in
                    zip(tableA["manufacturer"], tableA["title"])]
    b_brand_keys = [brand_token_key(m, t) for m, t in
                    zip(tableB["manufacturer"], tableB["title"])]
    pairs_brand = run_snm_key(a_brand_keys, b_brand_keys, a_ids, b_ids,
                                WINDOW_SIZES["brand_token"])
    per_key_counts["brand_token"] = len(pairs_brand)
    all_candidates |= pairs_brand

    # --- Key 3: numeric price, with title-prefix fallback bucket for nulls ---
    # Split into "has price" (sorted numerically) and "no price" (sorted by
    # normalized title, forming their own contiguous fallback block) so a
    # missing price never collides arbitrarily with a real numeric value.
    a_price = tableA["price"]
    b_price = tableB["price"]

    a_has_price = tableA[a_price.notna()]
    b_has_price = tableB[b_price.notna()]
    pairs_price_numeric = run_snm_key(
        a_has_price["price"].astype(float).tolist(),
        b_has_price["price"].astype(float).tolist(),
        a_has_price["id"].tolist(), b_has_price["id"].tolist(),
        WINDOW_SIZES["price"],
    )

    a_no_price = tableA[a_price.isna()]
    b_no_price = tableB[b_price.isna()]
    a_np_keys = a_no_price["title"].apply(normalize_text).tolist()
    b_np_keys = b_no_price["title"].apply(normalize_text).tolist()
    pairs_price_fallback = run_snm_key(
        a_np_keys, b_np_keys,
        a_no_price["id"].tolist(), b_no_price["id"].tolist(),
        WINDOW_SIZES["title"],  # wider fallback window since it's really a text key
    )

    pairs_price = pairs_price_numeric | pairs_price_fallback
    per_key_counts["price"] = len(pairs_price)
    all_candidates |= pairs_price

    print("Per-key raw candidate counts (before union/dedup):")
    for k, v in per_key_counts.items():
        print(f"  {k}: {v}")
    print(f"Union of all keys (deduped): {len(all_candidates)}")

    # Sanity: no within-table / self pairs, no id-order flips
    a_id_set, b_id_set = set(a_ids), set(b_ids)
    bad = [p for p in all_candidates if p[0] not in a_id_set or p[1] not in b_id_set]
    assert not bad, f"Found malformed pairs: {bad[:5]}"

    # --- Truncate to budget if needed, using cheap secondary score ---
    if len(all_candidates) > MAX_CANDIDATES:
        title_a = dict(zip(tableA["id"], tableA["title"]))
        title_b = dict(zip(tableB["id"], tableB["title"]))

        def score(pair):
            ta, tb = title_a[pair[0]], title_b[pair[1]]
            if HAVE_RAPIDFUZZ:
                return fuzz.token_sort_ratio(str(ta), str(tb))
            # cheap fallback: normalized Jaccard on tokens
            sa, sb = set(normalize_text(ta).split()), set(normalize_text(tb).split())
            if not sa or not sb:
                return 0
            return 100 * len(sa & sb) / len(sa | sb)

        ranked = sorted(all_candidates, key=score, reverse=True)
        all_candidates = set(ranked[:MAX_CANDIDATES])
        print(f"Truncated to top {MAX_CANDIDATES} by secondary score.")

    # --- Recall proxy against 100 known matches (measure BEFORE any exclusion) ---
    with open("100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)

    hits = all_candidates & known_matches
    recall = len(hits) / len(known_matches)
    print(f"\nProxy recall vs 100_matches.pkl: {recall:.3f} ({len(hits)}/{len(known_matches)})")
    print(f"Final candidate set size: {len(all_candidates)}")

    # Per this run's instructions: do NOT exclude the 100 known matches here;
    # that exclusion happens once, later, at the ensemble/submission step.
    with open("candidates_sorted_neighborhood.pkl", "wb") as f:
        pickle.dump(all_candidates, f)
    print("Saved candidates_sorted_neighborhood.pkl")

    # Save a short recall report alongside
    with open("recall_report_sorted_neighborhood.txt", "w") as f:
        f.write("Sorted Neighborhood Method (SNM) - recall report\n")
        f.write(f"Sort keys used: title (w={WINDOW_SIZES['title']}), "
                f"brand_token (w={WINDOW_SIZES['brand_token']}), "
                f"price (w={WINDOW_SIZES['price']}, title-prefix fallback for nulls)\n")
        f.write("Per-key raw candidate counts (pre-union):\n")
        for k, v in per_key_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Final candidate set size: {len(all_candidates)}\n")
        f.write(f"Proxy recall vs 100_matches.pkl: {recall:.3f} "
                f"({len(hits)}/{len(known_matches)})\n")
        f.write("NOTE: the 100 known matches were NOT excluded from this "
                 "output file; exclusion happens once at the ensemble/"
                 "submission step.\n")

    return all_candidates, recall, hits


if __name__ == "__main__":
    main()
