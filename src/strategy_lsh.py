"""
MinHash + LSH blocking strategy for entity matching hackathon.

Approach:
  1. Load TableA / TableB, normalize `title` (+ `manufacturer` when present, though it's
     very sparse in TableB: 2870/3226 null) into a single lowercase text field per record.
  2. Build a *combined* word-shingle set per record: unigrams (word tokens) union
     word-bigrams. Titles are short (median ~6 tokens), so pure bigrams alone throw away
     too much signal for short titles, and pure unigrams are too coarse (poor precision
     for ranking) -- the union of both gave the best recall/size tradeoff in local tuning
     (see parameter sweep notes below).
  3. Build MinHash signatures with `datasketch.MinHash` (num_perm=128) over these combined
     shingle sets, and index TableB records in a `datasketch.MinHashLSH` (Jaccard threshold
     0.2, deliberately low/recall-oriented since LSH banding is an approximate filter).
  4. For each TableA record, query the LSH index for approximate-match TableB candidates ->
     candidate (id_a, id_b) pairs. NEVER sort the tuple -- id_a always comes from TableA,
     id_b always from TableB. The two id namespaces fully overlap here (both 0..N-ish
     ints), so sorting would silently corrupt provenance -- confirmed via
     `set(tableA.id) & set(tableB.id)` (1363 ids in common) before building anything.
  5. Rank ALL raw LSH candidate pairs by exact Jaccard similarity of their combined shingle
     sets (cheap secondary similarity, exact recomputation rather than the MinHash estimate)
     and keep the top MAX_CANDIDATES. This is an intermediate/exploration file feeding an
     ensemble merge step, not the final submission, so the cap is set to a few thousand
     (3000) rather than the hard 2000 pair submission cap -- ranking+truncating a much
     larger raw candidate pool (10k-50k+) straight to 2000 costs a lot of recall (see sweep
     below), so we keep extra headroom for the ensemble step to work with.
  6. Measure proxy recall against 100_matches.pkl. Per this task's specific instructions,
     these 100 known pairs are NOT excluded from candidates_lsh.pkl -- exclusion of the
     exact 100 tuples happens later, once, at the ensemble/final-submission step.
  7. Save the final candidate set (a plain Python `set` of (id_a, id_b) tuples) to
     candidates_lsh.pkl via pickle.

Library note: `datasketch` (v1.10.0) IS installed in this environment, so we use its
MinHash/MinHashLSH implementation directly rather than hand-rolling banded LSH.

Parameter sweep notes (recall vs candidate-set size, proxy against 100_matches.pkl):
  shingle=bigrams only,  thresh=0.2  -> raw 9,680 pairs,  raw recall 0.83; capped@2000 -> 0.70
  shingle=bigrams only,  thresh=0.1  -> raw 16,183 pairs, raw recall 0.89; capped@2000 -> 0.69
  shingle=unigrams only, thresh=0.3  -> raw 18,610 pairs, raw recall 0.88
  shingle=uni+bigrams,   thresh=0.2  -> raw 33,938 pairs, raw recall 0.92; capped@3000 -> 0.82
  shingle=uni+bigrams,   thresh=0.15 -> raw 53,338 pairs, raw recall 0.94; capped@3000 -> 0.80
The uni+bigram / thresh=0.2 / cap=3000 configuration below was chosen as the best
recall-per-candidate tradeoff found.
"""

import pickle
import re
import time

import pandas as pd
from datasketch import MinHash, MinHashLSH

HERE = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"

# ---------------------------------------------------------------------------
# Tunable parameters (see sweep notes in the module docstring)
# ---------------------------------------------------------------------------
NUM_PERM = 128          # MinHash permutations
LSH_THRESHOLD = 0.2     # Jaccard threshold for the LSH index (recall-oriented, deliberately low)
MAX_CANDIDATES = 3000   # this is an exploration/ensemble-input file, not the final 2000-pair
                        # submission, so we keep a few thousand candidates for headroom

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(title, manufacturer):
    title = "" if pd.isna(title) else str(title)
    manufacturer = "" if pd.isna(manufacturer) else str(manufacturer)
    text = f"{title} {manufacturer}".lower()
    return TOKEN_RE.findall(text)


def combined_shingles(tokens):
    """Union of word-unigrams and word-bigrams -- best recall/size tradeoff found locally."""
    unigrams = set(tokens)
    bigrams = {" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)} if len(tokens) >= 2 else set()
    return unigrams | bigrams


def build_minhash(shingles, num_perm=NUM_PERM):
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(s.encode("utf8"))
    return mh


def main():
    t0 = time.time()

    tableA = pd.read_csv(f"{HERE}/TableA.csv")
    tableB = pd.read_csv(f"{HERE}/TableB.csv")

    tableA_ids = set(tableA["id"])
    tableB_ids = set(tableB["id"])
    overlap = tableA_ids & tableB_ids
    print(f"TableA rows: {len(tableA)}, TableB rows: {len(tableB)}")
    print(f"id namespace overlap between tables: {len(overlap)} "
          f"({'GOTCHA LIVE - never sort tuples' if overlap else 'no overlap'})")
    print(f"TableB manufacturer nulls: {tableB['manufacturer'].isna().sum()}/{len(tableB)}")

    # --- build shingle sets + minhashes for both tables --------------------
    a_shingles, a_minhash = {}, {}
    for row in tableA.itertuples(index=False):
        tokens = normalize_text(row.title, row.manufacturer)
        shingles = combined_shingles(tokens)
        a_shingles[row.id] = shingles
        a_minhash[row.id] = build_minhash(shingles)

    b_shingles, b_minhash = {}, {}
    for row in tableB.itertuples(index=False):
        tokens = normalize_text(row.title, row.manufacturer)
        shingles = combined_shingles(tokens)
        b_shingles[row.id] = shingles
        b_minhash[row.id] = build_minhash(shingles)

    empty_a = sum(1 for s in a_shingles.values() if len(s) == 0)
    empty_b = sum(1 for s in b_shingles.values() if len(s) == 0)
    print(f"Empty shingle sets: TableA={empty_a}, TableB={empty_b}")

    # --- build LSH index over TableB, query with TableA ---------------------
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for b_id, mh in b_minhash.items():
        # datasketch LSH keys must be unique; TableB ids may collide with TableA
        # ids in the shared namespace, so prefix explicitly.
        lsh.insert(f"b_{b_id}", mh)

    candidate_pairs = set()  # (id_a, id_b): id_a from TableA, id_b from TableB -- never sorted
    for a_id, mh in a_minhash.items():
        for key in lsh.query(mh):
            b_id = int(key[2:])  # strip "b_" prefix
            candidate_pairs.add((a_id, b_id))  # explicit provenance, no sorting

    print(f"Raw candidate pairs from LSH buckets: {len(candidate_pairs)}")

    # --- rank by exact Jaccard similarity on shingle sets --------------------
    def jaccard(a_id, b_id):
        sa, sb = a_shingles[a_id], b_shingles[b_id]
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    scored = [(jaccard(a_id, b_id), a_id, b_id) for (a_id, b_id) in candidate_pairs]
    scored.sort(key=lambda x: x[0], reverse=True)

    final_pairs = {(a_id, b_id) for (_, a_id, b_id) in scored[:MAX_CANDIDATES]}

    print(f"Final candidate set size (after ranking/truncation): {len(final_pairs)}")

    # --- proxy recall against 100_matches.pkl (NOT excluded here) -----------
    with open(f"{HERE}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)

    overlap_known = final_pairs & known_matches
    recall = len(overlap_known) / len(known_matches) if known_matches else float("nan")
    raw_recall = len(candidate_pairs & known_matches) / len(known_matches) if known_matches else float("nan")

    runtime = time.time() - t0

    print("=" * 60)
    print("MinHash + LSH blocking strategy report")
    print("=" * 60)
    print(f"Library used: datasketch (MinHash/MinHashLSH), num_perm={NUM_PERM}, "
          f"shingles=unigrams+bigrams, lsh_threshold={LSH_THRESHOLD}")
    print(f"Raw candidate pairs (pre-truncation): {len(candidate_pairs)}")
    print(f"Raw proxy recall (pre-truncation):    {raw_recall:.3f}")
    print(f"Final candidate set size:             {len(final_pairs)}")
    print(f"Final proxy recall vs 100_matches:     {recall:.3f}  "
          f"({len(overlap_known)}/{len(known_matches)})")
    print(f"Runtime: {runtime:.2f}s")
    print("NOTE: known matches are NOT excluded from candidates_lsh.pkl -- exclusion")
    print("      happens later at the ensemble/submission step, per task instructions.")

    with open(f"{HERE}/candidates_lsh.pkl", "wb") as f:
        pickle.dump(final_pairs, f)
    print(f"Saved {len(final_pairs)} candidate pairs to {HERE}/candidates_lsh.pkl")

    # write a short recall report file for the ensemble step
    report = {
        "strategy": "minhash_lsh",
        "num_perm": NUM_PERM,
        "shingles": "unigrams+bigrams",
        "lsh_threshold": LSH_THRESHOLD,
        "raw_candidate_count": len(candidate_pairs),
        "raw_proxy_recall": raw_recall,
        "final_candidate_count": len(final_pairs),
        "final_proxy_recall": recall,
        "runtime_seconds": runtime,
    }
    with open(f"{HERE}/candidates_lsh_report.pkl", "wb") as f:
        pickle.dump(report, f)


if __name__ == "__main__":
    main()
