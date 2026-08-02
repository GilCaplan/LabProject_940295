"""
Pure rapidfuzz string-similarity blocking strategy for the entity-matching hackathon.

This is deliberately a DIFFERENT kind of signal from the other strategies already
in this repo:
  - candidates_tfidf.pkl        -> sparse vector cosine similarity
  - candidates_lsh.pkl          -> MinHash/LSH over shingled tokens (approximate)
  - candidates_embeddings.pkl   -> dense sentence-embedding cosine similarity
  - candidates_sorted_neighborhood.pkl -> proximity in sorted order
  - candidates_deepblocker.pkl  -> autoencoder tuple-embedding nearest neighbors
  - candidates_fuzzy.pkl (this file) -> pure edit-distance-family string
    similarity (rapidfuzz token_set_ratio), scored directly and exhaustively
    over every TableA x TableB pair via rapidfuzz.process.cdist (vectorized
    C++ implementation), NOT a vector-space or hashing method.

Approach:
1. Load TableA.csv / TableB.csv.
2. Normalize text: concatenate title + manufacturer (when present), lowercase,
   strip punctuation, collapse whitespace.
3. Brute-force score ALL 1363 x 3226 ~= 4.4M pairs using
   rapidfuzz.process.cdist with fuzz.token_set_ratio (vectorized, C++ backed --
   feasible at this scale, no blocking key needed).
4. Also compute token_sort_ratio as an alternate scorer, empirically compare
   proxy recall against 100_matches.pkl, and keep whichever scorer (or a
   combination) yields higher recall for the final saved candidate set.
5. Rank all pairs by the chosen score, take the top-K (with headroom under the
   2000 hard submission cap for the later ensemble step), keeping id_a from
   TableA and id_b from TableB explicitly (never sorted -- the two tables'
   ids share an overlapping integer namespace).
6. Measure recall against 100_matches.pkl BEFORE excluding those pairs from
   the saved output (used only as a proxy metric / threshold-tuning signal;
   exclusion of the 100 known matches happens later at the ensemble/submission
   stage, not here).
7. Save the final candidate set (as a plain pickled Python set) to
   candidates_fuzzy.pkl.
"""

import pickle
import re
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

DATA_DIR = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"
TOP_K = 3000  # leave headroom above the eventual 2000 cap for ensemble merge


def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_corpus(df):
    title = df["title"].apply(normalize_text)
    manufacturer = df["manufacturer"].apply(normalize_text)
    combined = (title + " " + manufacturer).str.strip()
    return combined.tolist()


def top_k_pairs_from_matrix(score_matrix, a_ids, b_ids, k):
    flat_idx = np.argsort(score_matrix.ravel())[::-1][:k]
    row_idx, col_idx = np.unravel_index(flat_idx, score_matrix.shape)
    pairs = set()
    for r, c in zip(row_idx, col_idx):
        pairs.add((a_ids[r], b_ids[c]))  # (TableA id, TableB id) explicitly, never sorted
    return pairs, flat_idx


def main():
    t0 = time.time()

    tableA = pd.read_csv(f"{DATA_DIR}/tableA.csv")
    tableB = pd.read_csv(f"{DATA_DIR}/tableB.csv")

    print(f"TableA: {len(tableA)} rows, TableB: {len(tableB)} rows")
    print(f"Possible pairs (brute force): {len(tableA) * len(tableB):,}")

    tableA_ids = set(tableA["id"])
    tableB_ids = set(tableB["id"])
    overlap = tableA_ids & tableB_ids
    print(f"ID namespace overlap between tables: {len(overlap)} shared id values "
          f"(sorting pairs would be UNSAFE; using explicit provenance instead)")

    corpus_a = build_corpus(tableA)
    corpus_b = build_corpus(tableB)

    a_ids = tableA["id"].to_numpy()
    b_ids = tableB["id"].to_numpy()

    with open(f"{DATA_DIR}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)

    # --- Score with token_set_ratio (robust to word order / subset titles) ---
    t_s0 = time.time()
    sim_tsr = process.cdist(corpus_a, corpus_b, scorer=fuzz.token_set_ratio, workers=-1)
    t_s1 = time.time()
    print(f"token_set_ratio cdist computed in {t_s1 - t_s0:.2f}s, shape={sim_tsr.shape}")

    pairs_tsr, flat_idx_tsr = top_k_pairs_from_matrix(sim_tsr, a_ids, b_ids, TOP_K)
    recall_tsr = len(pairs_tsr & known_matches) / len(known_matches)
    print(f"[token_set_ratio] top-{TOP_K} proxy recall: {recall_tsr:.3f} "
          f"({len(pairs_tsr & known_matches)}/{len(known_matches)})")

    # --- Alternate scorer: token_sort_ratio ---
    t_s2 = time.time()
    sim_tsort = process.cdist(corpus_a, corpus_b, scorer=fuzz.token_sort_ratio, workers=-1)
    t_s3 = time.time()
    print(f"token_sort_ratio cdist computed in {t_s3 - t_s2:.2f}s, shape={sim_tsort.shape}")

    pairs_tsort, flat_idx_tsort = top_k_pairs_from_matrix(sim_tsort, a_ids, b_ids, TOP_K)
    recall_tsort = len(pairs_tsort & known_matches) / len(known_matches)
    print(f"[token_sort_ratio] top-{TOP_K} proxy recall: {recall_tsort:.3f} "
          f"({len(pairs_tsort & known_matches)}/{len(known_matches)})")

    # --- Alternate scorer: partial_ratio ---
    t_s4 = time.time()
    sim_partial = process.cdist(corpus_a, corpus_b, scorer=fuzz.partial_ratio, workers=-1)
    t_s5 = time.time()
    print(f"partial_ratio cdist computed in {t_s5 - t_s4:.2f}s, shape={sim_partial.shape}")

    pairs_partial, flat_idx_partial = top_k_pairs_from_matrix(sim_partial, a_ids, b_ids, TOP_K)
    recall_partial = len(pairs_partial & known_matches) / len(known_matches)
    print(f"[partial_ratio] top-{TOP_K} proxy recall: {recall_partial:.3f} "
          f"({len(pairs_partial & known_matches)}/{len(known_matches)})")

    # --- Combined: average of token_set_ratio + token_sort_ratio (cheap ensemble
    # of two edit-distance-family scorers, still "pure rapidfuzz string similarity") ---
    sim_combo = (sim_tsr + sim_tsort) / 2.0
    pairs_combo, flat_idx_combo = top_k_pairs_from_matrix(sim_combo, a_ids, b_ids, TOP_K)
    recall_combo = len(pairs_combo & known_matches) / len(known_matches)
    print(f"[avg(token_set_ratio, token_sort_ratio)] top-{TOP_K} proxy recall: "
          f"{recall_combo:.3f} ({len(pairs_combo & known_matches)}/{len(known_matches)})")

    # --- Pick whichever scorer wins empirically ---
    candidates = {
        "token_set_ratio": (recall_tsr, pairs_tsr),
        "token_sort_ratio": (recall_tsort, pairs_tsort),
        "partial_ratio": (recall_partial, pairs_partial),
        "avg_tsr_tsort": (recall_combo, pairs_combo),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_recall, best_pairs = candidates[best_name]
    print(f"\nBest scorer: {best_name} with proxy recall {best_recall:.3f}")

    candidate_pairs = best_pairs
    print(f"Candidate pairs after dedup: {len(candidate_pairs)} (requested top-{TOP_K})")

    # Sanity: confirm every candidate pair actually has id_a in TableA ids, id_b in TableB ids.
    bad = [p for p in candidate_pairs if p[0] not in tableA_ids or p[1] not in tableB_ids]
    assert not bad, f"Found {len(bad)} malformed pairs, e.g. {bad[:5]}"

    # NOTE: per instructions for this exploration step, we do NOT exclude the
    # 100 known matches here -- that happens once, later, at the
    # ensemble/submission stage. We save the raw ranked candidate set as-is.
    out_path = f"{DATA_DIR}/candidates_fuzzy.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(candidate_pairs, f)

    t1 = time.time()
    print(f"Saved {len(candidate_pairs)} candidate pairs to {out_path}")
    print(f"Total runtime: {t1 - t0:.2f}s")

    # Also report what recall would look like if truncated to exactly 2000
    # (the eventual hard submission cap), for reference, using the winning scorer.
    winning_matrix = {
        "token_set_ratio": sim_tsr,
        "token_sort_ratio": sim_tsort,
        "partial_ratio": sim_partial,
        "avg_tsr_tsort": sim_combo,
    }[best_name]
    top2000_pairs, _ = top_k_pairs_from_matrix(winning_matrix, a_ids, b_ids, 2000)
    recall_2000 = len(top2000_pairs & known_matches) / len(known_matches)
    print(f"[Reference] proxy recall if truncated to top-2000 ({best_name}): {recall_2000:.3f}")


if __name__ == "__main__":
    main()
