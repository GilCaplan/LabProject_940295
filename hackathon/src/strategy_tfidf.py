"""
TF-IDF + cosine similarity blocking strategy for the entity-matching hackathon.

Approach:
1. Load TableA.csv / TableB.csv.
2. Normalize text: concatenate title + manufacturer (when present), lowercase,
   strip punctuation.
3. Fit a single TF-IDF vectorizer on the combined corpus (TableA texts + TableB
   texts) so both tables share the same vocabulary/IDF weighting.
4. Brute-force cosine similarity between all TableA rows (1363) and all TableB
   rows (3226) -- 1363 x 3226 ~= 4.4M pairs, which is cheap as a single dense
   matmul (well under a second), so no extra blocking pre-filter is needed.
5. Rank all pairs by similarity score, take the top-K by score (K chosen with
   headroom under the 2000 cap for the ensemble step), keeping id_a from
   TableA and id_b from TableB explicitly (never sorted - the two tables'
   ids share an overlapping integer namespace).
6. Measure recall against 100_matches.pkl BEFORE excluding those pairs from
   the saved output (used only as a proxy metric / threshold-tuning signal).
7. Save the final candidate set (as a plain pickled Python set) to
   candidates_tfidf.pkl. This script does NOT exclude the 100 known matches
   from its own output (that exclusion happens once, later, at the
   ensemble/submission stage) and does NOT call save_submission.
"""

import pickle
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
    return combined


def main():
    t0 = time.time()

    tableA = pd.read_csv(f"{DATA_DIR}/TableA.csv")
    tableB = pd.read_csv(f"{DATA_DIR}/TableB.csv")

    print(f"TableA: {len(tableA)} rows, TableB: {len(tableB)} rows")
    print(f"Possible pairs (brute force): {len(tableA) * len(tableB):,}")

    tableA_ids = set(tableA["id"])
    tableB_ids = set(tableB["id"])
    overlap = tableA_ids & tableB_ids
    print(f"ID namespace overlap between tables: {len(overlap)} shared id values "
          f"(sorting pairs would be UNSAFE; using explicit provenance instead)")

    corpus_a = build_corpus(tableA)
    corpus_b = build_corpus(tableB)

    # Fit TF-IDF on the combined corpus so both tables share vocabulary + IDF.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    combined_corpus = pd.concat([corpus_a, corpus_b], ignore_index=True)
    vectorizer.fit(combined_corpus)

    tfidf_a = vectorizer.transform(corpus_a)  # (1363, V)
    tfidf_b = vectorizer.transform(corpus_b)  # (3226, V)

    print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")

    # Since both matrices are L2-normalized by TfidfVectorizer by default,
    # cosine similarity == dot product.
    t_sim0 = time.time()
    sim = (tfidf_a @ tfidf_b.T).toarray()  # dense (1363, 3226) ~ 4.4M floats, ~35MB
    t_sim1 = time.time()
    print(f"Cosine similarity matrix computed in {t_sim1 - t_sim0:.2f}s, "
          f"shape={sim.shape}")

    # Rank all pairs by score and take top-K.
    flat_idx = np.argsort(sim.ravel())[::-1][:TOP_K]
    row_idx, col_idx = np.unravel_index(flat_idx, sim.shape)

    a_ids = tableA["id"].to_numpy()
    b_ids = tableB["id"].to_numpy()

    candidate_pairs = set()
    for r, c in zip(row_idx, col_idx):
        id_a = a_ids[r]
        id_b = b_ids[c]
        candidate_pairs.add((id_a, id_b))  # (TableA id, TableB id) explicitly, never sorted

    print(f"Candidate pairs after dedup: {len(candidate_pairs)} (requested top-{TOP_K})")

    # --- Recall proxy against the 100 known matches (measured BEFORE exclusion) ---
    with open(f"{DATA_DIR}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)

    overlap_hits = candidate_pairs & known_matches
    recall = len(overlap_hits) / len(known_matches)
    print(f"Proxy recall against 100_matches.pkl (BEFORE exclusion): "
          f"{recall:.3f} ({len(overlap_hits)}/{len(known_matches)})")

    # Sanity: confirm every candidate pair actually has ida in TableA ids, idb in TableB ids.
    bad = [p for p in candidate_pairs if p[0] not in tableA_ids or p[1] not in tableB_ids]
    assert not bad, f"Found {len(bad)} malformed pairs, e.g. {bad[:5]}"

    # NOTE: per instructions for this exploration step, we do NOT exclude the
    # 100 known matches here -- that happens once, later, at the
    # ensemble/submission stage. We save the raw ranked candidate set as-is.
    out_path = f"{DATA_DIR}/candidates_tfidf.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(candidate_pairs, f)

    t1 = time.time()
    print(f"Saved {len(candidate_pairs)} candidate pairs to {out_path}")
    print(f"Total runtime: {t1 - t0:.2f}s")

    # Also report what recall would look like if we truncated to exactly 2000
    # (the eventual hard submission cap), for reference.
    top2000_idx = flat_idx[:2000]
    r2000, c2000 = np.unravel_index(top2000_idx, sim.shape)
    top2000_pairs = set((a_ids[r], b_ids[c]) for r, c in zip(r2000, c2000))
    recall_2000 = len(top2000_pairs & known_matches) / len(known_matches)
    print(f"[Reference] proxy recall if truncated to top-2000: {recall_2000:.3f}")


if __name__ == "__main__":
    main()
