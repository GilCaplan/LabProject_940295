"""
BM25 + character n-gram TF-IDF blocking strategy for the entity-matching hackathon.

Two structurally distinct signals, combined by union:

1. BM25 ranking (rank_bm25.BM25Okapi) over word-tokenized normalized
   `title + manufacturer` text. TableB is treated as the "corpus" (indexed),
   and each TableA record's tokens are used as a "query" against it. BM25
   differs from cosine TF-IDF via term-frequency saturation (diminishing
   returns for repeated terms) and document-length normalization, which often
   helps for short-text retrieval like product titles.

2. Character n-gram (3-5) TF-IDF cosine similarity over the same normalized
   text via sklearn's TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5)).
   This is robust to typos/abbreviations/spelling variants that word-level
   tokenization misses (e.g. "qckbks" vs "quickbooks").

Scale (1363 x 3226 ~= 4.4M pairs) is small enough for brute force on both.

For each signal we take the top-K BM25/cosine matches per TableA record
(rather than a single global top-K over the full 4.4M matrix) so that every
TableA record gets a chance to contribute candidates, then union the two
signals' pairs together.

Proxy recall against 100_matches.pkl is measured for BM25 alone, char n-gram
alone, and the union -- BEFORE any exclusion of those known matches (per
hackathon rules, exclusion happens once at the final ensemble/submission
step, not here).

Final combined set is capped at 3000 pairs (headroom above the 2000 submission
cap, to leave room for the ensemble merge to pick the best).
"""

import pickle
import re
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"
PER_RECORD_TOPK = 5   # top-k TableB matches per TableA record, for each signal
FINAL_CAP = 3000       # headroom above the eventual 2000-pair submission cap


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


def top_k_pairs_from_score_matrix(sim, a_ids, b_ids, k):
    """Given a dense (len(A), len(B)) score matrix, take top-k B matches per A row."""
    pairs = set()
    # argpartition for speed, then sort within the small k slice
    n_b = sim.shape[1]
    k = min(k, n_b)
    part_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    for r in range(sim.shape[0]):
        cols = part_idx[r]
        id_a = a_ids[r]
        for c in cols:
            id_b = b_ids[c]
            pairs.add((id_a, id_b))
    return pairs


def rank_within_record(scores_1d, k):
    """Return array mapping column index -> rank (0 = best) for the top-k
    entries of a 1-D score array; columns outside the top-k get rank = k
    (i.e. tied for "not in top-k")."""
    n = len(scores_1d)
    k = min(k, n)
    top_idx = np.argpartition(-scores_1d, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores_1d[top_idx])]
    rank_map = {}
    for rank, idx in enumerate(top_idx):
        rank_map[idx] = rank
    return rank_map


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

    a_ids = tableA["id"].to_numpy()
    b_ids = tableB["id"].to_numpy()

    corpus_a = build_corpus(tableA)
    corpus_b = build_corpus(tableB)

    with open(f"{DATA_DIR}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)

    # ------------------------------------------------------------------
    # Signal 1: BM25 -- TableB as corpus, TableA rows as queries.
    # ------------------------------------------------------------------
    t_bm0 = time.time()
    tokenized_b = [doc.split() for doc in corpus_b]
    tokenized_a = [doc.split() for doc in corpus_a]

    bm25 = BM25Okapi(tokenized_b)

    bm25_pairs = set()
    k_bm25 = min(PER_RECORD_TOPK, len(tableB))
    for i, q_tokens in enumerate(tokenized_a):
        if not q_tokens:
            continue
        scores = bm25.get_scores(q_tokens)  # shape (len(B),)
        top_idx = np.argpartition(-scores, kth=k_bm25 - 1)[:k_bm25]
        id_a = a_ids[i]
        for c in top_idx:
            if scores[c] <= 0:
                continue  # skip zero-score (no term overlap) matches
            bm25_pairs.add((id_a, b_ids[c]))

    t_bm1 = time.time()
    print(f"BM25 pass done in {t_bm1 - t_bm0:.2f}s, {len(bm25_pairs)} candidate pairs "
          f"(top-{k_bm25} per TableA record)")

    bm25_recall = len(bm25_pairs & known_matches) / len(known_matches)
    print(f"[BM25 alone] proxy recall: {bm25_recall:.3f} "
          f"({len(bm25_pairs & known_matches)}/{len(known_matches)})")

    # ------------------------------------------------------------------
    # Signal 2: char n-gram (3-5) TF-IDF cosine similarity.
    # ------------------------------------------------------------------
    t_cn0 = time.time()
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    combined_corpus = pd.concat([corpus_a, corpus_b], ignore_index=True)
    char_vectorizer.fit(combined_corpus)

    char_tfidf_a = char_vectorizer.transform(corpus_a)  # (1363, V)
    char_tfidf_b = char_vectorizer.transform(corpus_b)  # (3226, V)
    print(f"Char n-gram TF-IDF vocab size: {len(char_vectorizer.vocabulary_)}")

    char_sim = (char_tfidf_a @ char_tfidf_b.T).toarray()  # dense (1363, 3226)
    t_cn1 = time.time()
    print(f"Char n-gram cosine similarity matrix computed in {t_cn1 - t_cn0:.2f}s, "
          f"shape={char_sim.shape}")

    charngram_pairs = top_k_pairs_from_score_matrix(char_sim, a_ids, b_ids, PER_RECORD_TOPK)
    print(f"Char n-gram candidate pairs: {len(charngram_pairs)} (top-{PER_RECORD_TOPK} per TableA record)")

    charngram_recall = len(charngram_pairs & known_matches) / len(known_matches)
    print(f"[Char n-gram alone] proxy recall: {charngram_recall:.3f} "
          f"({len(charngram_pairs & known_matches)}/{len(known_matches)})")

    # ------------------------------------------------------------------
    # Combine (union) -- structurally different signals, keep both.
    # ------------------------------------------------------------------
    combined_pairs = bm25_pairs | charngram_pairs
    combined_recall = len(combined_pairs & known_matches) / len(known_matches)
    print(f"[Combined union] proxy recall: {combined_recall:.3f} "
          f"({len(combined_pairs & known_matches)}/{len(known_matches)}), "
          f"{len(combined_pairs)} raw candidate pairs")

    # Sanity: confirm every candidate pair actually has id_a in TableA ids, id_b in TableB ids.
    bad = [p for p in combined_pairs if p[0] not in tableA_ids or p[1] not in tableB_ids]
    assert not bad, f"Found {len(bad)} malformed pairs, e.g. {bad[:5]}"

    # If combined exceeds the cap, rank pairs by best (min) per-record rank
    # across the two signals -- NOT raw score blending, since BM25 scores
    # and TF-IDF cosine scores live on incomparable scales and naive
    # score-blended truncation was observed to tank recall (0.95 -> 0.84).
    # Rank-based fusion favors a pair that is a *top* match under either
    # signal, regardless of that signal's absolute score magnitude.
    final_pairs = combined_pairs
    if len(combined_pairs) > FINAL_CAP:
        row_of_a = {aid: i for i, aid in enumerate(a_ids)}

        # Per-TableA-record rank maps for each signal (0 = best match).
        bm25_rank_maps = []
        for i, q_tokens in enumerate(tokenized_a):
            if not q_tokens:
                bm25_rank_maps.append({})
                continue
            scores = bm25.get_scores(q_tokens)
            bm25_rank_maps.append(rank_within_record(scores, PER_RECORD_TOPK))

        charngram_rank_maps = []
        for i in range(char_sim.shape[0]):
            charngram_rank_maps.append(rank_within_record(char_sim[i], PER_RECORD_TOPK))

        col_of_b = {bid: j for j, bid in enumerate(b_ids)}
        big = PER_RECORD_TOPK  # sentinel worst-rank for "not in this signal's top-k"

        def fusion_key(pair):
            id_a, id_b = pair
            r = row_of_a[id_a]
            c = col_of_b[id_b]
            bm_rank = bm25_rank_maps[r].get(c, big)
            cn_rank = charngram_rank_maps[r].get(c, big)
            best_rank = min(bm_rank, cn_rank)
            # tie-break: prefer pairs strong in *both* signals
            sum_rank = bm_rank + cn_rank
            return (best_rank, sum_rank)

        ranked = sorted(combined_pairs, key=fusion_key)
        final_pairs = set(ranked[:FINAL_CAP])
        trunc_recall = len(final_pairs & known_matches) / len(known_matches)
        print(f"Truncated to top-{FINAL_CAP} by rank-fusion, "
              f"proxy recall after truncation: {trunc_recall:.3f}")

    out_path = f"{DATA_DIR}/candidates_bm25_charngram.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(final_pairs, f)

    t1 = time.time()
    print(f"Saved {len(final_pairs)} candidate pairs to {out_path}")
    print(f"Total runtime: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
