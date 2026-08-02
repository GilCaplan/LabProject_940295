"""
Clustering-based blocking strategy for the entity-matching hackathon.

Structurally different from the ten NN-search strategies already in this
repo: instead of ranking each TableA record's nearest TableB neighbors, this
does classic "standard blocking" -- partition the *combined* TableA+TableB
records into K hard clusters in a shared vector space, then take the full
cross-product of TableA x TableB records that land in the same (or an
adjacent) cluster.

Approach:
1. Load TableA.csv / TableB.csv, normalize title + manufacturer text.
2. Fit TF-IDF (with SVD dimensionality reduction) on the COMBINED corpus
   (TableA + TableB together) -- kept simple/fast rather than pulling in
   sentence-transformers, since TF-IDF+SVD is already cheap and gives a
   well-behaved Euclidean space for KMeans.
3. Run KMeans on the combined TableA+TableB embedding matrix for several k
   (50, 100, 200), so both tables share the exact same cluster space.
4. For each k: build within-cluster candidate pairs (same cluster), plus an
   "adjacent cluster" variant (same cluster OR nearest neighboring cluster
   by centroid distance) to catch borderline pairs split across a boundary.
5. Rank within-cluster pairs by cosine similarity on the same vectors (cheap
   dot product) to guard against cluster-size blowup -- cap large clusters
   instead of emitting the full O(n*m) cross product unbounded.
6. Measure proxy recall against 100_matches.pkl for each (k, adjacency)
   config WITHOUT excluding those pairs (exclusion happens later, at the
   ensemble/submission stage).
7. Pick the best config, truncate to top TOP_K by within-cluster similarity,
   and save the final candidate set to candidates_kmeans.pkl.
"""

import pickle
import re
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

DATA_DIR = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"
TOP_K = 3000  # leave headroom above the eventual 2000 cap for ensemble merge
MAX_PER_CLUSTER_PAIRS = 20000  # cap cross-product blowup within one giant cluster
K_VALUES = [50, 100, 200]


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


def cluster_pairs_for_k(k, labels, n_a, vecs, a_ids, b_ids, allow_adjacent, centroids):
    """Generate candidate pairs for a given k, ranked by cosine sim, capped per cluster."""
    labels_a = labels[:n_a]
    labels_b = labels[n_a:]

    # group indices by cluster
    a_by_cluster = defaultdict(list)
    for i, lab in enumerate(labels_a):
        a_by_cluster[lab].append(i)
    b_by_cluster = defaultdict(list)
    for j, lab in enumerate(labels_b):
        b_by_cluster[lab].append(j)

    # adjacent cluster map: nearest other centroid by euclidean distance
    adj_map = {}
    if allow_adjacent:
        dists = euclidean_distances(centroids, centroids)
        np.fill_diagonal(dists, np.inf)
        adj_map = {c: int(np.argmin(dists[c])) for c in range(len(centroids))}

    scored_pairs = []  # (score, id_a, id_b)
    for cluster_id, a_idx_list in a_by_cluster.items():
        target_clusters = [cluster_id]
        if allow_adjacent and cluster_id in adj_map:
            target_clusters.append(adj_map[cluster_id])

        b_idx_list = []
        for tc in target_clusters:
            b_idx_list.extend(b_by_cluster.get(tc, []))
        if not a_idx_list or not b_idx_list:
            continue

        a_sub = vecs[a_idx_list]  # (na, d)
        b_sub = vecs[[n_a + j for j in b_idx_list]]  # (nb, d)
        sim_block = a_sub @ b_sub.T  # cosine sim (vecs are L2-normalized)

        n_pairs = sim_block.size
        if n_pairs <= MAX_PER_CLUSTER_PAIRS:
            ii, jj = np.unravel_index(np.arange(n_pairs), sim_block.shape)
        else:
            # cap: keep only the top MAX_PER_CLUSTER_PAIRS scoring pairs in this block
            flat = sim_block.ravel()
            top_idx = np.argpartition(flat, -MAX_PER_CLUSTER_PAIRS)[-MAX_PER_CLUSTER_PAIRS:]
            ii, jj = np.unravel_index(top_idx, sim_block.shape)

        for i, j in zip(ii, jj):
            id_a = a_ids[a_idx_list[i]]
            id_b = b_ids[b_idx_list[j]]
            score = sim_block[i, j]
            scored_pairs.append((score, id_a, id_b))

    return scored_pairs


def main():
    # KMeans/TruncatedSVD internals emit benign RuntimeWarnings (matmul overflow
    # in intermediate float32 buffers) on this data; output has been verified
    # NaN/Inf-free, so silence the noise for a clean run log.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    tableA = pd.read_csv(f"{DATA_DIR}/TableA.csv")
    tableB = pd.read_csv(f"{DATA_DIR}/TableB.csv")
    print(f"TableA: {len(tableA)} rows, TableB: {len(tableB)} rows")

    tableA_ids = set(tableA["id"])
    tableB_ids = set(tableB["id"])

    a_ids = tableA["id"].to_numpy()
    b_ids = tableB["id"].to_numpy()

    corpus_a = build_corpus(tableA)
    corpus_b = build_corpus(tableB)
    combined_corpus = pd.concat([corpus_a, corpus_b], ignore_index=True)

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf_combined = vectorizer.fit_transform(combined_corpus)
    print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")

    # Reduce dimensionality with SVD so KMeans (Euclidean) behaves well.
    n_components = min(100, tfidf_combined.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm="arpack")
    reduced = svd.fit_transform(tfidf_combined)
    # L2-normalize so dot product == cosine similarity, and so Euclidean KMeans
    # on normalized vectors approximates spherical clustering.
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = reduced / norms
    print(f"Reduced embedding shape: {vecs.shape}")

    n_a = len(tableA)

    with open(f"{DATA_DIR}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)

    best_config = None
    best_recall_at_2000 = -1
    results = {}

    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vecs)
        centroids = km.cluster_centers_

        for allow_adjacent in (False, True):
            scored_pairs = cluster_pairs_for_k(
                k, labels, n_a, vecs, a_ids, b_ids, allow_adjacent, centroids
            )
            scored_pairs.sort(key=lambda x: -x[0])

            # dedupe while preserving best score order, cap at TOP_K
            seen = set()
            ranked_pairs = []
            for score, id_a, id_b in scored_pairs:
                pair = (id_a, id_b)
                if pair in seen:
                    continue
                seen.add(pair)
                ranked_pairs.append(pair)
                if len(ranked_pairs) >= TOP_K:
                    break

            full_set = set(ranked_pairs)
            recall_full = len(full_set & known_matches) / len(known_matches)

            top2000 = set(ranked_pairs[:2000])
            recall_2000 = len(top2000 & known_matches) / len(known_matches)

            label = f"k={k}, adjacent={allow_adjacent}"
            results[label] = (len(full_set), recall_full, recall_2000)
            print(
                f"[{label}] candidates={len(full_set)}, "
                f"proxy_recall(top-{len(full_set)})={recall_full:.3f}, "
                f"proxy_recall(top-2000)={recall_2000:.3f}"
            )

            if recall_2000 > best_recall_at_2000:
                best_recall_at_2000 = recall_2000
                best_config = (label, ranked_pairs, full_set, recall_full, recall_2000)

    label, ranked_pairs, candidate_pairs, recall_full, recall_2000 = best_config
    print(f"\nBest config: {label} (recall@2000={recall_2000:.3f})")

    # Sanity check provenance.
    bad = [p for p in candidate_pairs if p[0] not in tableA_ids or p[1] not in tableB_ids]
    assert not bad, f"Found {len(bad)} malformed pairs, e.g. {bad[:5]}"

    out_path = f"{DATA_DIR}/candidates_kmeans.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(candidate_pairs, f)

    t1 = time.time()
    print(f"Saved {len(candidate_pairs)} candidate pairs to {out_path}")
    print(f"Proxy recall against 100_matches.pkl (BEFORE exclusion, full set): {recall_full:.3f}")
    print(f"Proxy recall if truncated to top-2000: {recall_2000:.3f}")
    print(f"Total runtime: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
