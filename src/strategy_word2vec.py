"""
Static, non-contextual word-embedding blocking strategy for the entity-matching
hackathon (word2vec/GloVe averaging), distinct from strategy_embeddings.py which
uses a contextual sentence-transformer bi-encoder.

Approach:
1. Load TableA.csv / TableB.csv, normalize title + manufacturer text (lowercase,
   strip punctuation).
2. Load pretrained GloVe vectors via gensim.downloader ("glove-wiki-gigaword-100",
   400K vocab, 100-dim, trained on Wikipedia+Gigaword) -- these are static,
   non-contextual word vectors (one fixed vector per word type, no attention/
   context mixing), fundamentally different signal from the MiniLM sentence
   encoder used in strategy_embeddings.py.
3. Represent each record as the mean of its in-vocabulary word vectors
   (average pooling), skipping OOV tokens. Records with zero in-vocab tokens
   fall back to a zero vector (handled gracefully -- they simply won't match
   anything via this signal, other strategies cover them).
4. L2-normalize record vectors and brute-force cosine similarity between all
   TableA (1363) and TableB (3226) records -- 4.4M pairs, cheap as one matmul.
5. Take top-K nearest TableB neighbors per TableA record (K chosen so total
   candidate count stays near the 3000 cap with headroom for the ensemble).
6. Measure recall against 100_matches.pkl as a proxy metric (NOT excluded from
   the saved output -- exclusion happens later at the ensemble/submission step).
7. Save final candidate set (plain pickled Python set of (id_a, id_b), id_a
   from TableA and id_b from TableB explicitly -- never sorted/swapped, since
   TableA/TableB ids share an overlapping integer namespace) to
   candidates_word2vec.pkl.
"""

import pickle
import re
import time

import numpy as np
import pandas as pd

DATA_DIR = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"
TOP_K_PER_A = 3  # neighbors per TableA record; 1363*3 ~= 4089 raw, dedupe/cap below
FINAL_CAP = 3000  # headroom above eventual 2000-pair submission cap


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


def load_word_vectors():
    """Load pretrained static GloVe vectors; fall back to training a small
    Word2Vec model from scratch on the combined corpus if download fails."""
    try:
        import gensim.downloader as api

        t0 = time.time()
        kv = api.load("glove-wiki-gigaword-100")
        print(f"Loaded pretrained glove-wiki-gigaword-100 in {time.time() - t0:.1f}s "
              f"({len(kv)} words, dim={kv.vector_size})")
        return kv, "pretrained-glove-wiki-gigaword-100"
    except Exception as e:
        print(f"Pretrained download failed ({e}); training Word2Vec from scratch.")
        from gensim.models import Word2Vec

        return None, "fromscratch-word2vec"


def mean_pool(tokens, kv):
    vecs = [kv[t] for t in tokens if t in kv]
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def main():
    t_start = time.time()
    a = pd.read_csv(f"{DATA_DIR}/tableA.csv")
    b = pd.read_csv(f"{DATA_DIR}/tableB.csv")

    a_text = build_corpus(a)
    b_text = build_corpus(b)
    a_tokens = a_text.str.split()
    b_tokens = b_text.str.split()

    kv, method = load_word_vectors()

    if kv is None:
        # Train a lightweight Word2Vec model from scratch on the combined corpus.
        from gensim.models import Word2Vec

        sentences = pd.concat([a_tokens, b_tokens]).tolist()
        t0 = time.time()
        w2v = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=20,
        )
        print(f"Trained from-scratch Word2Vec in {time.time() - t0:.1f}s "
              f"(vocab={len(w2v.wv)})")
        kv = w2v.wv

    dim = kv.vector_size

    a_vecs = np.vstack([mean_pool(toks, kv) for toks in a_tokens]).astype(np.float32)
    b_vecs = np.vstack([mean_pool(toks, kv) for toks in b_tokens]).astype(np.float32)

    # L2-normalize for cosine similarity via dot product.
    def l2norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    a_vecs = l2norm(a_vecs)
    b_vecs = l2norm(b_vecs)

    t0 = time.time()
    sim = a_vecs @ b_vecs.T  # (1363, 3226) brute-force cosine similarity
    print(f"Computed similarity matrix in {time.time() - t0:.2f}s, shape={sim.shape}")

    a_ids = a["id"].to_numpy()
    b_ids = b["id"].to_numpy()

    candidates = set()
    top_k = TOP_K_PER_A
    # Take top_k neighbors per TableA row; if under cap, widen k.
    while True:
        candidates = set()
        idx_topk = np.argpartition(-sim, kth=min(top_k, sim.shape[1] - 1), axis=1)[:, :top_k]
        for i in range(sim.shape[0]):
            id_a = int(a_ids[i])
            for j in idx_topk[i]:
                id_b = int(b_ids[j])
                candidates.add((id_a, id_b))
        if len(candidates) >= FINAL_CAP or top_k >= 10:
            break
        top_k += 1

    # If over cap, rank all (i, j) candidate pairs by score and truncate.
    if len(candidates) > FINAL_CAP:
        id_a_to_i = {int(a_ids[i]): i for i in range(len(a_ids))}
        id_b_to_j = {int(b_ids[j]): j for j in range(len(b_ids))}
        scored = [
            (sim[id_a_to_i[ia], id_b_to_j[ib]], ia, ib) for (ia, ib) in candidates
        ]
        scored.sort(key=lambda x: -x[0])
        candidates = set((ia, ib) for _, ia, ib in scored[:FINAL_CAP])

    print(f"Final candidate count: {len(candidates)} (top_k_per_a={top_k})")

    # Proxy recall against the 100 known matches (NOT excluded from output).
    with open(f"{DATA_DIR}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)
    known_matches = set(known_matches)
    hits = known_matches & candidates
    recall = len(hits) / len(known_matches) if known_matches else float("nan")
    print(f"Proxy recall vs 100_matches.pkl: {recall:.3f} ({len(hits)}/{len(known_matches)})")

    with open(f"{DATA_DIR}/candidates_word2vec.pkl", "wb") as f:
        pickle.dump(candidates, f)

    print(f"Method: {method}")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
