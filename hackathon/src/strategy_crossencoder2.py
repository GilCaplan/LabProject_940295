"""
Second cross-encoder reranking strategy for entity-matching blocking.

candidates_crossencoder.pkl already reranked the strategy pool with
cross-encoder/ms-marco-MiniLM-L-6-v2 (proxy recall 0.81 at top-3000). This
strategy adds a DIFFERENT, larger/stronger cross-encoder for diversity:

  - Primary:  cross-encoder/ms-marco-MiniLM-L-12-v2
              (bigger model in the same MS-MARCO retrieval-ranking family --
              deeper transformer, same training objective as L-6, so we'd
              expect a similar ranking with somewhat better precision at the
              margins from the extra capacity).
  - Fallback: cross-encoder/stsb-roberta-large
              (different training objective entirely -- STS regression
              instead of MS-MARCO retrieval ranking -- and a much larger
              backbone; used only if the primary model fails to load).

Both were absent from the local HF cache at the start of this run; both were
successfully downloaded over the network (see load_cross_encoder(), which
still degrades gracefully to an empty set if neither is available).

Steps:
  1. Load TableA.csv / TableB.csv, build a normalized "title manufacturer"
     text per record (id -> text).
  2. Pool the union of ALL existing candidates_*.pkl files in this directory
     (rules, tfidf, lsh, embeddings, sorted_neighborhood, deepblocker, fuzzy,
     phonetic, simhash, kmeans, bm25_charngram, word2vec, and the first
     crossencoder's own output) -- NEVER sort/swap (id_a, id_b); id_a always
     comes from the TableA-origin slot and id_b from the TableB-origin slot
     already stored in each source pool.
  3. Score every pooled pair with the chosen CrossEncoder in batches of 64
     (larger model -> smaller batch than the L-6 pass).
  4. Rank all pooled pairs by score, descending.
  5. Load 100_matches.pkl (proxy known matches, NOT excluded here -- exclusion
     happens later at the ensemble/submission step) and report recall@k for a
     few cutoffs.
  6. Take the top-K (K=3000) and save as candidates_crossencoder2.pkl.
"""
import pickle
import re
import time

import pandas as pd
from sentence_transformers import CrossEncoder

DATA_DIR = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"

POOL_FILES = [
    "candidates_rules.pkl",
    "candidates_tfidf.pkl",
    "candidates_lsh.pkl",
    "candidates_embeddings.pkl",
    "candidates_sorted_neighborhood.pkl",
    "candidates_deepblocker.pkl",
    "candidates_fuzzy.pkl",
    "candidates_phonetic.pkl",
    "candidates_simhash.pkl",
    "candidates_kmeans.pkl",
    "candidates_bm25_charngram.pkl",
    "candidates_word2vec.pkl",
    "candidates_crossencoder.pkl",
]

MODEL_NAME_PRIMARY = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MODEL_NAME_FALLBACK = "cross-encoder/stsb-roberta-large"

TOP_K = 3000
BATCH_SIZE = 64


def normalize(s):
    if s is None or (isinstance(s, float)):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text_map(df):
    texts = {}
    for row in df.itertuples(index=False):
        title = normalize(getattr(row, "title", ""))
        manuf = normalize(getattr(row, "manufacturer", ""))
        texts[int(row.id)] = f"{title} {manuf}".strip()
    return texts


def load_pool():
    pool = set()
    used_files = []
    for fname in POOL_FILES:
        path = f"{DATA_DIR}/{fname}"
        try:
            with open(path, "rb") as f:
                pairs = pickle.load(f)
        except FileNotFoundError:
            print(f"  (skipping missing {fname})")
            continue
        for pair in pairs:
            id_a, id_b = int(pair[0]), int(pair[1])
            pool.add((id_a, id_b))
        used_files.append(fname)
    return pool, used_files


def load_cross_encoder():
    try:
        model = CrossEncoder(MODEL_NAME_PRIMARY)
        return model, MODEL_NAME_PRIMARY
    except Exception as e:
        print(f"Primary model {MODEL_NAME_PRIMARY} failed to load: {e}")
        try:
            model = CrossEncoder(MODEL_NAME_FALLBACK)
            return model, MODEL_NAME_FALLBACK
        except Exception as e2:
            print(f"Fallback model {MODEL_NAME_FALLBACK} also failed to load: {e2}")
            return None, None


def main():
    t_start = time.time()

    a_df = pd.read_csv(f"{DATA_DIR}/tableA.csv")
    b_df = pd.read_csv(f"{DATA_DIR}/tableB.csv")
    text_a = build_text_map(a_df)
    text_b = build_text_map(b_df)

    pool, used_files = load_pool()
    pool_list = list(pool)
    print(f"Pooled union of {len(used_files)} strategy files: {len(pool_list)} unique pairs")

    model, model_name = load_cross_encoder()
    if model is None:
        print("No cross-encoder could be loaded (no cache hit + no/limited internet). "
              "Skipping gracefully -- writing an empty candidate set.")
        with open(f"{DATA_DIR}/candidates_crossencoder2.pkl", "wb") as f:
            pickle.dump(set(), f)
        return

    print(f"Loaded cross-encoder: {model_name}")

    sentence_pairs = []
    valid_pairs = []
    for id_a, id_b in pool_list:
        ta = text_a.get(id_a, "")
        tb = text_b.get(id_b, "")
        sentence_pairs.append((ta, tb))
        valid_pairs.append((id_a, id_b))

    t_score_start = time.time()
    scores = model.predict(sentence_pairs, batch_size=BATCH_SIZE, show_progress_bar=True)
    t_score_end = time.time()
    print(f"Scored {len(sentence_pairs)} pairs in {t_score_end - t_score_start:.1f}s")

    scored_pairs = list(zip(valid_pairs, scores))
    scored_pairs.sort(key=lambda x: x[1], reverse=True)

    # Proxy recall check against 100_matches.pkl (NOT excluded from candidate set here)
    with open(f"{DATA_DIR}/100_matches.pkl", "rb") as f:
        known_matches = pickle.load(f)
    known_matches = {(int(a), int(b)) for a, b in known_matches}

    ranked_pairs_only = [p for p, s in scored_pairs]
    for k in (500, 1000, 1500, 2000, 3000, len(ranked_pairs_only)):
        k_eff = min(k, len(ranked_pairs_only))
        top_k_set = set(ranked_pairs_only[:k_eff])
        hits = len(top_k_set & known_matches)
        recall = hits / len(known_matches) if known_matches else float("nan")
        print(f"top-{k_eff}: recall={recall:.3f} ({hits}/{len(known_matches)})")

    pool_recall = len(pool & known_matches) / len(known_matches) if known_matches else float("nan")
    print(f"(reference) recall of the raw unranked pool of {len(pool)}: {pool_recall:.3f}")

    final_k = min(TOP_K, len(ranked_pairs_only))
    final_set = set(ranked_pairs_only[:final_k])

    with open(f"{DATA_DIR}/candidates_crossencoder2.pkl", "wb") as f:
        pickle.dump(final_set, f)

    t_end = time.time()
    print(f"Saved {len(final_set)} pairs to candidates_crossencoder2.pkl")
    print(f"Model used: {model_name}")
    print(f"Total runtime: {t_end - t_start:.1f}s")


if __name__ == "__main__":
    main()
