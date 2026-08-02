"""
Cross-encoder reranking strategy for entity-matching blocking.

Structurally different from candidates_embeddings.pkl (a bi-encoder that embeds
each record independently and compares via cosine similarity): here we use a
sentence-transformers CrossEncoder, which jointly encodes BOTH texts of a pair
in a single forward pass and outputs one relevance score. Cross-encoders are
typically much stronger than bi-encoder cosine similarity, but too slow to run
over all ~4.4M possible TableA x TableB pairs -- so we only rerank the union of
the 8 existing candidates_*.pkl pools (~16k pairs), which is small enough to
score exhaustively on CPU.

Steps:
  1. Load TableA.csv / TableB.csv, build a normalized "title manufacturer" text
     per record (id -> text).
  2. Load all 8 existing candidates_*.pkl files and take their union as the
     pool to rerank. NEVER sort/swap (id_a, id_b) -- id_a always comes from the
     TableA-origin pkl and id_b always from the TableB-origin pkl of each pair
     already stored in the pool.
  3. Score every pooled pair with CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
     in batches (batch size 128), building (textA, textB) input pairs.
  4. Rank all pooled pairs by score, descending.
  5. Load 100_matches.pkl (proxy known matches, NOT excluded here -- exclusion
     happens later at the ensemble/submission step) and report recall@k for a
     few cutoffs.
  6. Take the top-K (K=3000, leaving room under the 2000-pair final submission
     cap for the later ensemble merge) and save as candidates_crossencoder.pkl.
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
]

MODEL_NAME_PRIMARY = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_NAME_FALLBACK = "cross-encoder/stsb-roberta-base"

TOP_K = 3000
BATCH_SIZE = 128


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
    for fname in POOL_FILES:
        path = f"{DATA_DIR}/{fname}"
        with open(path, "rb") as f:
            pairs = pickle.load(f)
        for pair in pairs:
            id_a, id_b = int(pair[0]), int(pair[1])
            pool.add((id_a, id_b))
    return pool


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

    pool = load_pool()
    pool_list = list(pool)
    print(f"Pooled union of {len(POOL_FILES)} strategy files: {len(pool_list)} unique pairs")

    model, model_name = load_cross_encoder()
    if model is None:
        print("No cross-encoder could be loaded (no cache hit + no/limited internet). "
              "Skipping gracefully -- writing an empty candidate set.")
        with open(f"{DATA_DIR}/candidates_crossencoder.pkl", "wb") as f:
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

    with open(f"{DATA_DIR}/candidates_crossencoder.pkl", "wb") as f:
        pickle.dump(final_set, f)

    t_end = time.time()
    print(f"Saved {len(final_set)} pairs to candidates_crossencoder.pkl")
    print(f"Total runtime: {t_end - t_start:.1f}s")


if __name__ == "__main__":
    main()
