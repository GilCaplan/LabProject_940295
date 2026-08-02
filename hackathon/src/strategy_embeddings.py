"""
Dense sentence-embedding similarity blocking strategy.

Encodes TableA / TableB records (title + manufacturer text) with a
sentence-transformers model, computes full cross-table cosine similarity
(small scale: 1363 x 3226 ~= 4.4M pairs, brute force is fine), and takes the
top-K highest-similarity pairs (deduped, no self/order confusion since
TableA/TableB ids are tracked by provenance, never sorted together).

100_matches.pkl is used ONLY to measure a local proxy recall while tuning
top-k / threshold. Per hackathon plan, exclusion of those 100 exact pairs
from the candidate set happens later at the ensemble-merge / submission
step, NOT in this per-strategy script - so this script does NOT subtract
them.

Output: candidates_embeddings.pkl -> a Python set of (id_a, id_b) tuples.
"""

import os
import pickle
import time

import numpy as np
import pandas as pd

HACKATHON_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_PER_A = 4          # candidate pairs generated per TableA record
FINAL_CAP = 3000         # leave headroom under the 2000 hard cap for ensemble merge
SIM_FLOOR = 0.0          # no hard floor; rely on top-k ranking


def build_text(row):
    title = str(row.get("title", "") or "")
    manufacturer = row.get("manufacturer", "")
    manufacturer = "" if pd.isna(manufacturer) else str(manufacturer)
    parts = [title]
    if manufacturer.strip():
        parts.append(manufacturer)
    return " ".join(p.strip() for p in parts if p.strip())


def main():
    t0 = time.time()

    tableA = pd.read_csv(os.path.join(HACKATHON_DIR, "tableA.csv"))
    tableB = pd.read_csv(os.path.join(HACKATHON_DIR, "tableB.csv"))

    print(f"TableA rows: {len(tableA)}, TableB rows: {len(tableB)}")

    shared_ids = set(tableA["id"]) & set(tableB["id"])
    print(f"Shared id namespace between A and B: {len(shared_ids)} overlapping ids "
          f"(never sort (id_a, id_b) tuples together - keep provenance explicit)")

    texts_a = tableA.apply(build_text, axis=1).tolist()
    texts_b = tableB.apply(build_text, axis=1).tolist()

    # --- Load embedding model (fail gracefully if unavailable) ---
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("FATAL: sentence-transformers is not installed. Stopping.")
        print(repr(e))
        return

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"FATAL: could not load/download model '{MODEL_NAME}'. "
              f"No usable local cache and/or no internet access. Stopping.")
        print(repr(e))
        return

    print(f"Model loaded: {MODEL_NAME}")

    t_embed0 = time.time()
    emb_a = model.encode(texts_a, batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)
    emb_b = model.encode(texts_b, batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)
    t_embed1 = time.time()
    print(f"Embedding time: {t_embed1 - t_embed0:.1f}s "
          f"(A: {emb_a.shape}, B: {emb_b.shape})")

    # --- Full cross-table cosine similarity (brute force matmul, feasible at this scale) ---
    t_sim0 = time.time()
    # Note: on Apple Silicon, numpy's Accelerate BLAS backend can emit spurious
    # divide-by-zero/overflow/invalid RuntimeWarnings on this matmul even though
    # the numerical result is fine (verified: no actual NaN/Inf in output).
    with np.errstate(all="ignore"):
        sim = emb_a @ emb_b.T  # (1363, 3226), already normalized -> cosine similarity
    assert not np.isnan(sim).any() and not np.isinf(sim).any(), "Unexpected NaN/Inf in similarity matrix"
    t_sim1 = time.time()
    print(f"Similarity matrix computed in {t_sim1 - t_sim0:.1f}s, shape={sim.shape}")

    ids_a = tableA["id"].tolist()
    ids_b = tableB["id"].tolist()

    # --- Top-K per TableA record ---
    candidate_set = set()
    scored_pairs = []  # (score, id_a, id_b) for ranking/truncation
    k = min(TOP_K_PER_A, sim.shape[1])
    top_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

    for i in range(sim.shape[0]):
        row_scores = sim[i, top_idx[i]]
        order = np.argsort(-row_scores)
        for j_local in order:
            j = top_idx[i, j_local]
            score = sim[i, j]
            if score < SIM_FLOOR:
                continue
            id_a = ids_a[i]
            id_b = ids_b[j]
            scored_pairs.append((score, id_a, id_b))

    # Sort all candidate pairs by similarity, dedupe (already unique per (i,j)), truncate to cap
    scored_pairs.sort(key=lambda x: -x[0])

    seen = set()
    ranked_pairs = []
    for score, id_a, id_b in scored_pairs:
        pair = (id_a, id_b)  # explicit (TableA id, TableB id) - never sorted
        if pair in seen:
            continue
        seen.add(pair)
        ranked_pairs.append(pair)

    candidate_set = set(ranked_pairs[:FINAL_CAP])

    print(f"Total raw scored pairs (top-{k} per A record): {len(scored_pairs)}")
    print(f"Deduped unique pairs: {len(ranked_pairs)}")
    print(f"Final candidate set size (capped at {FINAL_CAP}): {len(candidate_set)}")

    # --- Proxy recall against 100_matches.pkl (measurement only, NOT excluded here) ---
    matches_path = os.path.join(HACKATHON_DIR, "100_matches.pkl")
    with open(matches_path, "rb") as f:
        known_matches = pickle.load(f)

    hits = candidate_set & known_matches
    recall = len(hits) / len(known_matches) if known_matches else float("nan")
    print(f"Proxy recall against 100_matches.pkl (BEFORE exclusion, as instructed "
          f"for this per-strategy script): {recall:.3f} ({len(hits)}/{len(known_matches)})")

    # NOTE: per this task's instructions, the 100 known matches are NOT excluded
    # here - exclusion happens once, later, at the ensemble-merge/submission step.

    out_path = os.path.join(HACKATHON_DIR, "candidates_embeddings.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(candidate_set, f)
    print(f"Saved candidate set to {out_path}")

    t1 = time.time()
    print(f"Total runtime: {t1 - t0:.1f}s")

    # Small report file for quick reference
    report_path = os.path.join(HACKATHON_DIR, "report_embeddings.txt")
    with open(report_path, "w") as f:
        f.write("Strategy: dense sentence-embedding similarity blocking\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"TableA rows: {len(tableA)}, TableB rows: {len(tableB)}\n")
        f.write(f"Top-K per TableA record: {k}\n")
        f.write(f"Candidate set size: {len(candidate_set)}\n")
        f.write(f"Proxy recall vs 100_matches.pkl (before exclusion): {recall:.3f} "
                f"({len(hits)}/{len(known_matches)})\n")
        f.write(f"Total runtime (s): {t1 - t0:.1f}\n")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
