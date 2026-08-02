"""
Metric-learning blocking strategy: fine-tune the same off-the-shelf
all-MiniLM-L6-v2 model (already used by strategy_embeddings.py, proxy
recall 0.82) using MultipleNegativesRankingLoss on the 100 known matches,
so the embedding space is reshaped specifically for this dataset's notion
of "same product" - not just picking a similarity threshold.

Pipeline:
  1. Load TableA/TableB/100_matches.pkl.
  2. Build (text_a, text_b) positive training pairs from the 100 known
     matches (normalized title + manufacturer text on each side).
  3. Fine-tune all-MiniLM-L6-v2 with MultipleNegativesRankingLoss (in-batch
     negatives - only needs positive pairs, most robust choice given just
     100 labeled examples). Save to finetuned_model/.
  4. Re-embed all of TableA/TableB with the fine-tuned model, brute-force
     cosine similarity (1363 x 3226), top-k per TableA record -> candidate set.
  5. Honest evaluation: 5-fold CV over the 100 matches - fine-tune on 4/5
     folds only, measure recall on the held-out fold, average across folds.
     This is reported SEPARATELY from the naive (optimistic, trained-on-all)
     proxy recall, since a model fine-tuned on pairs it is then evaluated on
     has effectively memorized them.

Output: candidates_finetuned_embeddings.pkl (Python set of (id_a, id_b)).
100_matches.pkl is used only for fine-tuning / recall measurement, per
hackathon rules - it is NOT excluded from the candidate set here (that
happens once, later, at ensemble-merge/submission time).
"""

import os
import pickle
import time
import copy

import numpy as np
import pandas as pd

HACKATHON_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FINAL_MODEL_DIR = os.path.join(HACKATHON_DIR, "finetuned_model")
TOP_K_PER_A = 4
FINAL_CAP = 3000

N_EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
N_FOLDS = 5
RANDOM_SEED = 42


def build_text(row):
    title = str(row.get("title", "") or "")
    manufacturer = row.get("manufacturer", "")
    manufacturer = "" if pd.isna(manufacturer) else str(manufacturer)
    parts = [title]
    if manufacturer.strip():
        parts.append(manufacturer)
    return " ".join(p.strip() for p in parts if p.strip())


def fine_tune_model(base_model_name, train_pairs, epochs, batch_size, lr, save_path=None, verbose=True):
    """Fine-tune all-MiniLM-L6-v2 on positive (text_a, text_b) pairs using
    MultipleNegativesRankingLoss (in-batch negatives)."""
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(base_model_name)

    examples = [InputExample(texts=[a, b]) for a, b in train_pairs]
    # drop_last=True so every batch has >=2 examples for in-batch negatives to be meaningful
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=min(batch_size, len(examples)),
                                   drop_last=len(examples) > batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(1, int(len(train_dataloader) * epochs * 0.1))
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=verbose,
    )

    if save_path:
        model.save(save_path)
    return model


def embed_and_topk_candidates(model, tableA, tableB, texts_a, texts_b, top_k, final_cap):
    emb_a = model.encode(texts_a, batch_size=64, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    emb_b = model.encode(texts_b, batch_size=64, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)

    with np.errstate(all="ignore"):
        sim = emb_a @ emb_b.T
    assert not np.isnan(sim).any() and not np.isinf(sim).any()

    ids_a = tableA["id"].tolist()
    ids_b = tableB["id"].tolist()

    k = min(top_k, sim.shape[1])
    top_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

    scored_pairs = []
    for i in range(sim.shape[0]):
        row_scores = sim[i, top_idx[i]]
        order = np.argsort(-row_scores)
        for j_local in order:
            j = top_idx[i, j_local]
            score = sim[i, j]
            scored_pairs.append((score, ids_a[i], ids_b[j]))

    scored_pairs.sort(key=lambda x: -x[0])
    seen = set()
    ranked_pairs = []
    for score, id_a, id_b in scored_pairs:
        pair = (id_a, id_b)
        if pair in seen:
            continue
        seen.add(pair)
        ranked_pairs.append(pair)

    return set(ranked_pairs[:final_cap])


def run_cross_validation(tableA, tableB, texts_a, texts_b, known_matches_list, id_to_text_a, id_to_text_b):
    """5-fold CV: fine-tune on 4/5 of the 100 matches, measure recall on the held-out fold.
    Uses a small top-k (bounded by fold pair count context) but same pipeline as final model,
    just trained on fewer positives and evaluated only on the held-out fold's pairs."""
    rng = np.random.default_rng(RANDOM_SEED)
    matches = list(known_matches_list)
    rng.shuffle(matches)

    folds = np.array_split(matches, N_FOLDS)
    fold_recalls = []

    for fold_i in range(N_FOLDS):
        held_out = list(folds[fold_i])
        train_matches = [p for j, f in enumerate(folds) if j != fold_i for p in f]

        train_pairs = [(id_to_text_a[a], id_to_text_b[b]) for a, b in train_matches]

        print(f"  [CV fold {fold_i+1}/{N_FOLDS}] train={len(train_pairs)} held_out={len(held_out)}")
        fold_model = fine_tune_model(MODEL_NAME, train_pairs, N_EPOCHS, BATCH_SIZE, LEARNING_RATE,
                                      save_path=None, verbose=False)

        candidate_set = embed_and_topk_candidates(fold_model, tableA, tableB, texts_a, texts_b,
                                                   TOP_K_PER_A, FINAL_CAP)

        held_out_set = set((int(a), int(b)) for a, b in held_out)
        hits = candidate_set & held_out_set
        recall = len(hits) / len(held_out_set) if held_out_set else float("nan")
        print(f"  [CV fold {fold_i+1}/{N_FOLDS}] held-out recall: {recall:.3f} ({len(hits)}/{len(held_out_set)})")
        fold_recalls.append(recall)

        del fold_model

    return fold_recalls


def main():
    t0 = time.time()

    tableA = pd.read_csv(os.path.join(HACKATHON_DIR, "tableA.csv"))
    tableB = pd.read_csv(os.path.join(HACKATHON_DIR, "tableB.csv"))
    print(f"TableA rows: {len(tableA)}, TableB rows: {len(tableB)}")

    matches_path = os.path.join(HACKATHON_DIR, "100_matches.pkl")
    with open(matches_path, "rb") as f:
        known_matches = pickle.load(f)
    known_matches = set((int(a), int(b)) for a, b in known_matches)
    print(f"Loaded {len(known_matches)} known matches for fine-tuning / recall measurement")

    texts_a = tableA.apply(build_text, axis=1).tolist()
    texts_b = tableB.apply(build_text, axis=1).tolist()
    id_to_text_a = dict(zip(tableA["id"].tolist(), texts_a))
    id_to_text_b = dict(zip(tableB["id"].tolist(), texts_b))

    try:
        import sentence_transformers  # noqa
        import torch  # noqa
    except ImportError as e:
        print("FATAL: sentence-transformers/torch not installed. Stopping.")
        print(repr(e))
        return

    # ---------------------------------------------------------------
    # Step A: HONEST evaluation via 5-fold cross-validation (BEFORE
    # touching the full-data model), so held-out folds are never seen
    # during their own fine-tuning.
    # ---------------------------------------------------------------
    print("\n=== Honest cross-validated recall (5-fold) ===")
    t_cv0 = time.time()
    fold_recalls = run_cross_validation(tableA, tableB, texts_a, texts_b, known_matches,
                                         id_to_text_a, id_to_text_b)
    honest_cv_recall = float(np.mean(fold_recalls))
    t_cv1 = time.time()
    print(f"Per-fold recalls: {[round(r, 3) for r in fold_recalls]}")
    print(f"HONEST cross-validated recall (avg across {N_FOLDS} folds): {honest_cv_recall:.3f}")
    print(f"CV runtime: {t_cv1 - t_cv0:.1f}s")

    # ---------------------------------------------------------------
    # Step B: fine-tune on ALL 100 known matches for the final candidate
    # set (maximum quality). Recall measured against these same 100 pairs
    # will be optimistic/inflated since the model has effectively
    # memorized them - this is expected and reported as such.
    # ---------------------------------------------------------------
    print("\n=== Fine-tuning final model on ALL 100 known matches ===")
    t_ft0 = time.time()
    train_pairs_all = [(id_to_text_a[a], id_to_text_b[b]) for a, b in known_matches]
    final_model = fine_tune_model(MODEL_NAME, train_pairs_all, N_EPOCHS, BATCH_SIZE, LEARNING_RATE,
                                   save_path=FINAL_MODEL_DIR, verbose=True)
    t_ft1 = time.time()
    print(f"Fine-tuning time: {t_ft1 - t_ft0:.1f}s. Model saved to {FINAL_MODEL_DIR}")

    print("\n=== Re-embedding all records with fine-tuned model + brute-force cosine sim ===")
    t_embed0 = time.time()
    candidate_set = embed_and_topk_candidates(final_model, tableA, tableB, texts_a, texts_b,
                                               TOP_K_PER_A, FINAL_CAP)
    t_embed1 = time.time()
    print(f"Embedding + similarity + top-k time: {t_embed1 - t_embed0:.1f}s")
    print(f"Final candidate set size (capped at {FINAL_CAP}): {len(candidate_set)}")

    naive_hits = candidate_set & known_matches
    naive_recall = len(naive_hits) / len(known_matches) if known_matches else float("nan")
    print(f"\nNAIVE proxy recall vs 100_matches.pkl (model trained on these SAME pairs - "
          f"INFLATED/optimistic, NOT a fair generalization estimate): "
          f"{naive_recall:.3f} ({len(naive_hits)}/{len(known_matches)})")
    print(f"HONEST cross-validated recall (model never saw held-out pairs during their fold's "
          f"training - meaningful generalization estimate): {honest_cv_recall:.3f}")

    out_path = os.path.join(HACKATHON_DIR, "candidates_finetuned_embeddings.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(candidate_set, f)
    print(f"\nSaved candidate set to {out_path}")

    t1 = time.time()
    print(f"Total runtime: {t1 - t0:.1f}s")

    report_path = os.path.join(HACKATHON_DIR, "report_finetuned_embeddings.txt")
    with open(report_path, "w") as f:
        f.write("Strategy: fine-tuned (metric-learning) sentence-embedding similarity blocking\n")
        f.write(f"Base model: {MODEL_NAME}\n")
        f.write(f"Loss: MultipleNegativesRankingLoss (in-batch negatives), epochs={N_EPOCHS}, "
                f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE}\n")
        f.write(f"TableA rows: {len(tableA)}, TableB rows: {len(tableB)}\n")
        f.write(f"Top-K per TableA record: {TOP_K_PER_A}\n")
        f.write(f"Final candidate set size: {len(candidate_set)}\n\n")
        f.write(f"NAIVE proxy recall (trained on all 100, evaluated on same 100 - INFLATED): "
                f"{naive_recall:.3f} ({len(naive_hits)}/{len(known_matches)})\n")
        f.write(f"HONEST cross-validated recall (5-fold, held-out never trained on): "
                f"{honest_cv_recall:.3f}\n")
        f.write(f"Per-fold recalls: {[round(r, 3) for r in fold_recalls]}\n\n")
        f.write(f"Fine-tuning time (final model, all 100 pairs): {t_ft1 - t_ft0:.1f}s\n")
        f.write(f"Cross-validation total time (5 folds): {t_cv1 - t_cv0:.1f}s\n")
        f.write(f"Total script runtime (s): {t1 - t0:.1f}\n")
        f.write(f"Fine-tuned model saved to: {FINAL_MODEL_DIR}\n")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
