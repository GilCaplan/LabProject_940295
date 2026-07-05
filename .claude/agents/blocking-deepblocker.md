---
name: blocking-deepblocker
description: Implements DeepBlocker-style self-supervised deep representation blocking (autoencoder/hybrid tuple embeddings trained on TableA/TableB directly, no labels needed) as the slowest but potentially highest-recall strategy, writes a truncated top-K candidate set plus a recall report against 100_matches.pkl. Launch this one FIRST/in parallel with the others since it is the slowest to train.
tools: Read, Write, Bash
model: sonnet
---

You are implementing ONE blocking strategy for the entity-matching hackathon in this directory: DeepBlocker-style self-supervised tuple-embedding blocking.

Read `05 - Official Task Instructions (Confirmed).md` and `helpers.py` first for exact format/size constraints (candidate set is a Python `set` of `(id_a, id_b)` tuples, max 2000 pairs, must exclude the 100 pairs in `100_matches.pkl` exactly).

This is a time-boxed reimplementation of the DeepBlocker idea (https://github.com/qcri/DeepBlocker), NOT a requirement to clone/install the actual repo — do not spend time on `git clone`/pip-installing an unfamiliar external package under time pressure. Implement the core idea directly with libraries already available (`sentence-transformers`, `sklearn`, `numpy`, `torch` if present):

1. Load TableA.csv / TableB.csv, build a concatenated text representation per record (title + description + any other useful field), same normalization as the other strategy agents (lowercase, strip punctuation).
2. Build tuple embeddings self-supervised, in order of preference given time budget:
   a. **Hybrid/autoencoder approach (preferred if time allows)**: train a small autoencoder (or use average pooled word embeddings via a lightweight approach like TF-IDF-weighted word vectors) on the concatenated text of BOTH tables combined, unsupervised — this is what makes it "self-supervised" and distinct from the `blocking-embeddings` agent's pretrained sentence-transformer approach.
   b. **Fallback**: if training a custom autoencoder is too slow/complex to get right quickly, use a pretrained sentence-transformer as the embedding function but pair it with `ExactTopKVectorPairing`-style logic (for each TableA record, take its top-K nearest TableB neighbors by cosine/L2 in embedding space, not a global threshold) — this is DeepBlocker's actual pairing strategy and is the part most worth replicating even if the embedding step reuses off-the-shelf embeddings. Note in your report if you fell back to this so it's understood as structurally different from `blocking-embeddings` (per-record top-K nearest-neighbor pairing vs global top-K by score).
3. Use `faiss` (already installed) for efficient nearest-neighbor search if row counts are large enough that brute-force pairwise distance is slow.
4. For each TableA record, take its top-K nearest TableB neighbors (K small, e.g. 3-5, tuned so the total stays well within budget after dedup), union across all TableA records, dedupe (plain set add, no `sorted()` on the tuple — track provenance explicitly), drop self-pairs, truncate to 2000 by nearest-neighbor distance/similarity if the union exceeds budget.
5. Load `100_matches.pkl`, compute recall of your candidate set against it BEFORE excluding those 100 pairs, then explicitly remove any of those 100 exact tuples from the final saved set.
6. Save your candidate set to `candidates_deepblocker.pkl` (plain pickle of the Python set). Do NOT call `save_submission`.

Keep the script self-contained (e.g. `strategy_deepblocker.py`). Report back a concise summary: recall achieved, candidate count, runtime, which embedding/pairing approach was used (custom autoencoder vs pretrained-embeddings-with-per-record-topK fallback), and whether compute/time made the preferred approach practical here.
