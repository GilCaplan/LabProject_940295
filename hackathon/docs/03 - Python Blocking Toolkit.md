# Python Blocking Toolkit

Ready-to-adapt code for Sunday. Goal: go from two dataframes to a deduplicated, size-capped, ranked list of candidate pairs as fast as possible. Pair with [[01 - Blocking Strategies Cheat Sheet]] for when to use which piece, and [[02 - Evaluation Metrics]] to validate before submitting.

## Setup (run Saturday, before the event — don't burn hackathon time on installs)
```bash
pip install pandas numpy scikit-learn rapidfuzz sentence-transformers faiss-cpu datasketch recordlinkage
```
- `pandas`/`numpy` — data handling
- `rapidfuzz` — fast string similarity (Levenshtein, token sort ratio) in C, much faster than `fuzzywuzzy`
- `scikit-learn` — TF-IDF vectorizer + cosine similarity for token-overlap scoring
- `sentence-transformers` — semantic embeddings
- `faiss-cpu` — fast approximate nearest neighbor search over embeddings
- `datasketch` — MinHash/LSH if the dataset is very large
- `recordlinkage` — verified, battle-tested `Index` class with built-in `.block()` / `.sortedneighbourhood()` — a faster starting point than hand-rolled loops (see section 0 below)

## 0. Fastest path: recordlinkage's built-in Index (verified API)
If you want a correct baseline in minutes instead of writing loops by hand:
```python
import recordlinkage as rl

indexer = rl.Index()
indexer.block(left_on="brand", right_on="brand")           # exact-match blocking key(s)
indexer.sortedneighbourhood(left_on="title_norm", right_on="title_norm", window=9)  # add a second pass
candidate_pairs = indexer.index(df1, df2)  # pandas MultiIndex of (idx1, idx2) tuples
```
Multiple `.block()`/`.sortedneighbourhood()` calls before `.index()` are unioned automatically (recordlinkage's default `Index` object stacks all added algorithms). Convert to a plain set of tuples with `list(candidate_pairs)` and feed straight into the scoring functions in section 4/5 below.

## 1. Normalization (do this first, always)
```python
import re

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)   # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

STOPWORDS = {"inc", "the", "a", "edition", "version", "co", "corp", "ltd"}

def normalize_tokens(s: str) -> list[str]:
    return [t for t in normalize(s).split() if t not in STOPWORDS]
```

## 2. Token-based blocking (inverted index) — fast, high recall, first pass
```python
from collections import defaultdict
from itertools import combinations

def token_blocking(df1, df2, text_col, max_block_size=500):
    """df1, df2 must have a unique 'id' column. Returns a set of (id1, id2) candidate pairs."""
    inverted = defaultdict(lambda: [[], []])  # token -> ([ids from df1], [ids from df2])
    for i, row in df1.iterrows():
        for tok in set(normalize_tokens(row[text_col])):
            inverted[tok][0].append(row["id"])
    for i, row in df2.iterrows():
        for tok in set(normalize_tokens(row[text_col])):
            inverted[tok][1].append(row["id"])

    pairs = set()
    for tok, (ids1, ids2) in inverted.items():
        if len(ids1) == 0 or len(ids2) == 0:
            continue
        if len(ids1) * len(ids2) > max_block_size * max_block_size:
            continue  # drop overly-common tokens to protect runtime/budget
        for a in ids1:
            for b in ids2:
                pairs.add((a, b))
    return pairs
```
For **deduplication within a single dataset** (one table, not two), use `combinations(ids, 2)` instead of the cross product, and dedupe the token index the same way.

## 3. Sorted Neighborhood (single or multi-pass)
```python
def sorted_neighborhood(df, key_col, window=5):
    """Within one combined dataframe (concat df1+df2 with an 'id' and 'source' col)."""
    sorted_df = df.sort_values(key_col).reset_index(drop=True)
    pairs = set()
    n = len(sorted_df)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            a, b = sorted_df.loc[i, "id"], sorted_df.loc[j, "id"]
            # skip pairs within the same source table if you only want cross-table matches
            if sorted_df.loc[i, "source"] != sorted_df.loc[j, "source"]:
                pairs.add(tuple(sorted([a, b])))
    return pairs
```
Run this multiple times with different `key_col` (e.g., normalized title, then normalized brand+model) and union the results — multi-pass beats single-key blocking almost every time.

## 3b. LSH blocking with datasketch (verified API, use for very large datasets)
```python
from datasketch import MinHash, MinHashLSH

def lsh_candidates(df1, df2, text_col, threshold=0.4, num_perm=128):
    """Jaccard-similarity LSH blocking. Good when token blocking's inverted index gets too slow/large."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    for _, row in df2.iterrows():
        m = MinHash(num_perm=num_perm)
        for tok in set(normalize_tokens(row[text_col])):
            m.update(tok.encode("utf8"))
        minhashes[row["id"]] = m
        lsh.insert(row["id"], m)

    pairs = set()
    for _, row in df1.iterrows():
        m = MinHash(num_perm=num_perm)
        for tok in set(normalize_tokens(row[text_col])):
            m.update(tok.encode("utf8"))
        for match_id in lsh.query(m):
            pairs.add((row["id"], match_id))
    return pairs
```
Lower `threshold` = higher recall, more (noisier) candidates — tune based on your budget. `num_perm` trades accuracy for speed (128 is a reasonable default).

## 4. Scoring candidate pairs (turns blocking output into a ranking)
```python
from rapidfuzz import fuzz

def score_pairs(pairs, id_to_text):
    """id_to_text: dict id -> normalized text string. Returns list of (score, id1, id2)."""
    scored = []
    for a, b in pairs:
        score = fuzz.token_sort_ratio(id_to_text[a], id_to_text[b])  # 0-100
        scored.append((score, a, b))
    scored.sort(reverse=True)  # highest similarity first
    return scored
```
`token_sort_ratio` handles reordered words well and is very fast. For numeric/ID fields, add an exact-match bonus (e.g., `+50` if model numbers match exactly) since those are strong, low-noise signals per the DITTO domain-knowledge idea from the lecture.

## 5. TF-IDF cosine scoring (alternative/complementary signal)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def tfidf_score_pairs(pairs, id_to_text):
    ids = list(id_to_text.keys())
    idx = {rid: i for i, rid in enumerate(ids)}
    vec = TfidfVectorizer().fit([id_to_text[i] for i in ids])
    X = vec.transform([id_to_text[i] for i in ids])
    scored = []
    for a, b in pairs:
        sim = cosine_similarity(X[idx[a]], X[idx[b]])[0, 0]
        scored.append((sim, a, b))
    scored.sort(reverse=True)
    return scored
```

## 6. Semantic/embedding blocking with ANN (best recall, use if time/compute allows)
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def semantic_candidates(df1, df2, text_col, top_k=10, model_name="all-MiniLM-L6-v2"):
    # all-MiniLM-L6-v2: safe, fast, well-tested default (~80MB, CPU-friendly).
    # If quality matters more than speed and you have GPU/time: "BAAI/bge-small-en-v1.5"
    # is a stronger 2024/2025-era small model at similar size — swap model_name and re-test.
    model = SentenceTransformer(model_name)
    emb1 = model.encode(df1[text_col].map(normalize).tolist(), normalize_embeddings=True)
    emb2 = model.encode(df2[text_col].map(normalize).tolist(), normalize_embeddings=True)

    index = faiss.IndexFlatIP(emb2.shape[1])  # inner product == cosine since normalized
    index.add(np.array(emb2, dtype="float32"))

    scores, neighbors = index.search(np.array(emb1, dtype="float32"), top_k)

    scored = []
    for i, row in df1.reset_index(drop=True).iterrows():
        for rank in range(top_k):
            j = neighbors[i, rank]
            if j == -1:
                continue
            scored.append((float(scores[i, rank]), row["id"], df2.iloc[j]["id"]))
    scored.sort(reverse=True)
    return scored
```
`top_k` per record controls how many candidates each record contributes — tune it down if it blows the pair budget, up if you have budget to spare and want better recall.

## 7. Dedupe + truncate to budget (the step everyone forgets)

**Hackathon-specific correction:** the `tuple(sorted((a, b)))` pattern below is for generic single-table dedup. For this task, output pairs must be `(id_a, id_b)` with `id_a` always from TableA and `id_b` always from TableB — sorting can flip that if the two tables' id namespaces overlap. `solution_starter.py` and [[05 - Official Task Instructions (Confirmed)]] use plain `(a, b)` tuples (no sorting) for exactly this reason; adapt the snippets below to drop the `sorted(...)` call.

```python
def finalize(scored_pairs, K):
    """scored_pairs: list of (score, id1, id2), possibly with duplicates/self-pairs."""
    seen = set()
    out = []
    for score, a, b in scored_pairs:
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append((score, a, b))
        if len(out) >= K:
            break
    return out
```
Always run every candidate source (token blocking + sorted neighborhood + embeddings) through this **one** shared dedupe/truncate step at the end, not per-method — union everything, sort by best available score, then cut to K.

This `finalize()` is exactly **Cardinality Edge Pruning (CEP)** from the meta-blocking literature — see [[01 - Blocking Strategies Cheat Sheet]]'s "Formal framing" section. The alternative worth A/B-testing is **Cardinality Node Pruning (CNP)**: instead of one global cutoff, cap each *record's* candidates at `k` before combining, which stops a handful of "hub" records (generic titles that fuzzy-match everything) from eating your whole budget and starving other records of any candidates at all.

```python
from collections import defaultdict

def finalize_cnp(scored_pairs, k_per_record, K):
    """CNP-style: cap candidates per record first, then dedupe/truncate globally to K."""
    by_record = defaultdict(list)
    for score, a, b in scored_pairs:
        if a == b:
            continue
        by_record[a].append((score, a, b))
        by_record[b].append((score, a, b))

    capped = set()
    for rid, plist in by_record.items():
        plist.sort(reverse=True)
        for score, a, b in plist[:k_per_record]:
            capped.add((score, tuple(sorted((a, b)))))

    capped = sorted(capped, reverse=True)
    seen, out = set(), []
    for score, key in capped:
        if key in seen:
            continue
        seen.add(key)
        out.append((score, key[0], key[1]))
        if len(out) >= K:
            break
    return out
```
Try both `finalize()` (global CEP) and `finalize_cnp()` (CNP) against a labeled validation sample and keep whichever yields higher PC — see [[02 - Evaluation Metrics]].

## 8. Full pipeline skeleton
```python
# 1. load + normalize
df1, df2 = load_data()  # add 'id' cols
id_to_text = {**{r.id: normalize(r[text_col]) for r in df1.itertuples()},
              **{r.id: normalize(r[text_col]) for r in df2.itertuples()}}

# 2. generate candidates from multiple strategies, union them
cands = set()
cands |= token_blocking(df1, df2, text_col)
cands |= {(a, b) for _, a, b in semantic_candidates(df1, df2, text_col, top_k=15)}

# 3. score all unioned candidates with the strongest available signal
scored = score_pairs(cands, id_to_text)

# 4. dedupe + cut to the exact size limit K from the brief
final_pairs = finalize(scored, K=SIZE_LIMIT)

# 5. write output in whatever format the grading doc specifies (check column names/order!)
```

## Sanity checks before submitting
- `len(final_pairs) <= K` exactly (or `==` if the brief requires exact size).
- No self-pairs, no duplicate unordered pairs.
- If given a labeled sample, compute PC/PQ locally (see [[02 - Evaluation Metrics]]) and sanity check the number isn't suspiciously low (e.g., <20%) before running out of time.
- Output file matches the required schema (id column names, delimiter, header) — a wrong format can zero out an otherwise good solution.

## If you have time before Sunday: try DeepBlocker
[DeepBlocker](https://github.com/qcri/DeepBlocker) (QCRI, VLDB'21) is self-supervised — no labeled data needed, matching this task's constraints — and directly implements the top-K framing:
```python
from tuple_embedding_models import AutoEncoderTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
from deep_blocker import DeepBlocker

tuple_embedding_model = AutoEncoderTupleEmbedding()
topK_vector_pairing_model = ExactTopKVectorPairing(K=SIZE_LIMIT // len(df1))  # top-K neighbors per left record
db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
candidate_set_df = db.block_datasets(df1, df2, cols_to_block=["title", "brand", "modelno"])
```
Requires cloning the repo (it's not a pip package) and its `requirements.txt` (fastText pretrained embeddings + a DL framework) — worth test-running once Saturday so it isn't a surprise Sunday. If setup friction is too high on the day, fall back to the hand-rolled embedding pipeline in section 6, which achieves a similar effect with sentence-transformers + FAISS.

## Sources (verified against official docs/repos)
- [recordlinkage docs — Index/comparing](https://recordlinkage.readthedocs.io/en/latest/ref-index.html)
- [RapidFuzz docs — fuzz scorers](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html)
- [DeepBlocker repo](https://github.com/qcri/DeepBlocker)
- [sentence-transformers docs](https://www.sbert.net/)
