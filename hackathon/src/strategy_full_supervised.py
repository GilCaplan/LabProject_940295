"""
strategy_full_supervised.py

Different technique from all other blocking-then-scoring strategies in this repo:
skip blocking entirely and directly score the FULL cross-product of
1363 x 3226 ~= 4.4M pairs with a supervised classifier trained on the
100 known matches vs. sampled negatives, using only features computable
via vectorized matrix operations at this scale (TF-IDF cosine similarity
matrix via sparse matmul, price-difference matrix via broadcasting,
manufacturer exact-match matrix).

Outputs:
  - candidates_full_supervised.pkl : top-K (id_a, id_b) pairs by predicted probability
"""
import time
import pickle
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
A = pd.read_csv("tableA.csv")
B = pd.read_csv("tableB.csv")
with open("100_matches.pkl", "rb") as f:
    known_matches = pickle.load(f)
known_matches = set((int(a), int(b)) for a, b in known_matches)

nA, nB = len(A), len(B)
print(f"TableA: {nA} rows, TableB: {nB} rows -> {nA*nB:,} pairs")

def norm_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

A["text"] = (A["title"].fillna("") + " " + A["manufacturer"].fillna("")).map(norm_text)
B["text"] = (B["title"].fillna("") + " " + B["manufacturer"].fillna("")).map(norm_text)
A["manu_norm"] = A["manufacturer"].map(norm_text)
B["manu_norm"] = B["manufacturer"].map(norm_text)

A_ids = A["id"].to_numpy()
B_ids = B["id"].to_numpy()
A_idx = {int(v): i for i, v in enumerate(A_ids)}
B_idx = {int(v): i for i, v in enumerate(B_ids)}

# ---------------------------------------------------------------------------
# 2. Vectorized feature matrices over the FULL cross product
# ---------------------------------------------------------------------------
# TF-IDF cosine similarity matrix (sparse matmul, no python loops)
vect = TfidfVectorizer(min_df=1, stop_words="english")
combined = pd.concat([A["text"], B["text"]], ignore_index=True)
vect.fit(combined)
tfidf_A = vect.transform(A["text"])  # (nA, V), rows already L2-normalized by TfidfVectorizer
tfidf_B = vect.transform(B["text"])  # (nB, V)

sim_matrix = (tfidf_A @ tfidf_B.T).toarray()  # (nA, nB) cosine similarity since rows are L2-normed
print(f"TF-IDF cosine sim matrix computed: shape {sim_matrix.shape}, "
      f"time {time.time()-t0:.1f}s")

# Price features via broadcasting
priceA = A["price"].to_numpy(dtype=float)  # (nA,)
priceB = B["price"].to_numpy(dtype=float)  # (nB,)
priceA_known = ~np.isnan(priceA)
priceB_known = ~np.isnan(priceB)

priceA_filled = np.where(priceA_known, priceA, 0.0)
priceB_filled = np.where(priceB_known, priceB, 0.0)

abs_diff = np.abs(priceA_filled[:, None] - priceB_filled[None, :])  # (nA, nB)
denom = np.maximum(priceA_filled[:, None], priceB_filled[None, :])
denom = np.where(denom == 0, 1.0, denom)
rel_diff = abs_diff / denom

both_known = priceA_known[:, None] & priceB_known[None, :]
# When price unknown on either side, zero out the diff signal and rely on the flag features
abs_diff = np.where(both_known, abs_diff, 0.0)
rel_diff = np.where(both_known, rel_diff, 0.0)

# Manufacturer exact-normalized-match matrix
manuA = A["manu_norm"].to_numpy()
manuB = B["manu_norm"].to_numpy()
manuA_nonempty = manuA != ""
manuB_nonempty = manuB != ""
manu_match = (manuA[:, None] == manuB[None, :]) & manuA_nonempty[:, None] & manuB_nonempty[None, :]

print(f"All feature matrices built. time {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Helper to assemble a feature row-matrix for a given list of (i, j) index pairs
# (indices into A/B arrays, not raw ids)
# ---------------------------------------------------------------------------
def features_for_pairs(i_idx, j_idx):
    f_sim = sim_matrix[i_idx, j_idx]
    # log1p-compress absolute price diff so a handful of huge-price outliers
    # don't dominate/blow up an unregularized-scale logistic regression
    f_abs = np.log1p(abs_diff[i_idx, j_idx])
    f_rel = rel_diff[i_idx, j_idx]
    f_both_known = both_known[i_idx, j_idx].astype(float)
    f_a_known = priceA_known[i_idx].astype(float)
    f_b_known = priceB_known[j_idx].astype(float)
    f_manu = manu_match[i_idx, j_idx].astype(float)
    return np.column_stack([f_sim, f_abs, f_rel, f_both_known, f_a_known, f_b_known, f_manu])

FEATURE_NAMES = ["tfidf_cos_sim", "price_abs_diff", "price_rel_diff",
                  "both_price_known", "a_price_known", "b_price_known", "manu_exact_match"]

# ---------------------------------------------------------------------------
# 3. Build training set: 100 known positives + sampled negatives
# ---------------------------------------------------------------------------
pos_pairs_idx = []
for a_id, b_id in known_matches:
    if a_id in A_idx and b_id in B_idx:
        pos_pairs_idx.append((A_idx[a_id], B_idx[b_id]))
pos_pairs_idx = np.array(pos_pairs_idx)
print(f"Positive training pairs found: {len(pos_pairs_idx)} / {len(known_matches)}")

rng = np.random.default_rng(42)
N_NEG = 15000
neg_i = rng.integers(0, nA, size=N_NEG * 2)
neg_j = rng.integers(0, nB, size=N_NEG * 2)
known_set_idx = set(map(tuple, pos_pairs_idx.tolist()))
neg_pairs = []
for i, j in zip(neg_i, neg_j):
    if (i, j) not in known_set_idx:
        neg_pairs.append((i, j))
    if len(neg_pairs) >= N_NEG:
        break
neg_pairs_idx = np.array(neg_pairs)
print(f"Negative training pairs sampled: {len(neg_pairs_idx)}")

train_i = np.concatenate([pos_pairs_idx[:, 0], neg_pairs_idx[:, 0]])
train_j = np.concatenate([pos_pairs_idx[:, 1], neg_pairs_idx[:, 1]])
y = np.concatenate([np.ones(len(pos_pairs_idx)), np.zeros(len(neg_pairs_idx))])

X_train = features_for_pairs(train_i, train_j)
print(f"Training matrix: {X_train.shape}")

# ---------------------------------------------------------------------------
# 4. Cross-validated honest performance estimate
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
)
oof_prob_lr = cross_val_predict(logreg, X_train, y, cv=skf, method="predict_proba")[:, 1]
auc_lr = roc_auc_score(y, oof_prob_lr)
ap_lr = average_precision_score(y, oof_prob_lr)
print(f"[CV] LogisticRegression  AUC={auc_lr:.4f}  AP={ap_lr:.4f}")

gbc = GradientBoostingClassifier(
    n_estimators=100, max_depth=2, learning_rate=0.05,
    subsample=0.7, random_state=42
)
oof_prob_gbc = cross_val_predict(gbc, X_train, y, cv=skf, method="predict_proba")[:, 1]
auc_gbc = roc_auc_score(y, oof_prob_gbc)
ap_gbc = average_precision_score(y, oof_prob_gbc)
print(f"[CV] GradientBoosting    AUC={auc_gbc:.4f}  AP={ap_gbc:.4f}")

# Pick the better model by cross-validated AP (handles class imbalance better than AUC here)
if ap_gbc >= ap_lr:
    print("Selected model: GradientBoostingClassifier (higher CV AP)")
    final_model = gbc
else:
    print("Selected model: LogisticRegression (higher CV AP)")
    final_model = logreg

final_model.fit(X_train, y)

# ---------------------------------------------------------------------------
# 5. Score ALL 4.4M pairs (the actual point of this strategy) and take top-K
# ---------------------------------------------------------------------------
TOP_K = 5000

all_i, all_j = np.meshgrid(np.arange(nA), np.arange(nB), indexing="ij")
all_i = all_i.ravel()
all_j = all_j.ravel()
print(f"Scoring full cross-product: {len(all_i):,} pairs")

# Score in chunks to bound peak memory
CHUNK = 2_000_000
scores = np.empty(len(all_i), dtype=np.float32)
for start in range(0, len(all_i), CHUNK):
    end = min(start + CHUNK, len(all_i))
    Xc = features_for_pairs(all_i[start:end], all_j[start:end])
    if hasattr(final_model, "predict_proba"):
        scores[start:end] = final_model.predict_proba(Xc)[:, 1]
    else:
        scores[start:end] = final_model.decision_function(Xc)
    print(f"  scored {end:,}/{len(all_i):,}  ({time.time()-t0:.1f}s elapsed)")

top_idx = np.argpartition(-scores, TOP_K)[:TOP_K]
top_idx = top_idx[np.argsort(-scores[top_idx])]

candidate_set = set()
for k in top_idx:
    a_id = int(A_ids[all_i[k]])
    b_id = int(B_ids[all_j[k]])
    candidate_set.add((a_id, b_id))  # id_a from TableA, id_b from TableB -- never sort/swap

print(f"Final candidate set size: {len(candidate_set)}")

# ---------------------------------------------------------------------------
# 6. Proxy recall against the 100 known matches (do NOT exclude them here)
# ---------------------------------------------------------------------------
hits = known_matches & candidate_set
recall = len(hits) / len(known_matches)
print(f"Proxy recall @ top-{TOP_K}: {recall:.4f}  ({len(hits)}/{len(known_matches)})")
missing = known_matches - candidate_set
print(f"Known matches NOT recovered: {len(missing)}")
if missing:
    print("Missing pairs:", missing)

# Cross-reference against union of existing candidates_*.pkl (if any), to see
# whether this strategy recovers matches no other strategy's pool contained.
import glob
existing_union = set()
for fn in glob.glob("candidates_*.pkl"):
    if fn == "candidates_full_supervised.pkl":
        continue
    try:
        with open(fn, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, (set, list, tuple)):
            existing_union |= set(obj)
    except Exception as e:
        print(f"  (skipping {fn}: {e})")

not_in_existing = known_matches - existing_union
recovered_novel = hits & not_in_existing
print(f"Known matches missing from ALL existing candidates_*.pkl: {len(not_in_existing)} -> {not_in_existing}")
print(f"Of those, recovered by this full-supervised strategy: {len(recovered_novel)} -> {recovered_novel}")

with open("candidates_full_supervised.pkl", "wb") as f:
    pickle.dump(candidate_set, f)

print(f"Saved candidates_full_supervised.pkl ({len(candidate_set)} pairs)")
print(f"Total runtime: {time.time()-t0:.1f}s")
