"""
RESEARCH-ONLY (does NOT touch ensemble_merge.py / final_candidates.pkl).

Question: feature_numeric_tokens.py found numeric_jaccard/has_numeric_mismatch
lift a TOY 2-feature (word_jaccard + token_sort_ratio) base set's CV AUC from
0.789 to 0.810 (standalone AUC 0.632). Does that lift SURVIVE once the model
already has the full, richer 15-feature set from ensemble_merge.py -- cross-
encoder score, TF-IDF word/char cosine, rapidfuzz ratios, abbreviation-expanded
text features, and inferred-manufacturer features? It might be redundant with
those richer signals (e.g. the cross-encoder or char-ngram TF-IDF may already
implicitly pick up numeric-token overlap).

Uses research_features_common.py's faithful 15-feature reproduction (pool +
X_pool), and appends the 2 numeric-token columns computed directly with
feature_numeric_tokens.py's importable functions -- never modifying that file
or ensemble_merge.py.

Methodology: same rigorous multi-seed style used throughout this project.
For each of several negative-sampling seeds, sample N_NEG_MULT*100 pool
negatives (these are strategy near-neighbor candidates, i.e. hard negatives,
not random pairs -- consistent with the rest of the project's honest
evaluation), run 5-fold CV LogisticRegression AUC for base(15) vs plus(17),
report mean/std/min across seeds.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

import research_features_common as rfc
from feature_numeric_tokens import numeric_jaccard, has_numeric_mismatch

BASE = rfc.BASE
N_NEG_MULT = 8
N_SEEDS = 12

d = rfc.build()
pool, pool_index, known, X_pool, FEATS = d["pool"], d["pool_index"], d["known"], d["X_pool"], d["FEATS"]
A_title, B_title = d["A_title"], d["B_title"]

# numeric-token columns for the whole pool
num_cols = np.zeros((len(pool), 2))
for i, (a, b) in enumerate(pool):
    ta, tb = A_title[a], B_title[b]
    num_cols[i, 0] = numeric_jaccard(ta, tb)
    num_cols[i, 1] = has_numeric_mismatch(ta, tb)
X_pool_plus = np.hstack([X_pool, num_cols])

from feature_numeric_tokens import numeric_tokens
n_pos_has_num = sum(1 for a, b in known if numeric_tokens(A_title[a]) or numeric_tokens(B_title[b]))
print(f"of {len(known)} known matches, {n_pos_has_num} have a numeric token on at least one side")

pos = [p for p in known if p in pool_index]
nm = [p for p in pool if p not in known]
cv = StratifiedKFold(5, shuffle=True, random_state=0)

BASE_COLS = list(range(len(FEATS)))          # all 15 original features
PLUS_COLS = BASE_COLS + [len(FEATS), len(FEATS) + 1]  # +2 numeric-token features


def eval_seed(seed):
    rng = np.random.default_rng(seed)
    neg = [nm[i] for i in rng.choice(len(nm), size=N_NEG_MULT * len(pos), replace=False)]
    train = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    idx = np.array([pool_index[p] for p in train])
    Xtr_base = X_pool[idx][:, BASE_COLS]
    Xtr_plus = X_pool_plus[idx][:, PLUS_COLS]

    def cv_auc(X):
        aucs = []
        for tr, te in cv.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        return np.mean(aucs)

    return cv_auc(Xtr_base), cv_auc(Xtr_plus)


seeds = [1, 7, 13, 21, 42, 3, 55, 99, 123, 2024, 77, 888][:N_SEEDS]
base_aucs, plus_aucs = [], []
print(f"\nrunning {N_SEEDS} seeds: full 15-feature base vs +numeric_jaccard/+numeric_mismatch (17 feats)...")
for s in seeds:
    ab, ap = eval_seed(s)
    base_aucs.append(ab); plus_aucs.append(ap)
    print(f"  seed={s:5d}  base(15)={ab:.4f}  plus(17)={ap:.4f}  delta={ap-ab:+.4f}")

base_aucs = np.array(base_aucs); plus_aucs = np.array(plus_aucs)
delta = plus_aucs - base_aucs

# also: standalone AUC of numeric_jaccard alone against a fixed hard-negative eval
# (reuse the 100 known + one negative sample for a quick standalone check)
rng0 = np.random.default_rng(42)
neg0 = [nm[i] for i in rng0.choice(len(nm), size=N_NEG_MULT * len(pos), replace=False)]
eval_pairs = pos + neg0
y0 = np.array([1] * len(pos) + [0] * len(neg0))
nj0 = np.array([numeric_jaccard(A_title[a], B_title[b]) for a, b in eval_pairs])
try:
    standalone_auc = roc_auc_score(y0, nj0)
except Exception:
    standalone_auc = float("nan")

# correlation of numeric_jaccard with the existing jac (word-jaccard, col idx 2) and ce (col idx 8)
idx_eval = np.array([pool_index[p] for p in eval_pairs])
corr_jac = np.corrcoef(nj0, X_pool[idx_eval, 2])[0, 1]
corr_ce = np.corrcoef(nj0, X_pool[idx_eval, 8])[0, 1]

report = []
report.append("=== Numeric-Token Feature vs FULL 15-Feature Set ===\n")
report.append(f"Of {len(known)} known matches, {n_pos_has_num} have a numeric/version token on at least one side.")
report.append(f"Standalone AUC of numeric_jaccard alone (vs pool hard negatives): {standalone_auc:.4f}")
report.append(f"correlation(numeric_jaccard, existing word-jaccard feature): {corr_jac:.4f}")
report.append(f"correlation(numeric_jaccard, existing cross-encoder feature): {corr_ce:.4f}\n")
report.append(f"{N_SEEDS} seeds, 5-fold CV LogisticRegression AUC:")
report.append(f"  base (15 feats)      : mean={base_aucs.mean():.4f}  std={base_aucs.std():.4f}  min={base_aucs.min():.4f}")
report.append(f"  plus (+2 numeric)    : mean={plus_aucs.mean():.4f}  std={plus_aucs.std():.4f}  min={plus_aucs.min():.4f}")
report.append(f"  delta mean: {delta.mean():+.4f}  std: {delta.std():.4f}  (wins: {(delta>0).sum()}/{N_SEEDS})\n")

if delta.mean() > 0.005 and (delta > 0).sum() >= N_SEEDS * 0.7:
    verdict = "STILL ADDS VALUE: numeric-token features give a small but fairly consistent AUC lift even on top of the full 15-feature set -- low-risk, worth adding as 2 extra columns."
elif abs(delta.mean()) <= 0.005:
    verdict = "REDUNDANT: lift essentially vanishes once cross-encoder + char-ngram TF-IDF + abbreviation features are present -- these richer features already capture the numeric/version-token signal. Not worth the added complexity."
else:
    verdict = "NO LONGER HELPS (or hurts slightly) on top of the full feature set -- do not add."
report.append(f"VERDICT: {verdict}")

report_text = "\n".join(report)
print("\n" + report_text)
with open(f"{BASE}/research_numeric_full_report.txt", "w") as f:
    f.write(report_text + "\n")
print(f"\nsaved report -> {BASE}/research_numeric_full_report.txt")
