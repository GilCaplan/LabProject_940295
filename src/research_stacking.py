"""
RESEARCH-ONLY (does NOT touch ensemble_merge.py / final_candidates.pkl).

Question: the deployed blend is a naive equal-weight average of
LogisticRegression + GradientBoosting + MLPClassifier probabilities. Does PROPER
STACKING (generate 5-fold out-of-fold predictions from each of the 3 base models,
then fit a LogisticRegression meta-learner on those 3 OOF columns) beat naive
averaging on honest OOF recall, using the SAME rigorous multi-seed methodology
used throughout this project (average over several negative-sampling seeds,
report mean/std/min, never a single lucky run)?

Uses research_features_common.py's faithful reproduction of ensemble_merge.py's
15-feature set (FEATS) and pool.

Methodology (kept strictly non-leaky):
  For each negative-sampling seed:
    - Build train = 100 positives + N_NEG_MULT*100 sampled negatives.
    - 5-fold CV over `train` (StratifiedKFold). Within each fold's TRAIN split,
      fit the 3 base models: OOF predictions for the fold's TEST split come from
      models fit only on the fold's TRAIN split (never touching the fold's test
      labels) -- standard OOF stacking.
    - Collect the 3 base-model OOF columns for ALL of `train` (out-of-fold, so
      the meta-learner is trained on genuinely held-out base predictions).
    - Fit the LR meta-learner on those OOF columns vs true labels (5-fold CV
      internally for the meta-learner AUC, but the important honest number is
      the outer proxy-recall evaluation below).
    - Naive-average baseline: for the same fold structure, compute the simple
      mean of the 3 base OOF columns per row (no learned weights).
    - Honest OOF top-2000 proxy recall (matching ensemble_merge.py's greedy 1:1
      bipartite selection) is computed by re-scoring the FULL pool: for each of
      the 3 base models, fit on ALL of `train`, predict pool, but OVERRIDE the
      score for pairs IN `train` with their fold-OOF value (exactly mirroring
      ensemble_merge.py's blend_scores(honest=True) pattern) -- both for the
      naive-average pool score and for the stacked (meta-learner-combined) pool
      score.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

import research_features_common as rfc

BASE = rfc.BASE
N_NEG_MULT = 8
N_SEEDS = 12  # honest OOF seeds (more than the deployed 9 since this is a comparison study)

d = rfc.build()
pool, pool_index, known, X_pool, FEATS = d["pool"], d["pool_index"], d["known"], d["X_pool"], d["FEATS"]

pos = [p for p in known if p in pool_index]
nm = [p for p in pool if p not in known]
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def make_models():
    return {
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gb": GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=1500, random_state=0)),
    }


def rank_greedy_bipartite(scores):
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    usedA, usedB = set(), set(); chosen = []; leftover = []
    for i in order:
        a, b = pool[i]
        if a not in usedA and b not in usedB:
            usedA.add(a); usedB.add(b); chosen.append(pool[i])
        else:
            leftover.append(pool[i])
    return chosen + leftover


def proxy_recall(ranked):
    return len(set(ranked[:2000]) & known) / 100


def honest_pool_scores(seed, combine_fn, fit_combiner_fn=None):
    """Returns (pool_score_array). combine_fn(oof_cols_train, y_train, base_pool_cols) -> pool_score
    where oof_cols_train is (n_train,3) OOF base predictions, base_pool_cols is (n_pool,3)
    full-fit base predictions for the whole pool. fit_combiner_fn if given returns a fitted
    combiner used to produce pool_score from base_pool_cols (meta-learner path); for naive
    averaging combine_fn just means()."""
    rng = np.random.default_rng(seed)
    neg = [nm[i] for i in rng.choice(len(nm), size=N_NEG_MULT * len(pos), replace=False)]
    train = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    Xtr = np.array([X_pool[pool_index[p]] for p in train])

    models = make_models()
    n_models = len(models)
    oof_train = np.zeros((len(train), n_models))       # OOF base preds on train rows
    full_pool_base = np.zeros((len(pool), n_models))    # full-fit base preds on WHOLE pool

    for mi, (name, m) in enumerate(models.items()):
        for tr, te in cv.split(Xtr, y):
            mm = clone(m); mm.fit(Xtr[tr], y[tr])
            oof_train[te, mi] = mm.predict_proba(Xtr[te])[:, 1]
        full = clone(m); full.fit(Xtr, y)
        full_pool_base[:, mi] = full.predict_proba(X_pool)[:, 1]

    # honest override: for train rows, pool score should reflect only the fold that held
    # each row out (this mirrors ensemble_merge.py's honest-OOF override pattern)
    train_pos_in_pool = np.array([pool_index[p] for p in train])

    # -------- naive average --------
    naive_pool = full_pool_base.mean(axis=1)
    naive_pool_honest = naive_pool.copy()
    naive_pool_honest[train_pos_in_pool] = oof_train.mean(axis=1)

    # -------- stacking meta-learner (trained ONLY on OOF columns) --------
    meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta.fit(oof_train, y)
    stack_pool = meta.predict_proba(full_pool_base)[:, 1]
    # for train rows, honest stacked score = meta applied to the row's own OOF base columns
    # (meta itself was fit on all OOF rows, so for a fully honest per-row estimate we'd need
    # nested CV; as a practical approximation matching the rest of this pipeline's rigor level,
    # we refit the meta-learner in an outer 5-fold split mirroring cv so train-row predictions
    # are also out-of-fold w.r.t. the meta-learner)
    stack_train_honest = np.zeros(len(train))
    for tr, te in cv.split(oof_train, y):
        mm = LogisticRegression(max_iter=2000, class_weight="balanced")
        mm.fit(oof_train[tr], y[tr])
        stack_train_honest[te] = mm.predict_proba(oof_train[te])[:, 1]
    stack_pool_honest = stack_pool.copy()
    stack_pool_honest[train_pos_in_pool] = stack_train_honest

    return naive_pool_honest, stack_pool_honest


print(f"\nrunning {N_SEEDS} honest-OOF seeds comparing naive averaging vs stacking meta-learner...")
naive_recalls, stack_recalls = [], []
wins = ties = losses = 0
seeds = [1, 7, 13, 21, 42, 3, 55, 99, 123, 2024, 77, 888][:N_SEEDS]
for s in seeds:
    naive_pool, stack_pool = honest_pool_scores(s, None)
    r_naive = proxy_recall(rank_greedy_bipartite(naive_pool))
    r_stack = proxy_recall(rank_greedy_bipartite(stack_pool))
    naive_recalls.append(r_naive); stack_recalls.append(r_stack)
    tag = "WIN " if r_stack > r_naive else ("TIE " if r_stack == r_naive else "LOSS")
    if r_stack > r_naive: wins += 1
    elif r_stack == r_naive: ties += 1
    else: losses += 1
    print(f"  seed={s:5d}  naive={r_naive:.3f}  stack={r_stack:.3f}  {tag}")

naive_recalls = np.array(naive_recalls); stack_recalls = np.array(stack_recalls)

report = []
report.append("=== Stacking Meta-Learner vs Naive Averaging (honest OOF, greedy 1:1 top-2000 proxy recall) ===\n")
report.append(f"{N_SEEDS} negative-sampling seeds: {seeds}\n")
report.append(f"naive average : mean={naive_recalls.mean():.4f}  std={naive_recalls.std():.4f}  min={naive_recalls.min():.4f}  max={naive_recalls.max():.4f}")
report.append(f"stacking      : mean={stack_recalls.mean():.4f}  std={stack_recalls.std():.4f}  min={stack_recalls.min():.4f}  max={stack_recalls.max():.4f}")
report.append(f"stacking - naive per-seed delta mean: {(stack_recalls-naive_recalls).mean():+.4f}")
report.append(f"record: {wins} WINS / {ties} TIES / {losses} LOSSES for stacking vs naive\n")

if stack_recalls.mean() > naive_recalls.mean() + 0.003 and wins > losses:
    verdict = "STACKING WINS: meaningful, consistent improvement -- worth folding into the deployed pipeline."
elif abs(stack_recalls.mean() - naive_recalls.mean()) <= 0.003:
    verdict = "ESSENTIALLY TIED: stacking is not meaningfully better than naive averaging on this 100-positive scale; added complexity (nested CV, meta-learner) not obviously worth it."
else:
    verdict = "NAIVE AVERAGING WINS OR STACKING IS WORSE/NOISIER: do not replace the deployed naive average."
report.append(f"VERDICT: {verdict}")

report_text = "\n".join(report)
print("\n" + report_text)
with open(f"{BASE}/research_stacking_report.txt", "w") as f:
    f.write(report_text + "\n")
print(f"\nsaved report -> {BASE}/research_stacking_report.txt")
