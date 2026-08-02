"""
Reciprocal Rank Fusion (RRF) — exploratory alternative COMBINATION technique.

This is NOT a new candidate-generation strategy. It's an alternative way to
combine the same candidates_*.pkl strategy outputs that ensemble_merge.py
already pools, but using a classical, label-free rank-fusion formula instead
of a trained classifier.

RRF formula (k=60 standard default):
    RRF_score(pair) = sum over strategies s that contain `pair` of
                      1 / (k + rank_s(pair))
where rank_s(pair) is the pair's 0-indexed rank within strategy s's own
candidate list (rank 0 = that strategy's top choice).

------------------------------------------------------------------------
Rank reconstruction problem
------------------------------------------------------------------------
Every strategy_<name>.py script computes some internal score, sorts pairs
by it, truncates to a target cap, and ONLY THEN converts to a plain Python
`set()` before pickling (see e.g. strategy_rules.py:250-251,
strategy_lsh.py:144-147, strategy_embeddings.py:119-130, etc.). The saved
candidates_*.pkl files are therefore *unordered sets* -- none of the 14
scripts persist their per-pair scores or ranks alongside the final set
(the lone exception, candidates_lsh_report.pkl, is just aggregate summary
stats, not per-pair scores).

Faithfully reproducing all 14 bespoke scoring formulas (several of which
depend on cached ML models: sentence-transformers embeddings, a
cross-encoder, a DeepBlocker-style autoencoder, word2vec) purely to recover
"true" per-pair rank would mean re-running most of the expensive strategies
end-to-end a second time -- expensive and, worse, error-prone (a
reimplementation that subtly diverges from the original script would yield
a rank order that *looks* real but isn't).

Given that constraint, this script uses a single, cheap, CONSISTENT proxy
score (char-ngram TF-IDF title cosine + price closeness) to approximate the
within-strategy rank order for every strategy uniformly. This is an
explicit approximation, not each strategy's true internal ranking -- but it
is a reasonable stand-in (most of these strategies' internal scores are
themselves driven substantially by title textual similarity) and, critically,
it is applied identically everywhere so no strategy is unfairly advantaged.

For comparison/robustness we ALSO compute a pure presence/tied-rank RRF
variant (every pair gets that strategy's *average* rank, i.e. this reduces
to a weighted multi-strategy-agreement count) and report both.

Quirk handled: candidates_deepblocker.pkl already had the 100 known matches
pre-excluded by its generator (strategy_deepblocker.py). This does NOT bias
RRF the way it biased the supervised classifier's features (RRF never
trains on the known matches at all -- they're used only for the recall
readout at the end), it just means deepblocker can never itself contribute
to a known match's RRF score.

Pairs are always (id_a from TableA, id_b from TableB); NEVER sorted() --
dedup via plain set union/add only.
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import pickle
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from helpers import validate_candidate_set

BASE = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"
RNG = np.random.default_rng(42)
K_RRF = 60
CUTOFFS = [1500, 2000, 2500]
FINAL_CAP = 2000

t0 = time.time()

# ------------------------------------------------------------------ load
tableA = pd.read_csv(f"{BASE}/tableA.csv")
tableB = pd.read_csv(f"{BASE}/tableB.csv")
tableA_ids, tableB_ids = set(tableA["id"]), set(tableB["id"])

with open(f"{BASE}/100_matches.pkl", "rb") as f:
    known = {(int(a), int(b)) for a, b in pickle.load(f)}

strategy_files = sorted(glob.glob(f"{BASE}/candidates_*.pkl"))
strategy_files = [f for f in strategy_files if "report" not in f]

strat_sets = {}
for fpath in strategy_files:
    name = fpath.split("candidates_")[-1].replace(".pkl", "")
    with open(fpath, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, set):
        continue  # skip non-candidate-set files defensively
    strat_sets[name] = {(int(a), int(b)) for a, b in obj}

print(f"Loaded {len(strat_sets)} strategies: {sorted(strat_sets)}")
if "deepblocker" in strat_sets:
    print("  note: deepblocker had the 100 known matches pre-excluded at save time")
if "full_supervised" in strat_sets:
    print("  note: full_supervised's own candidates were themselves produced by a "
          "classifier trained on the 100 known matches -- its presence in the RRF "
          "pool is not a fully label-free signal; included anyway since it's a "
          "generated candidates_*.pkl file, but flagged for the report.")

pool = set().union(*strat_sets.values())
print(f"pool size (union of all strategies): {len(pool)}   "
      f"(includes {len(pool & known)}/100 known matches)")

# ------------------------------------------------------------------ cheap proxy score
# Used ONLY to approximate within-strategy rank order (see module docstring) --
# not used to decide strategy membership, only to order pairs already proposed
# by that strategy.
def norm(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

A_title = {int(r.id): norm(r.title) for r in tableA.itertuples()}
B_title = {int(r.id): norm(r.title) for r in tableB.itertuples()}
A_price = {int(r.id): r.price for r in tableA.itertuples()}
B_price = {int(r.id): r.price for r in tableB.itertuples()}

vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
vc.fit(list(A_title.values()) + list(B_title.values()))
Ac = {i: vc.transform([t]) for i, t in A_title.items()}
Bc = {i: vc.transform([t]) for i, t in B_title.items()}

def ccos(a, b):
    return float(Ac[a].multiply(Bc[b]).sum())

def pclose(a, b):
    pa, pb = A_price.get(a), B_price.get(b)
    if pd.isna(pa) or pd.isna(pb):
        return 0.0
    m = max(abs(pa), abs(pb), 1.0)
    return 1.0 - min(abs(pa - pb) / m, 1.0)

print("computing proxy score for pooled pairs...")
proxy_score = {}
for (a, b) in pool:
    proxy_score[(a, b)] = 0.7 * ccos(a, b) + 0.3 * pclose(a, b)
print(f"  done in {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ RRF (approx-rank variant)
rrf_approx = {p: 0.0 for p in pool}
rrf_tied = {p: 0.0 for p in pool}
n_strats_map = {p: 0 for p in pool}

for name, s in strat_sets.items():
    ordered = sorted(s, key=lambda p: -proxy_score[p])
    n = len(ordered)
    avg_rank = (n - 1) / 2.0
    tied_weight = 1.0 / (K_RRF + avg_rank)
    for rank, p in enumerate(ordered):
        rrf_approx[p] += 1.0 / (K_RRF + rank)
        rrf_tied[p] += tied_weight
        n_strats_map[p] += 1

ranked_approx = sorted(pool, key=lambda p: -rrf_approx[p])
ranked_tied = sorted(pool, key=lambda p: -rrf_tied[p])

def proxy_recall(ranked, cutoff):
    return len(set(ranked[:cutoff]) & known) / 100

print("\n--- RRF (approx within-strategy rank via proxy score) ---")
for c in CUTOFFS:
    print(f"  top-{c}: proxy recall = {proxy_recall(ranked_approx, c):.3f}")

print("\n--- RRF (tied rank / pure presence-weighted, for comparison) ---")
for c in CUTOFFS:
    print(f"  top-{c}: proxy recall = {proxy_recall(ranked_tied, c):.3f}")

print("\n--- per-strategy proxy recall (context) ---")
for s in sorted(strat_sets):
    print(f"  {s:20s} n={len(strat_sets[s]):6d} recall={len(strat_sets[s] & known) / 100:.3f}")

n_strats_agree = np.array([n_strats_map[p] for p in ranked_approx[:2000]])
print(f"\ntop-2000 (RRF approx) mean #strategies-agreeing: {n_strats_agree.mean():.2f}, "
      f"median: {np.median(n_strats_agree):.0f}")

print(f"\n--- reference: supervised blended classifier (ensemble_merge.py) honest OOF "
      f"top-2000 proxy recall = 0.82 ---")

# ------------------------------------------------------------------ hybrid: RRF score as an extra feature
print("\n--- hybrid check: RRF score as one additional feature in a lightweight "
      "logistic-regression classifier (honest OOF, 5-fold CV) ---")
pool_list = list(pool)
pool_index = {p: i for i, p in enumerate(pool_list)}

def jac(a, b):
    sa, sb = set(A_title[a].split()), set(B_title[b].split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

FEATS = ["ccos", "jac", "pclose", "n_strats", "rrf"]
X = np.zeros((len(pool_list), len(FEATS)))
for i, (a, b) in enumerate(pool_list):
    X[i, 0] = ccos(a, b)
    X[i, 1] = jac(a, b)
    X[i, 2] = pclose(a, b)
    X[i, 3] = n_strats_map[(a, b)]
    X[i, 4] = rrf_approx[(a, b)]

pos = [p for p in known if p in pool_index]
nm = [p for p in pool_list if p not in known]
neg_idx = RNG.choice(len(nm), size=min(8 * len(pos), len(nm)), replace=False)
neg = [nm[i] for i in neg_idx]
train = pos + neg
y = np.array([1] * len(pos) + [0] * len(neg))
Xtr = np.array([X[pool_index[p]] for p in train])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
oof = np.zeros(len(train))
for tr_i, te_i in cv.split(Xtr, y):
    mm = clone(model)
    mm.fit(Xtr[tr_i], y[tr_i])
    oof[te_i] = mm.predict_proba(Xtr[te_i])[:, 1]

full = clone(model).fit(Xtr, y)
scores_full = full.predict_proba(X)[:, 1].copy()
for j, p in enumerate(train):
    scores_full[pool_index[p]] = oof[j]  # honest override for labeled pairs

hybrid_ranked = [pool_list[i] for i in np.argsort(-scores_full)]
hybrid_recall_2000 = proxy_recall(hybrid_ranked, 2000)
print(f"  hybrid (rrf-as-feature) logistic regression honest OOF top-2000 proxy recall: "
      f"{hybrid_recall_2000:.3f}")
print(f"  (uses only 5 cheap features incl. rrf; ensemble_merge.py's full 3-model "
      f"10-feature blend, incl. a real cross-encoder pass, gets 0.82 -- this is a "
      f"lightweight sanity check of the hybrid idea, not an apples-to-apples rerun)")

# ------------------------------------------------------------------ final RRF-based submission set
# Per task: SAVE THE RRF-BASED SET (approx-rank variant, the primary RRF result),
# not the hybrid. Exclude the 100 known matches (exact-tuple) LAST, top up from
# ranked leftovers to stay close to 2000.
final = []
seen = set()
for p in ranked_approx:
    if p in known or p in seen:
        continue
    seen.add(p)
    final.append(p)
    if len(final) >= FINAL_CAP:
        break
final_set = set(final)
assert final_set.isdisjoint(known), "known match leaked into final RRF set"

validate_candidate_set(final_set, tableA_ids, tableB_ids)
with open(f"{BASE}/candidates_rrf.pkl", "wb") as f:
    pickle.dump(final_set, f)

print("\n================ FINAL (RRF) ================")
print(f"final pair count: {len(final_set)}  (validation PASSED, 100 known matches excluded)")
print(f"RRF (approx-rank) top-2000 proxy recall (before exclusion): "
      f"{proxy_recall(ranked_approx, 2000):.3f}")
print(f"saved -> {BASE}/candidates_rrf.pkl")
print(f"\ntotal runtime: {time.time() - t0:.1f}s")
