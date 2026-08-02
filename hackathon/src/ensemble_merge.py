"""
Ensemble merger run 8 (deployed) for the entity-matching blocking hackathon.

Pools ALL candidates_*.pkl strategies, engineers 17 non-leaky per-pair
similarity features (word/char TF-IDF cosine, Jaccard, rapidfuzz token ratios,
price closeness+known-flag, a re-derived CONTINUOUS cross-encoder relevance
score on RAW text, a cross-strategy presence count, run-6's two abbreviated-
TableB fixes, run-7's Fable asymmetric-containment feature, PLUS run-8's
CROSS-ENCODER-ON-ABBREVIATION-EXPANDED-TEXT score), trains a 3-model blend
(LogisticRegression + GradientBoosting + small MLP) with HARD-NEGATIVE MINING,
averages 9 negative samples for stability, ranks with a GREEDY 1:1 BIPARTITE
selection, excludes the 100 known matches (exact-tuple, LAST), tops up to the
2000-pair cap, validates, and saves final_candidates.pkl.

Run 8 change (validated with the same rigorous multi-seed honest-OOF methodology
as prior runs; ADDITIVE and confirmed to lift the recall floor without regressing
it, across 10 independent fresh-seed groups):
  * FEATURE: add a SECOND cross-encoder score computed on ABBREVIATION-EXPANDED
    text (both sides run through the run-6 ABBR dictionary before being fed to
    the same already-loaded CrossEncoder) as a 17th feature (index 16). This is
    ADDITIVE alongside the existing raw-text CE score (index 8), not a
    replacement. A prior research pass found expansion lifts CE AUC 0.816->0.826
    overall and +0.026 on the 245 pairs where expansion changes the text (43 of
    which are known matches), with large per-pair score jumps for previously
    near-zero-scored true matches (e.g. A=396/B=2470 raw +0.08 -> expanded +7.10;
    A=604/B=1512 +0.88 -> +7.42). Costs one extra batched CE .predict() over the
    pool (~15s).

Run 8 measured impact (deployed metric: 9-seed averaged honest-OOF blend ->
single greedy 1:1; over 10 independent FRESH-seed groups, seeds disjoint from the
9 deployed seeds to avoid seed-selection bias):
  * run-7 (16 features)        : mean 0.9210 / min 0.920  (reproduces deployed ~0.922)
  * +CE-expanded (17 features) : mean 0.9310 / min 0.930   <- beats run-7 in ALL 10 groups
  * Also tested but NOT adopted: 2-round iterative hard-negative mining (re-mine
    a harder band from a model refit on the round-1 negatives). Net-neutral alone
    (mean 0.9220 vs 0.9210) and it DAMPENED the CE-expanded gain (17f+2r mean
    0.9300 vs 17f+1r 0.9310, losing the 0.940 upside). Kept at a single round.

Run 7 changes (retained): Fable asymmetric IDF-weighted title-containment feature
(index 15) and hard-negative mining (per seed: fit a quick LR+GB on random
negatives, draw 400 "hard" negatives from the 0.70-0.97 quantile band of the
non-match score distribution mixed with 800 random negatives).

Why greedy 1:1 (kept from run 6): the ground truth is essentially 1:1 on BOTH
sides (99 A ids with 1 match + 1 A with 2, and 100 DISTINCT B ids). CNP k=1 only
constrains the A side, wasting ~486 duplicate-B slots. Greedy 1:1 (sort pooled
pairs by score desc; take a pair only if NEITHER its A id NOR its B id is used;
then relax and fill the remaining budget by global score) removes that waste.

Explicitly evaluated but NOT deployed:
  * 2-round iterative hard-negative mining (run 8): see above -- net-negative.
  * fs_prob (honest nested-CV full_supervised probability): net-negative on the
    honest OOF metric; mildly leaky w.r.t. the 100 positives. Excluded.
  * near-tie 2-match exception: no robust gain, fires far too broadly at low
    thresholds and never at conservative ones. Excluded.
  * fine-tuned-embedding continuous cosine, Hungarian global assignment: tied or
    worse in prior runs. Excluded.

Gotchas handled:
  * Pairs are ALWAYS (id_a from TableA, id_b from TableB). Never sorted() -- id
    namespaces overlap, so sorting would flip provenance. Dedup via set add only.
  * Cross-strategy presence count EXCLUDES strategies whose relationship to the
    100 known matches is biased (deepblocker + rrf pre-removed the 100; finetuned
    _embeddings + full_supervised trained ON the 100). Their pairs still populate
    the pool; only the count feature omits them.
  * The 100 known matches are excluded from the final output (exact-tuple), done
    LAST, and freed slots are topped up from the ranked leftover pool.

Honesty note: model/config choices were made on out-of-fold (OOF) proxy recall,
never on full-fit recall. The DEPLOYED ranking is full-fit (uses all 100 labels),
averaged over 9 negative samples for stability; the reported ~0.931 is the honest
OOF estimate from the same pipeline.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, re, glob, os, math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sentence_transformers import CrossEncoder
from helpers import validate_candidate_set

BASE = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"

# strategies whose presence is a BIASED signal for the 100 knowns (see docstring)
EXCLUDE_FROM_COUNT = {"deepblocker", "rrf", "finetuned_embeddings", "full_supervised"}
N_NEG_MULT = 8     # negatives per positive (total training-negative budget)
N_REPS = 9         # average over this many negative samples for a stable ranking
# run-7 hard-negative mining: of the N_NEG_MULT*|pos| negatives per seed, draw
# N_HARD from the confusable score band and the remainder at random.
N_RANDOM = 800
N_HARD = 400
HARD_LO, HARD_HI = 0.70, 0.97   # quantile band of the non-match score distribution

# ------------------------------------------------------------------ load
tableA = pd.read_csv(f"{BASE}/tableA.csv"); tableB = pd.read_csv(f"{BASE}/tableB.csv")
tableA_ids, tableB_ids = set(tableA["id"]), set(tableB["id"])
with open(f"{BASE}/100_matches.pkl", "rb") as f:
    known = {(int(a), int(b)) for a, b in pickle.load(f)}

strat_sets = {}
for fn in sorted(glob.glob(f"{BASE}/candidates_*.pkl")):
    if "report" in fn:
        continue
    name = os.path.basename(fn)[len("candidates_"):-4]
    obj = pickle.load(open(fn, "rb"))
    if not isinstance(obj, (set, list, tuple)):
        continue
    strat_sets[name] = {(int(a), int(b)) for a, b in obj}

print("--- per-strategy proxy recall vs 100 known matches ---")
for s in sorted(strat_sets):
    tag = ""
    if s in ("deepblocker", "rrf"):
        tag = "  (100 pre-removed -> understated)"
    elif s in ("finetuned_embeddings", "full_supervised"):
        tag = "  (trained on the 100 -> optimistic)"
    print(f"  {s:22s} n={len(strat_sets[s]):6d} recall={len(strat_sets[s]&known)/100:.3f}{tag}")
union_all = set().union(*strat_sets.values())
print(f"  UNION of all           n={len(union_all):6d} recall={len(union_all&known)/100:.3f}  <- coverage ceiling")

COUNT_STRATS = [s for s in strat_sets if s not in EXCLUDE_FROM_COUNT]
# Deterministic pool order: sort the ALREADY-CANONICAL (A-id, B-id) tuples so that
# greedy tie-breaks (and thus the reported honest number) are reproducible run to
# run. This sorts the LIST of pairs for iteration only -- it never reorders ids
# WITHIN a tuple, so the (TableA-id, TableB-id) provenance gotcha does not apply.
pool = sorted(union_all | known)  # include positives so they participate in feature/training
pool_index = {p: i for i, p in enumerate(pool)}
print(f"pool size (incl 100 matches): {len(pool)}")

# ------------------------------------------------------------------ text prep
def norm(x):
    if pd.isna(x): return ""
    s = str(x).lower(); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

A_title = {int(r.id): norm(r.title) for r in tableA.itertuples()}
B_title = {int(r.id): norm(r.title) for r in tableB.itertuples()}
A_mfr = {int(r.id): norm(r.manufacturer) for r in tableA.itertuples()}
B_mfr = {int(r.id): norm(r.manufacturer) for r in tableB.itertuples()}
B_mfr_raw = {int(r.id): r.manufacturer for r in tableB.itertuples()}
A_price = {int(r.id): r.price for r in tableA.itertuples()}
B_price = {int(r.id): r.price for r in tableB.itertuples()}

# ---- FIX 1 (run 6): infer B manufacturer from title prefix when null ----
_partre = re.compile(r"^[a-z]*\d+[a-z\d-]*$")
def _looks_part(t):
    if not t: return False
    if _partre.match(t): return True
    d = sum(c.isdigit() for c in t)
    return d > 0 and d >= len(t) * 0.4
_STOP_BRAND = {"the", "a", "for", "and", "with", "new", "re"}
def infer_bmfr(bid):
    if not pd.isna(B_mfr_raw[bid]):
        return B_mfr[bid]
    toks = B_title[bid].split()
    if not toks: return ""
    i = 1 if _looks_part(toks[0]) else 0   # skip a leading part number if present
    brand = []
    for t in toks[i:i + 3]:
        if _looks_part(t): break
        if t in _STOP_BRAND and not brand: continue
        brand.append(t)
    return " ".join(brand)
B_mfr_inf = {bid: infer_bmfr(bid) for bid in B_title}

# ---- FIX 2 (run 6): domain abbreviation expansion on BOTH titles ----
ABBR = {
    "qckbks": "quickbooks", "qb": "quickbooks", "wkgp": "workgroup", "wkgrp": "workgroup",
    "sbe": "small business edition", "sb": "small business", "prem": "premium",
    "prof": "professional", "pro": "professional", "std": "standard", "ed": "edition",
    "edt": "edition", "upg": "upgrade", "upgrd": "upgrade", "lic": "license", "lics": "license",
    "mgr": "manager", "acct": "accounting", "acctg": "accounting", "corp": "corporation",
    "intl": "international", "dev": "developer", "biz": "business", "mfg": "manufacturing",
    "win": "windows", "mac": "macintosh", "ver": "version", "pkg": "package",
    "ent": "enterprise", "svr": "server", "govt": "government", "acad": "academic",
    "sw": "software", "dlx": "deluxe", "sys": "system", "ctr": "center",
    "natl": "national", "educ": "education",
}
def expand(t): return " ".join(ABBR.get(w, w) for w in t.split())
A_title_x = {i: expand(t) for i, t in A_title.items()}
B_title_x = {i: expand(t) for i, t in B_title.items()}

# ---- RUN 7 FEATURE: Fable asymmetric IDF-weighted title-containment ----
# Deterministic, non-leaky. Reuses strategy_fable.py's normalization + formula:
# score = max(contain(A_core, B_ext), contain(B_core, A_ext)) + 0.10*jaccard.
_FAB_ABBREV = {
    "upg": "upgrade", "upgrd": "upgrade", "win": "windows", "windows": "windows",
    "mac": "mac", "macintosh": "mac", "ed": "edition", "edn": "edition",
    "std": "standard", "pro": "professional", "prof": "professional", "acad": "academic",
    "ac": "academic", "lic": "license", "lics": "license", "licence": "license",
    "vol": "volume", "ver": "version", "prepaid": "prepaid", "pre-paid": "prepaid",
    "sec": "security", "int": "internet", "intl": "international", "dlx": "deluxe",
    "prem": "premium", "prm": "premium", "qckbks": "quickbooks", "rpt": "report",
    "rpts": "report", "wkgp": "workgroup", "trng": "training", "gr": "grade",
    "beginners": "beginning", "producers": "producer", "&": "and",
}
_FAB_STOP = {"the", "for", "and", "with", "of", "by", "a", "an", "in", "to", "on",
    "software", "1", "user", "users", "complete", "package", "full", "version",
    "retail", "box", "cd", "dvd", "rom", "jewel", "case", "windows", "mac", "s"}
_fab_num_glue = re.compile(r"\bv\s*(\d+)\s*\.\s*(\d+)"); _fab_tok = re.compile(r"[a-z0-9\.]+")
def _fab_normalize(text):
    t = str(text).lower(); t = _fab_num_glue.sub(r"v\1.\2", t); t = t.replace("-", " ")
    toks = _fab_tok.findall(t); out = []
    for tk in toks:
        tk = tk.strip(".")
        if not tk: continue
        tk = _FAB_ABBREV.get(tk, tk)
        if len(tk) > 4 and tk.isalpha() and tk.endswith("s"): tk = tk[:-1]
        out.append(tk)
    glued = [out[i] + out[i + 1] for i in range(len(out) - 1)
             if out[i].isalpha() and out[i + 1].isalpha()]
    return out, glued
def _fab_content(tokens): return {t for t in tokens if t not in _FAB_STOP and len(t) > 1}
def _fab_build(df):
    core, ext = {}, {}
    for r in df.itertuples():
        toks, glued = _fab_normalize(f"{r.title} {r.manufacturer}")
        c = _fab_content(toks); core[int(r.id)] = c; ext[int(r.id)] = c | set(glued)
    return core, ext
A_fcore, A_fext = _fab_build(tableA); B_fcore, B_fext = _fab_build(tableB)
_fab_dfc = defaultdict(int)
for toks in list(A_fcore.values()) + list(B_fcore.values()):
    for t in toks: _fab_dfc[t] += 1
_fab_N = len(A_fcore) + len(B_fcore)
_fab_idf = {t: math.log(_fab_N / (1 + c)) for t, c in _fab_dfc.items()}
def _fab_contain(core_self, ext_other):
    denom = num = 0.0
    for t in core_self:
        w = _fab_idf.get(t, 0.1); denom += w
        if t in ext_other: num += w
    return num / denom if denom > 0 else 0.0
def fable_score(a, b):
    ca, cb = A_fcore[a], B_fcore[b]
    c = max(_fab_contain(ca, B_fext[b]), _fab_contain(cb, A_fext[a]))
    j = len(ca & cb) / len(ca | cb) if (ca | cb) else 0.0
    return c + 0.10 * j

# base (original-title) TF-IDF vectorizers
vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
vw.fit(list(A_title.values()) + list(B_title.values()))
Aw = {i: vw.transform([t]) for i, t in A_title.items()}
Bw = {i: vw.transform([t]) for i, t in B_title.items()}
vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
vc.fit(list(A_title.values()) + list(B_title.values()))
Ac = {i: vc.transform([t]) for i, t in A_title.items()}
Bc = {i: vc.transform([t]) for i, t in B_title.items()}
# expanded-title word TF-IDF vectorizer (Fix 2)
vwx = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
vwx.fit(list(A_title_x.values()) + list(B_title_x.values()))
Awx = {i: vwx.transform([t]) for i, t in A_title_x.items()}
Bwx = {i: vwx.transform([t]) for i, t in B_title_x.items()}

def wcos(a, b): return float(Aw[a].multiply(Bw[b]).sum())
def ccos(a, b): return float(Ac[a].multiply(Bc[b]).sum())
def wcosx(a, b): return float(Awx[a].multiply(Bwx[b]).sum())
def jac(a, b):
    sa, sb = set(A_title[a].split()), set(B_title[b].split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
def jacx(a, b):
    sa, sb = set(A_title_x[a].split()), set(B_title_x[b].split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
def pclose(a, b):
    pa, pb = A_price[a], B_price[b]
    if pd.isna(pa) or pd.isna(pb): return 0.0
    m = max(abs(pa), abs(pb), 1.0); return 1.0 - min(abs(pa - pb) / m, 1.0)
def pknown(a, b):
    return 1.0 if (not pd.isna(A_price[a]) and not pd.isna(B_price[b])) else 0.0

# ------------------------------------------------------------------ cross-encoder (continuous)
# Two passes over the SAME pool with the SAME model object: raw text (feature 8,
# run<=7) and abbreviation-expanded text (feature 16, run 8). Both are continuous
# relevance scores; the expanded pass rescues heavily-abbreviated TableB titles.
print("scoring pool with cross-encoder (raw + expanded, continuous features)...")
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
cta = {i: f"{A_title[i]} {A_mfr[i]}".strip() for i in A_title}
ctb = {i: f"{B_title[i]} {B_mfr[i]}".strip() for i in B_title}
cta_x = {i: f"{A_title_x[i]} {A_mfr[i]}".strip() for i in A_title}
ctb_x = {i: f"{B_title_x[i]} {B_mfr[i]}".strip() for i in B_title}
ce_scores = ce.predict([(cta[a], ctb[b]) for a, b in pool], batch_size=256, show_progress_bar=False)
ce_map = {pool[i]: float(ce_scores[i]) for i in range(len(pool))}
ce_scores_x = ce.predict([(cta_x[a], ctb_x[b]) for a, b in pool], batch_size=256, show_progress_bar=False)
ce_map_x = {pool[i]: float(ce_scores_x[i]) for i in range(len(pool))}

# ------------------------------------------------------------------ features
# 17 features: the original 10 (unexpanded text + raw manufacturer + raw-text CE),
# run-6's two targeted fixes as ADDITIONAL signals (Fix 2 abbrev-expanded text:
# wcos_x, jac_x, title_tset_x; Fix 1 inferred-B manufacturer: mfr_inf_tset,
# mfr_inf_partial), run-7's Fable asymmetric-containment score (fable), and run-8's
# CROSS-ENCODER-ON-EXPANDED-TEXT score (ce_x). All ADDITIVE (originals untouched),
# so the model can only gain information.
FEATS = ["wcos", "ccos", "jac", "title_tsr", "title_tset", "mfr_tsr",
         "pclose", "pknown", "ce", "n_strats",
         "wcos_x", "jac_x", "title_tset_x", "mfr_inf_tset", "mfr_inf_partial",
         "fable", "ce_x"]
def featurize(pairs):
    X = np.zeros((len(pairs), len(FEATS)))
    for k, (a, b) in enumerate(pairs):
        X[k, 0] = wcos(a, b); X[k, 1] = ccos(a, b); X[k, 2] = jac(a, b)
        X[k, 3] = fuzz.token_sort_ratio(A_title[a], B_title[b]) / 100
        X[k, 4] = fuzz.token_set_ratio(A_title[a], B_title[b]) / 100
        X[k, 5] = fuzz.token_sort_ratio(A_mfr[a], B_mfr[b]) / 100
        X[k, 6] = pclose(a, b); X[k, 7] = pknown(a, b)
        X[k, 8] = ce_map[(a, b)]
        X[k, 9] = sum((a, b) in strat_sets[s] for s in COUNT_STRATS)
        # Fix 2: abbreviation-expanded text similarity
        X[k, 10] = wcosx(a, b); X[k, 11] = jacx(a, b)
        X[k, 12] = fuzz.token_set_ratio(A_title_x[a], B_title_x[b]) / 100
        # Fix 1: inferred-B-manufacturer match vs A manufacturer
        X[k, 13] = fuzz.token_set_ratio(A_mfr[a], B_mfr_inf[b]) / 100
        X[k, 14] = fuzz.partial_ratio(A_mfr[a], B_mfr_inf[b]) / 100
        # Run 7: Fable asymmetric IDF-weighted containment
        X[k, 15] = fable_score(a, b)
        # Run 8: cross-encoder on abbreviation-expanded text
        X[k, 16] = ce_map_x[(a, b)]
    return X

print("featurizing pool...")
X_pool = featurize(pool)
# base 15-feature view used only by the hard-negative SELECTOR (matches the
# feature set the mining was validated with; the final models use all 17). The
# run-8 ce_x feature enters only the FINAL blend, keeping the mining selector
# identical to the validated run-7 configuration.
X_sel = X_pool[:, :15]

# ------------------------------------------------------------------ models
def make_models():
    return {
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gb": GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=1500, random_state=0)),
    }

pos = [p for p in known if p in pool_index]
nm = [p for p in pool if p not in known]
nm_idx = np.array([pool_index[p] for p in nm])
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# ------------------------------------------------------------------ hard-negative mining
def sample_negatives(seed):
    """Run-7 hard-negative mining (single round; run-8 confirmed a second round
    does not help): fit a quick LR+GB on a random-negative sample (base 15
    features), score the non-match pool, then draw N_HARD 'hard' negatives from
    the [HARD_LO, HARD_HI] quantile band of the non-match score distribution
    (confusable near-misses) mixed with N_RANDOM random negatives. Hard negatives
    are genuine non-matches (not in `known`), so labelling them 0 is correct; this
    only sharpens the boundary and never leaks positive labels."""
    rng = np.random.default_rng(seed)
    r0 = [nm[i] for i in rng.choice(len(nm), size=N_NEG_MULT * len(pos), replace=False)]
    train0 = pos + r0
    y0 = np.array([1] * len(pos) + [0] * len(r0))
    Xtr0 = np.array([X_sel[pool_index[p]] for p in train0])
    sc = np.zeros(len(pool))
    for m in (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
              GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)):
        mm = clone(m); mm.fit(Xtr0, y0); sc += mm.predict_proba(X_sel)[:, 1]
    sc /= 2.0
    nm_sc = sc[nm_idx]
    qlo, qhi = np.quantile(nm_sc, HARD_LO), np.quantile(nm_sc, HARD_HI)
    band = [nm[i] for i in range(len(nm)) if qlo <= nm_sc[i] <= qhi]
    rng2 = np.random.default_rng(seed + 1000)
    if len(band) > N_HARD:
        band = [band[i] for i in rng2.choice(len(band), size=N_HARD, replace=False)]
    rand = [nm[i] for i in rng2.choice(len(nm), size=N_RANDOM, replace=False)]
    return list(dict.fromkeys(rand + band))   # dedup, preserve order

# ------------------------------------------------------------------ CV AUC (sanity, one negative sample)
neg0 = sample_negatives(42)
train0 = pos + neg0
y0 = np.array([1] * len(pos) + [0] * len(neg0))
Xtr0 = np.array([X_pool[pool_index[p]] for p in train0])
print(f"\nensemble training: {len(pos)} positives, {len(neg0)} negatives per rep (hard-neg mined)")
print("--- 5-fold CV ROC-AUC (one negative sample) ---")
for name, m in make_models().items():
    auc = cross_val_score(m, Xtr0, y0, cv=cv, scoring="roc_auc")
    print(f"  {name:4s} AUC = {auc.mean():.3f} +/- {auc.std():.3f}")

# ------------------------------------------------------------------ honest OOF + deploy scores (avg over N_REPS)
def blend_scores(seed, honest):
    neg = sample_negatives(seed)
    train = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    Xtr = np.array([X_pool[pool_index[p]] for p in train])
    blend = np.zeros(len(pool))
    for name, m in make_models().items():
        if honest:
            oof = np.zeros(len(train))
            for tr, te in cv.split(Xtr, y):
                mm = clone(m); mm.fit(Xtr[tr], y[tr]); oof[te] = mm.predict_proba(Xtr[te])[:, 1]
            full = clone(m); full.fit(Xtr, y)
            sc = full.predict_proba(X_pool)[:, 1].copy()
            for j, p in enumerate(train): sc[pool_index[p]] = oof[j]  # honest override for train pairs
        else:
            mm = clone(m); mm.fit(Xtr, y); sc = mm.predict_proba(X_pool)[:, 1]
        blend += sc
    return blend / 3.0

print(f"\ncomputing honest-OOF and deploy blends (avg over {N_REPS} hard-neg-mined samples)...")
seeds = [1, 7, 13, 21, 42, 3, 55, 99, 123][:N_REPS]
honest_blend = np.mean([blend_scores(s, honest=True) for s in seeds], axis=0)
deploy_blend = np.mean([blend_scores(s, honest=False) for s in seeds], axis=0)

# ------------------------------------------------------------------ selection: greedy 1:1 bipartite
def rank_greedy_bipartite(scores):
    """Sort pooled pairs by score desc; take a pair only if NEITHER its TableA id
    NOR its TableB id has been used yet (enforces the ~1:1 ground-truth structure
    on both sides, no wasted duplicate-B budget). Then relax and append the
    leftover pairs in score order so downstream exclude-100 + top-up still fill
    the full 2000. Returns the FULL ranked pool (chosen 1:1 first, then leftover)."""
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    usedA, usedB = set(), set(); chosen = []; leftover = []
    for i in order:
        a, b = pool[i]
        if a not in usedA and b not in usedB:
            usedA.add(a); usedB.add(b); chosen.append(pool[i])
        else:
            leftover.append(pool[i])
    return chosen + leftover

# CNP kept only as a diagnostic reference point in the report below
def rank_cnp(scores, k):
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    per = defaultdict(int); chosen = []; left = []
    for i in order:
        p = pool[i]
        if per[p[0]] < k: per[p[0]] += 1; chosen.append(p)
        else: left.append(p)
    return chosen + left
def proxy_recall(ranked): return len(set(ranked[:2000]) & known) / 100

global_honest = len(set([pool[i] for i in np.argsort(-honest_blend)][:2000]) & known) / 100
print("\n--- HONEST (out-of-fold) top-2000 proxy recall ---")
print(f"  global (no constraint) : {global_honest:.3f}")
print(f"  CNP k=1 (run 4)        : {proxy_recall(rank_cnp(honest_blend, 1)):.3f}")
print(f"  CNP k=2                : {proxy_recall(rank_cnp(honest_blend, 2)):.3f}")
honest_recall = proxy_recall(rank_greedy_bipartite(honest_blend))
print(f"  DEPLOYED (greedy 1:1)  : {honest_recall:.3f}   <- trusted estimate   (run 7: ~0.922)")

deploy_ranked = rank_greedy_bipartite(deploy_blend)
print("\ntop-2000 (deploy) pairs by contributing strategy:")
top2000 = set(deploy_ranked[:2000])
for s in sorted(strat_sets):
    print(f"  {s:22s} {len(top2000 & strat_sets[s])}")

# ------------------------------------------------------------------ exclude 100 + top-up + validate
final = []
seen = set()
for p in deploy_ranked:
    if p in known or p in seen: continue
    seen.add(p); final.append(p)
    if len(final) >= 2000: break
final_set = set(final)
assert final_set.isdisjoint(known), "known match leaked into final set"
validate_candidate_set(final_set, tableA_ids, tableB_ids)

with open(f"{BASE}/final_candidates.pkl", "wb") as f:
    pickle.dump(final_set, f)

print("\n================ FINAL ================")
print(f"union coverage ceiling            : {len(union_all&known)/100:.3f}")
print(f"honest OOF top-2000 proxy recall  : {honest_recall:.3f}   (run 7: ~0.922)")
print(f"final pair count                  : {len(final_set)}   (validation PASSED)")
print(f"saved -> {BASE}/final_candidates.pkl")
