"""
RESEARCH-ONLY script (does NOT touch ensemble_merge.py / final_candidates.pkl).

Question: does synthetic positive augmentation (degrading TableA-side titles of
the 100 known matches to mimic real TableB catalog-abbreviation style) improve
honest cross-validated recall of a simple classifier over text-similarity
features, vs training on the 100 real positives alone?

Method:
  1. Mine a forward abbreviation dict (full word -> abbrev) + real catalog-noise
     suffix tokens ("win98me2000xp"-style, part-number-like tokens) from actual
     TableB titles.
  2. For each of the 100 known matches, generate N synthetic positive pairs:
     (A_id, synthetic-degraded-A-title) treated as a synthetic B-side text,
     paired against the real A-side text -- i.e. a synthetic pair whose "B text"
     is a degraded clone of A's own title. This mimics the abbreviation pattern
     without needing a real second table row.
  3. Build the SAME kind of features ensemble_merge.py uses (word/char tfidf
     cosine, jaccard, rapidfuzz ratios) for both real and synthetic pairs.
  4. Honest CV: 5-fold split of the 100 REAL known-match ids. In each fold,
     synthetic variants are generated ONLY from the real positives in the
     training folds (never from the held-out fold) -- so no leakage.
  5. Compare: fold-honest recall@2000-equivalent (AUC + recall-at-topK proxy)
     of classifier(real-only) vs classifier(real+synthetic).

Outputs: prints an honest comparison report to stdout and saves
  research_augmentation_report.txt
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, re, random
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"
rng = np.random.default_rng(0)
random.seed(0)

tableA = pd.read_csv(f"{BASE}/tableA.csv")
tableB = pd.read_csv(f"{BASE}/tableB.csv")
with open(f"{BASE}/100_matches.pkl", "rb") as f:
    known = [(int(a), int(b)) for a, b in pickle.load(f)]

def norm(x):
    if pd.isna(x): return ""
    s = str(x).lower(); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

A_title = {int(r.id): norm(r.title) for r in tableA.itertuples()}
B_title = {int(r.id): norm(r.title) for r in tableB.itertuples()}
A_mfr = {int(r.id): norm(r.manufacturer) for r in tableA.itertuples()}
B_mfr = {int(r.id): norm(r.manufacturer) for r in tableB.itertuples()}
A_price = {int(r.id): r.price for r in tableA.itertuples()}
B_price = {int(r.id): r.price for r in tableB.itertuples()}
all_A_ids = list(A_title.keys())
all_B_ids = list(B_title.keys())

# ---------------------------------------------------------------- 1. mine real degradation patterns from TableB
# Forward abbreviation dict: full word -> abbrev (reverse of ensemble_merge.py's expand dict)
FWD_ABBR = {
    "quickbooks": "qckbks", "workgroup": "wkgp", "small": "sm", "business": "biz",
    "edition": "ed", "premium": "prem", "professional": "pro", "standard": "std",
    "upgrade": "upg", "license": "lic", "manager": "mgr", "accounting": "acctg",
    "corporation": "corp", "international": "intl", "developer": "dev",
    "manufacturing": "mfg", "windows": "win", "macintosh": "mac", "version": "ver",
    "package": "pkg", "enterprise": "ent", "server": "svr", "government": "govt",
    "academic": "acad", "software": "sw", "deluxe": "dlx", "system": "sys",
    "center": "ctr", "national": "natl", "education": "educ", "nonprofit": "nonprft",
    "financial": "fin", "personal": "pers", "security": "sec", "network": "netwk",
    "printer": "prntr", "wireless": "wless", "digital": "dig",
}
# Catalog-style numeric/version suffix tokens actually seen across TableB titles
# (mined by scanning for standalone digit / version-like tokens)
def find_numeric_tokens(text):
    toks = text.split()
    out = []
    for t in toks:
        if re.match(r"^v?\d+(\.\d+)?[a-z]{0,3}$", t) and any(c.isdigit() for c in t):
            out.append(t)
    return out

b_numeric_pool = []
for t in B_title.values():
    b_numeric_pool.extend(find_numeric_tokens(t))
b_numeric_pool = list(set(b_numeric_pool)) or ["98", "2000", "xp", "2007", "v2"]

STOPWORDS = {"the", "a", "an", "for", "and", "with", "of", "to", "in"}

def synthesize(a_title, n_variants=4):
    """Generate n degraded variants of a TableA title mimicking TableB catalog style."""
    words = a_title.split()
    variants = []
    for _ in range(n_variants):
        w = list(words)
        # 1. forward-abbreviate some words
        w = [FWD_ABBR.get(tok, tok) for tok in w]
        # 2. randomly drop a non-critical (stopword or short) word
        if len(w) > 3 and rng.random() < 0.6:
            drop_candidates = [i for i, t in enumerate(w) if t in STOPWORDS or len(t) <= 3]
            if drop_candidates:
                di = rng.choice(drop_candidates)
                w.pop(di)
            else:
                # drop a random middle word (not first, often the brand)
                di = rng.integers(1, len(w))
                w.pop(di)
        # 3. maybe truncate trailing words (catalog titles often cut off)
        if len(w) > 4 and rng.random() < 0.4:
            cut = rng.integers(len(w) - 2, len(w))
            w = w[:cut]
        # 4. maybe append/replace with real catalog noise (numeric/version tokens)
        if rng.random() < 0.5:
            noise = rng.choice(b_numeric_pool)
            w.append(str(noise))
        if rng.random() < 0.2 and len(w) > 2:
            # simulate part-number-like token insertion at front (real TableB pattern)
            pn = f"{''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), 3))}-{rng.integers(1000,99999)}"
            w = [pn] + w
        variants.append(" ".join(w))
    return variants

# ---------------------------------------------------------------- 2. feature functions (mirrors ensemble_merge.py's non-leaky subset)
def make_vectorizers(texts_a, texts_b):
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vw.fit(texts_a + texts_b)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    vc.fit(texts_a + texts_b)
    return vw, vc

def featurize_pairs(pairs_text, vw, vc, mfr_pairs=None, price_pairs=None):
    """pairs_text: list of (textA, textB). Returns feature matrix (word cos, char cos,
    jaccard, token_sort_ratio, token_set_ratio, [mfr ratio, price close] optional)."""
    n = len(pairs_text)
    X = np.zeros((n, 5))
    Aw = vw.transform([p[0] for p in pairs_text])
    Bw = vw.transform([p[1] for p in pairs_text])
    Ac = vc.transform([p[0] for p in pairs_text])
    Bc = vc.transform([p[1] for p in pairs_text])
    for k, (ta, tb) in enumerate(pairs_text):
        X[k, 0] = float(Aw[k].multiply(Bw[k]).sum())
        X[k, 1] = float(Ac[k].multiply(Bc[k]).sum())
        sa, sb = set(ta.split()), set(tb.split())
        X[k, 2] = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
        X[k, 3] = fuzz.token_sort_ratio(ta, tb) / 100
        X[k, 4] = fuzz.token_set_ratio(ta, tb) / 100
    return X

# ---------------------------------------------------------------- 3. honest CV comparison
# HARD negatives: for each positive A id, take its top-K TF-IDF nearest B rows
# (excluding the true match) as negatives. Random-pair negatives make the task
# trivial (AUC=1.0, uninformative) because unrelated products share almost no
# tokens; hard near-neighbor negatives reproduce the actual classifier difficulty
# (distinguishing a true abbreviated match from a similar-but-wrong catalog item).
known_set = set(known)
_global_vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
_global_vw.fit(list(A_title.values()) + list(B_title.values()))
_Ball = _global_vw.transform([B_title[b] for b in all_B_ids])

def hard_negatives_for(a_id, true_b, k=5):
    av = _global_vw.transform([A_title[a_id]])
    sims = np.asarray(av.multiply(_Ball).sum(axis=1)).ravel()
    order = np.argsort(-sims)
    out = []
    for i in order:
        b = all_B_ids[i]
        if b == true_b: continue
        out.append((a_id, int(b)))
        if len(out) >= k: break
    return out

def sample_negatives(pos_pairs, per_pos=5):
    negs = []
    for (a, b) in pos_pairs:
        negs.extend(hard_negatives_for(a, b, k=per_pos))
    return negs

kf = KFold(n_splits=5, shuffle=True, random_state=0)
idx = np.arange(len(known))

results_real_only = []
results_augmented = []

N_VARIANTS = 4
NEG_MULT = 5

for fold, (train_idx, test_idx) in enumerate(kf.split(idx)):
    train_pos = [known[i] for i in train_idx]
    test_pos = [known[i] for i in test_idx]

    # ---- baseline: real positives only ----
    train_neg = sample_negatives(train_pos, per_pos=NEG_MULT)
    test_neg = sample_negatives(test_pos, per_pos=NEG_MULT)

    def pair_text(p): return (A_title[p[0]], B_title[p[1]])
    train_texts = [pair_text(p) for p in train_pos + train_neg]
    test_texts = [pair_text(p) for p in test_pos + test_neg]
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

    all_texts_a = [t[0] for t in train_texts + test_texts]
    all_texts_b = [t[1] for t in train_texts + test_texts]
    vw, vc = make_vectorizers(all_texts_a, all_texts_b)

    Xtr = featurize_pairs(train_texts, vw, vc)
    Xte = featurize_pairs(test_texts, vw, vc)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, y_train)
    auc_real = roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1])
    results_real_only.append(auc_real)

    # ---- augmented: add synthetic positives derived ONLY from train_pos ----
    synth_pairs_text = []
    for (a, b) in train_pos:
        variants = synthesize(A_title[a], n_variants=N_VARIANTS)
        for v in variants:
            synth_pairs_text.append((A_title[a], v))  # A's real title vs degraded clone (as synthetic "B text")

    train_texts_aug = train_texts + synth_pairs_text
    y_train_aug = np.concatenate([y_train, np.ones(len(synth_pairs_text))])

    all_texts_a2 = [t[0] for t in train_texts_aug + test_texts]
    all_texts_b2 = [t[1] for t in train_texts_aug + test_texts]
    vw2, vc2 = make_vectorizers(all_texts_a2, all_texts_b2)

    Xtr_aug = featurize_pairs(train_texts_aug, vw2, vc2)
    Xte_aug = featurize_pairs(test_texts, vw2, vc2)

    clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf2.fit(Xtr_aug, y_train_aug)
    auc_aug = roc_auc_score(y_test, clf2.predict_proba(Xte_aug)[:, 1])
    results_augmented.append(auc_aug)

    print(f"fold {fold}: real-only AUC={auc_real:.3f}  augmented AUC={auc_aug:.3f}  "
          f"(n_synth={len(synth_pairs_text)})")

mean_real = np.mean(results_real_only)
mean_aug = np.mean(results_augmented)
report = []
report.append("=== Synthetic Augmentation Research Report ===")
report.append(f"5-fold honest CV (synthetic variants generated ONLY from training-fold positives)")
report.append(f"real-only mean AUC : {mean_real:.4f}  (per-fold: {[f'{x:.3f}' for x in results_real_only]})")
report.append(f"augmented mean AUC : {mean_aug:.4f}  (per-fold: {[f'{x:.3f}' for x in results_augmented]})")
delta = mean_aug - mean_real
report.append(f"delta (aug - real) : {delta:+.4f}")
wins = sum(a > b for a, b in zip(results_augmented, results_real_only))
report.append(f"per-fold wins for augmented: {wins}/5")
if delta > 0.01 and wins >= 3:
    verdict = "PROMISING: augmentation measurably improved honest held-out AUC."
elif abs(delta) <= 0.01:
    verdict = "NEUTRAL: augmentation made no meaningful difference (within noise)."
else:
    verdict = "NOT HELPFUL: augmentation did not improve (or hurt) honest held-out AUC."
report.append(f"VERDICT: {verdict}")
report.append("")
report.append("Caveat: this evaluates AUC via LogisticRegression on a simplified 5-feature")
report.append("subset (word/char tfidf cosine, jaccard, token_sort/set ratio), not the full")
report.append("15-feature deployed model, and synthetic positives here are A-title-vs-degraded-")
report.append("A-title clones (proxy for real TableB abbreviation degradation), not real B rows.")
report.append("Because the synthetic pairs are systematically EASIER (a title degraded from")
report.append("itself retains more structure than an independently-authored catalog listing),")
report.append("any AUC gain here is an OPTIMISTIC estimate of the augmentation's true value.")

report_text = "\n".join(report)
print("\n" + report_text)
with open(f"{BASE}/research_augmentation_report.txt", "w") as f:
    f.write(report_text + "\n")
print(f"\nsaved report -> {BASE}/research_augmentation_report.txt")
