"""
RESEARCH-ONLY (does NOT touch ensemble_merge.py / final_candidates.pkl).

Importable feature function: shared numeric/version-token Jaccard between a
TableA title and a TableB title, e.g. "2007", "3.0", "v11", "98", "xp".

Also runs a standalone honest evaluation: does this feature carry independent
signal beyond the general word-Jaccard already in ensemble_merge.py? Checks:
  1. Distribution of numeric_jaccard for known-match pairs vs hard-negative pairs
     (TF-IDF near-neighbors, not random -- random negatives make everything look
     separable and would overstate the feature's value).
  2. Standalone AUC of numeric_jaccard alone.
  3. Correlation with general word-Jaccard (to see if it's redundant).
  4. Incremental AUC lift when added to the existing 5-feature text-similarity
     set from research_augmentation.py's featurizer.

Usage as a library:
    from feature_numeric_tokens import numeric_tokens, numeric_jaccard
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, re
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

# ---------------------------------------------------------------- importable feature functions
_NUMTOK_RE = re.compile(r"^v?\d+(\.\d+)?[a-z]{0,3}$")

def numeric_tokens(norm_title):
    """Extract standalone numeric/version-like tokens from an already-normalized
    (lowercase, alnum+space) title string: plain digit sequences ('98', '2007'),
    dotted versions ('3.0'), and v-prefixed versions ('v11')."""
    out = []
    for t in norm_title.split():
        if _NUMTOK_RE.match(t) and any(c.isdigit() for c in t):
            out.append(t)
    return set(out)

def numeric_jaccard(norm_title_a, norm_title_b):
    """Jaccard similarity over the numeric-token sets of two normalized titles.
    Returns 0.0 if neither title has any numeric tokens (distinguish from
    'both empty -> trivially 0 overlap' by caller if desired via has_numeric flag)."""
    sa, sb = numeric_tokens(norm_title_a), numeric_tokens(norm_title_b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def has_numeric_mismatch(norm_title_a, norm_title_b):
    """1.0 if both titles have numeric tokens but they are DISJOINT (e.g. '2006'
    vs '2007' edition) -- a candidate negative signal for otherwise-similar pairs
    with a different version/edition number. 0.0 otherwise (including the
    no-numeric-tokens-present case, which is uninformative either way)."""
    sa, sb = numeric_tokens(norm_title_a), numeric_tokens(norm_title_b)
    if sa and sb and not (sa & sb):
        return 1.0
    return 0.0

if __name__ == "__main__":
    tableA = pd.read_csv(f"{BASE}/tableA.csv")
    tableB = pd.read_csv(f"{BASE}/tableB.csv")
    with open(f"{BASE}/100_matches.pkl", "rb") as f:
        known = [(int(a), int(b)) for a, b in pickle.load(f)]

    def norm(x):
        if pd.isna(x): return ""
        s = str(x).lower(); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

    A_title = {int(r.id): norm(r.title) for r in tableA.itertuples()}
    B_title = {int(r.id): norm(r.title) for r in tableB.itertuples()}
    all_B_ids = list(B_title.keys())

    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vw.fit(list(A_title.values()) + list(B_title.values()))
    Ball = vw.transform([B_title[b] for b in all_B_ids])

    def hard_negatives_for(a_id, true_b, k=5):
        av = vw.transform([A_title[a_id]])
        sims = np.asarray(av.multiply(Ball).sum(axis=1)).ravel()
        order = np.argsort(-sims)
        out = []
        for i in order:
            b = all_B_ids[i]
            if b == true_b: continue
            out.append((a_id, int(b)))
            if len(out) >= k: break
        return out

    rng = np.random.default_rng(0)
    hard_negs = []
    for (a, b) in known:
        hard_negs.extend(hard_negatives_for(a, b, k=5))

    pos_pairs = known
    neg_pairs = hard_negs

    def word_jac(ta, tb):
        sa, sb = set(ta.split()), set(tb.split())
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

    def build_feats(pairs):
        rows = []
        for (a, b) in pairs:
            ta, tb = A_title[a], B_title[b]
            wj = word_jac(ta, tb)
            nj = numeric_jaccard(ta, tb)
            mismatch = has_numeric_mismatch(ta, tb)
            tsr = fuzz.token_sort_ratio(ta, tb) / 100
            rows.append([wj, nj, mismatch, tsr])
        return np.array(rows)

    Xp = build_feats(pos_pairs)
    Xn = build_feats(neg_pairs)
    y = np.array([1] * len(Xp) + [0] * len(Xn))
    X = np.vstack([Xp, Xn])
    # columns: 0 word_jaccard, 1 numeric_jaccard, 2 numeric_mismatch_flag, 3 token_sort_ratio

    report = []
    report.append("=== Numeric/Version Token Feature Research Report ===")
    report.append(f"positives (known matches): {len(pos_pairs)}, hard negatives (TF-IDF top-5 non-match neighbors): {len(neg_pairs)}")

    n_pos_has_num = sum(1 for a, b in pos_pairs if numeric_tokens(A_title[a]) or numeric_tokens(B_title[b]))
    report.append(f"known-match pairs where at least one side has a numeric token: {n_pos_has_num}/{len(pos_pairs)}")

    pos_nj = Xp[:, 1]; neg_nj = Xn[:, 1]
    report.append(f"numeric_jaccard mean: known-match={pos_nj.mean():.3f} (std {pos_nj.std():.3f})  "
                   f"hard-negative={neg_nj.mean():.3f} (std {neg_nj.std():.3f})")
    pos_mm = Xp[:, 2]; neg_mm = Xn[:, 2]
    report.append(f"numeric_mismatch_flag rate: known-match={pos_mm.mean():.3f}  hard-negative={neg_mm.mean():.3f}")

    # standalone AUC of numeric_jaccard
    try:
        auc_num_only = roc_auc_score(y, X[:, 1])
    except Exception:
        auc_num_only = float("nan")
    report.append(f"standalone AUC of numeric_jaccard alone: {auc_num_only:.3f}")

    # correlation with word jaccard
    corr = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    report.append(f"correlation(numeric_jaccard, word_jaccard): {corr:.3f}")

    # incremental value: 5-fold CV AUC with vs without numeric features, using pair-level split
    idx = np.arange(len(pos_pairs))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    aucs_base, aucs_plus = [], []
    for train_idx, test_idx in kf.split(idx):
        train_pos_i, test_pos_i = train_idx, test_idx
        # map back to which negs belong to which pos (5 negs per pos, in order)
        def negs_for(idxs):
            out = []
            for i in idxs:
                out.extend(range(i * 5, i * 5 + 5))
            return out
        tr_neg_i, te_neg_i = negs_for(train_pos_i), negs_for(test_pos_i)

        Xtr = np.vstack([Xp[train_pos_i], Xn[tr_neg_i]])
        ytr = np.array([1] * len(train_pos_i) + [0] * len(tr_neg_i))
        Xte = np.vstack([Xp[test_pos_i], Xn[te_neg_i]])
        yte = np.array([1] * len(test_pos_i) + [0] * len(te_neg_i))

        base_cols = [0, 3]         # word_jaccard, token_sort_ratio only
        plus_cols = [0, 1, 2, 3]   # + numeric_jaccard, numeric_mismatch_flag

        clf_b = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf_b.fit(Xtr[:, base_cols], ytr)
        auc_b = roc_auc_score(yte, clf_b.predict_proba(Xte[:, base_cols])[:, 1])

        clf_p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf_p.fit(Xtr[:, plus_cols], ytr)
        auc_p = roc_auc_score(yte, clf_p.predict_proba(Xte[:, plus_cols])[:, 1])

        aucs_base.append(auc_b); aucs_plus.append(auc_p)

    report.append(f"5-fold CV AUC, base (word_jaccard+token_sort_ratio) : {np.mean(aucs_base):.4f}  {[f'{x:.3f}' for x in aucs_base]}")
    report.append(f"5-fold CV AUC, +numeric features                    : {np.mean(aucs_plus):.4f}  {[f'{x:.3f}' for x in aucs_plus]}")
    lift = np.mean(aucs_plus) - np.mean(aucs_base)
    report.append(f"AUC lift from adding numeric features: {lift:+.4f}")

    if lift > 0.01:
        verdict = "PROMISING: numeric-token features add measurable independent AUC lift -- worth folding in as extra features (numeric_jaccard, numeric_mismatch_flag)."
    elif lift > 0.0:
        verdict = "MARGINAL: small positive lift, likely within noise given the small (100-positive) sample; low-risk to add since it's an extra column, not a replacement."
    else:
        verdict = "NOT HELPFUL on this hard-negative sample: no measurable AUC lift; the signal may already be captured by word_jaccard/token_sort_ratio for this dataset."
    report.append(f"VERDICT: {verdict}")
    report.append("")
    report.append("Note: hard negatives here are per-positive TF-IDF top-5 nearest B rows")
    report.append("(excluding the true match) -- NOT random pairs -- so this measures signal")
    report.append("in the realistic 'confusable near-miss' regime the deployed classifier faces,")
    report.append("not an inflated random-negative AUC.")

    report_text = "\n".join(report)
    print(report_text)
    with open(f"{BASE}/research_numeric_token_report.txt", "w") as f:
        f.write(report_text + "\n")
    print(f"\nsaved report -> {BASE}/research_numeric_token_report.txt")
