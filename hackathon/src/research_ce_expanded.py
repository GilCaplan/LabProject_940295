"""
RESEARCH-ONLY (does NOT touch ensemble_merge.py / final_candidates.pkl).

Question: the deployed cross-encoder feature (cross-encoder/ms-marco-MiniLM-L-6-v2)
scores RAW normalized "title manufacturer" text for both TableA/TableB. Since then,
an abbreviation-expansion dictionary was added to ensemble_merge.py to fix the
diagnosed failure mode (heavily-abbreviated TableB titles, e.g. "qckbks prem" ->
"quickbooks premium"). Does re-scoring the SAME pairs with the cross-encoder fed
EXPANDED text (both sides run through the abbreviation dict first) correlate better
with true-match labels than the raw-text CE score -- specifically for the subset of
pairs where expansion actually changes the text (the previously-diagnosed hard cases)?

Method:
  1. Rebuild the pool (union of all candidates_*.pkl + the 100 known matches),
     exactly as ensemble_merge.py does (read-only reproduction).
  2. Identify the "expansion-fired" subset: pairs where abbreviation expansion
     changes A's title text or B's title text (almost always B, since B is the
     heavily-abbreviated side).
  3. Score ALL pool pairs with the cross-encoder twice: once on raw normalized
     "title manufacturer" text (matches deployed feature), once on abbreviation-
     expanded text.
  4. For validation, build a labeled set: the 100 known matches (positives) plus a
     sampled set of hard negatives (TF-IDF near-neighbors of each known match's A
     row, restricted to the pool) as negatives. Compare AUC of raw CE score vs
     expanded CE score (a) over the FULL labeled set and (b) restricted to the
     expansion-fired subset only.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, re, glob, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sentence_transformers import CrossEncoder

BASE = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"

# ------------------------------------------------------------------ load (mirrors ensemble_merge.py)
tableA = pd.read_csv(f"{BASE}/tableA.csv"); tableB = pd.read_csv(f"{BASE}/tableB.csv")
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
union_all = set().union(*strat_sets.values())
pool = sorted(union_all | known)
print(f"pool size (incl 100 matches): {len(pool)}")

def norm(x):
    if pd.isna(x): return ""
    s = str(x).lower(); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()

A_title = {int(r.id): norm(r.title) for r in tableA.itertuples()}
B_title = {int(r.id): norm(r.title) for r in tableB.itertuples()}
A_mfr = {int(r.id): norm(r.manufacturer) for r in tableA.itertuples()}
B_mfr = {int(r.id): norm(r.manufacturer) for r in tableB.itertuples()}

# same hand-curated abbreviation dictionary as ensemble_merge.py's FIX 2
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

# ------------------------------------------------------------------ expansion-fired subset
def fired(a, b):
    return A_title_x[a] != A_title[a] or B_title_x[b] != B_title[b]

pool_fired = np.array([fired(a, b) for a, b in pool])
print(f"pairs where expansion changes A or B title text: {pool_fired.sum()} / {len(pool)} "
      f"({pool_fired.mean():.4f})")

known_fired = [(a, b) for a, b in known if fired(a, b)]
print(f"of the 100 known matches, expansion fires for: {len(known_fired)} pairs")
for a, b in known_fired:
    print(f"    A={a:5d} B={b:5d}  A-title='{A_title[a]}'  B-title='{B_title[b]}' -> '{B_title_x[b]}'")

# ------------------------------------------------------------------ build labeled eval set: 100 knowns + hard negatives
vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
vw.fit(list(A_title.values()) + list(B_title.values()))
all_B_ids = list(B_title.keys())
Ball = vw.transform([B_title[b] for b in all_B_ids])

def hard_negs_for(a_id, true_b, k=5):
    av = vw.transform([A_title[a_id]])
    sims = np.asarray(av.multiply(Ball).sum(axis=1)).ravel()
    order = np.argsort(-sims)
    out = []
    for i in order:
        b = all_B_ids[i]
        if b == true_b or (a_id, int(b)) not in set(pool):
            continue
        out.append((a_id, int(b)))
        if len(out) >= k:
            break
    return out

rng = np.random.default_rng(0)
pool_set = set(pool)
hard_negs = []
for (a, b) in known:
    hard_negs.extend(hard_negs_for(a, b, k=5))
hard_negs = list(dict.fromkeys(hard_negs))  # dedup, preserve order
print(f"hard negatives built: {len(hard_negs)}")

eval_pairs = list(known) + hard_negs
eval_y = np.array([1] * len(known) + [0] * len(hard_negs))
eval_fired = np.array([fired(a, b) for a, b in eval_pairs])
print(f"eval set: {len(known)} positives + {len(hard_negs)} hard negatives; "
      f"{eval_fired.sum()} of {len(eval_pairs)} are expansion-fired")

# ------------------------------------------------------------------ cross-encoder scoring: raw vs expanded
print("\nloading cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

cta_raw = {i: f"{A_title[i]} {A_mfr[i]}".strip() for i in A_title}
ctb_raw = {i: f"{B_title[i]} {B_mfr[i]}".strip() for i in B_title}
cta_exp = {i: f"{A_title_x[i]} {A_mfr[i]}".strip() for i in A_title}
ctb_exp = {i: f"{B_title_x[i]} {B_mfr[i]}".strip() for i in B_title}

print("scoring eval pairs, raw text...")
ce_raw = ce.predict([(cta_raw[a], ctb_raw[b]) for a, b in eval_pairs], batch_size=256, show_progress_bar=False)
print("scoring eval pairs, expanded text...")
ce_exp = ce.predict([(cta_exp[a], ctb_exp[b]) for a, b in eval_pairs], batch_size=256, show_progress_bar=False)
ce_raw = np.asarray(ce_raw); ce_exp = np.asarray(ce_exp)

# ------------------------------------------------------------------ report
report = []
report.append("=== Cross-Encoder: Raw Text vs Abbreviation-Expanded Text ===\n")
report.append(f"Pool size: {len(pool)}; expansion fires on {pool_fired.sum()} pairs ({pool_fired.mean():.4f}).")
report.append(f"Of the 100 known matches, expansion fires for {len(known_fired)} pairs.")
report.append(f"Eval set: {len(known)} positives + {len(hard_negs)} hard negatives "
              f"({eval_fired.sum()} expansion-fired).\n")

def auc_safe(y, s):
    try:
        return roc_auc_score(y, s)
    except Exception:
        return float("nan")

auc_raw_all = auc_safe(eval_y, ce_raw)
auc_exp_all = auc_safe(eval_y, ce_exp)
report.append(f"FULL eval set  -- AUC raw-text CE: {auc_raw_all:.4f}   AUC expanded-text CE: {auc_exp_all:.4f}   "
              f"delta: {auc_exp_all-auc_raw_all:+.4f}")

mask = eval_fired
if mask.sum() >= 4 and mask.sum() < len(mask):
    auc_raw_f = auc_safe(eval_y[mask], ce_raw[mask])
    auc_exp_f = auc_safe(eval_y[mask], ce_exp[mask])
    report.append(f"FIRED subset ({mask.sum()} pairs) -- AUC raw-text CE: {auc_raw_f:.4f}   "
                  f"AUC expanded-text CE: {auc_exp_f:.4f}   delta: {auc_exp_f-auc_raw_f:+.4f}")
else:
    report.append(f"FIRED subset too small/degenerate for AUC ({mask.sum()} pairs, "
                  f"{eval_y[mask].sum() if mask.sum() else 0} positives) -- reporting raw score means instead.")

if mask.sum() > 0:
    pos_mask = mask & (eval_y == 1)
    neg_mask = mask & (eval_y == 0)
    report.append(f"\nWithin fired subset -- mean CE score (raw / expanded):")
    if pos_mask.sum():
        report.append(f"  known matches   (n={pos_mask.sum():3d}): raw={ce_raw[pos_mask].mean():+.3f}  "
                      f"expanded={ce_exp[pos_mask].mean():+.3f}  shift={ce_exp[pos_mask].mean()-ce_raw[pos_mask].mean():+.3f}")
    if neg_mask.sum():
        report.append(f"  hard negatives  (n={neg_mask.sum():3d}): raw={ce_raw[neg_mask].mean():+.3f}  "
                      f"expanded={ce_exp[neg_mask].mean():+.3f}  shift={ce_exp[neg_mask].mean()-ce_raw[neg_mask].mean():+.3f}")

# per-pair shift for known_fired matches specifically (the diagnosed hard cases)
report.append("\nPer-pair CE score shift for the known-match pairs where expansion fires:")
kf_set = set(known_fired)
for idx, (a, b) in enumerate(eval_pairs):
    if eval_y[idx] == 1 and (a, b) in kf_set:
        report.append(f"    A={a:5d} B={b:5d}  raw={ce_raw[idx]:+.3f}  expanded={ce_exp[idx]:+.3f}  "
                      f"shift={ce_exp[idx]-ce_raw[idx]:+.3f}")

corr = np.corrcoef(ce_raw, ce_exp)[0, 1]
report.append(f"\ncorrelation(raw CE score, expanded CE score) over full eval set: {corr:.4f}")

report_text = "\n".join(report)
print("\n" + report_text)
with open(f"{BASE}/research_ce_expanded_report.txt", "w") as f:
    f.write(report_text + "\n")
print(f"\nsaved report -> {BASE}/research_ce_expanded_report.txt")
