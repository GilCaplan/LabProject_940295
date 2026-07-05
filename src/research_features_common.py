"""
RESEARCH-ONLY shared helper (does NOT touch ensemble_merge.py / final_candidates.pkl).

Faithful reproduction of ensemble_merge.py's pool-building + 15-feature
featurization (FEATS list, wcos/ccos/jac/title_tsr/title_tset/mfr_tsr/pclose/
pknown/ce/n_strats/wcos_x/jac_x/title_tset_x/mfr_inf_tset/mfr_inf_partial),
so research_stacking.py and research_numeric_full.py can share one (expensive,
CE-scoring) featurization pass instead of recomputing it twice. Caches to
research_pool_features.pkl on first run.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, re, glob, os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"
CACHE = f"{BASE}/research_pool_features.pkl"

FEATS = ["wcos", "ccos", "jac", "title_tsr", "title_tset", "mfr_tsr",
         "pclose", "pknown", "ce", "n_strats",
         "wcos_x", "jac_x", "title_tset_x", "mfr_inf_tset", "mfr_inf_partial"]

EXCLUDE_FROM_COUNT = {"deepblocker", "rrf", "finetuned_embeddings", "full_supervised"}

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


def norm(x):
    if pd.isna(x): return ""
    s = str(x).lower(); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s).strip()


def build(force=False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        print(f"[research_features_common] loaded cached pool+features from {CACHE} "
              f"(pool size {len(d['pool'])})")
        return d

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
    union_all = set().union(*strat_sets.values())
    COUNT_STRATS = [s for s in strat_sets if s not in EXCLUDE_FROM_COUNT]
    pool = sorted(union_all | known)
    pool_index = {p: i for i, p in enumerate(pool)}
    print(f"[research_features_common] pool size (incl 100 matches): {len(pool)}")

    A_title = {int(r.id): norm(r.title) for r in tableA.itertuples()}
    B_title = {int(r.id): norm(r.title) for r in tableB.itertuples()}
    A_mfr = {int(r.id): norm(r.manufacturer) for r in tableA.itertuples()}
    B_mfr = {int(r.id): norm(r.manufacturer) for r in tableB.itertuples()}
    B_mfr_raw = {int(r.id): r.manufacturer for r in tableB.itertuples()}
    A_price = {int(r.id): r.price for r in tableA.itertuples()}
    B_price = {int(r.id): r.price for r in tableB.itertuples()}

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
        i = 1 if _looks_part(toks[0]) else 0
        brand = []
        for t in toks[i:i + 3]:
            if _looks_part(t): break
            if t in _STOP_BRAND and not brand: continue
            brand.append(t)
        return " ".join(brand)
    B_mfr_inf = {bid: infer_bmfr(bid) for bid in B_title}

    def expand(t): return " ".join(ABBR.get(w, w) for w in t.split())
    A_title_x = {i: expand(t) for i, t in A_title.items()}
    B_title_x = {i: expand(t) for i, t in B_title.items()}

    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vw.fit(list(A_title.values()) + list(B_title.values()))
    Aw = {i: vw.transform([t]) for i, t in A_title.items()}
    Bw = {i: vw.transform([t]) for i, t in B_title.items()}
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    vc.fit(list(A_title.values()) + list(B_title.values()))
    Ac = {i: vc.transform([t]) for i, t in A_title.items()}
    Bc = {i: vc.transform([t]) for i, t in B_title.items()}
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

    print("[research_features_common] scoring pool with cross-encoder (one-time, cached)...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cta = {i: f"{A_title[i]} {A_mfr[i]}".strip() for i in A_title}
    ctb = {i: f"{B_title[i]} {B_mfr[i]}".strip() for i in B_title}
    ce_scores = ce.predict([(cta[a], ctb[b]) for a, b in pool], batch_size=256, show_progress_bar=False)
    ce_map = {pool[i]: float(ce_scores[i]) for i in range(len(pool))}

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
            X[k, 10] = wcosx(a, b); X[k, 11] = jacx(a, b)
            X[k, 12] = fuzz.token_set_ratio(A_title_x[a], B_title_x[b]) / 100
            X[k, 13] = fuzz.token_set_ratio(A_mfr[a], B_mfr_inf[b]) / 100
            X[k, 14] = fuzz.partial_ratio(A_mfr[a], B_mfr_inf[b]) / 100
        return X

    print("[research_features_common] featurizing full pool (15 features)...")
    X_pool = featurize(pool)

    d = dict(pool=pool, pool_index=pool_index, known=known, X_pool=X_pool, FEATS=FEATS,
             tableA_ids=tableA_ids, tableB_ids=tableB_ids,
             A_title=A_title, B_title=B_title)
    with open(CACHE, "wb") as f:
        pickle.dump(d, f)
    print(f"[research_features_common] cached -> {CACHE}")
    return d


if __name__ == "__main__":
    build(force=True)
