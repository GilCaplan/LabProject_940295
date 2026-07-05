"""
strategy_fable.py -- asymmetric IDF-weighted title-containment blocking.

Qualitative pattern (found by reading matched pairs side-by-side, see
100_matches.pkl): TableB titles are typically TableA's title *plus* junk:

    B title = [manufacturer prefix] [vendor part number] <A-title core>
              [platform/edition suffix: "( win 98 me 2000 xp )",
               "software for windows full version", "1 user std", ...]

e.g.  A: "punch ! master landscape & home design"
      B: "punch software 26100 punch ! master landscape & home design
          architectural 1 user ( s ) pc"

Symmetric similarity (Jaccard, cosine TF-IDF, embeddings) penalizes B's
extra tokens. The right-shaped score here is *asymmetric containment*:
what fraction of A's (IDF-weighted) title mass is covered by B's tokens.
On the 100 known matches, >=60% raw containment holds for ~78/100 pairs
before normalization.

Domain normalizations applied first (also read off the matched pairs):
  * split version glue: "v3 .0" / "v 1.5" / "v4.0" -> "v" + number
  * abbreviation canonicalization: upg->upgrade, win->windows,
    ed->edition, std->standard, prof->professional, acad/academic,
    lic/lics->license, prepaid/pre-paid, etc.
  * hyphen/space unification ("pc-cillin" == "pc cillin") and
    de-concatenation is approximated by also indexing 4-gram-joined
    bigrams ("resume maker" -> "resumemaker") so glued variants match.

Output: candidates_fable.pkl -- a Python set of (id_a, id_b) tuples with
native ids as read from the CSVs (ints). Known matches are NOT excluded
here (that happens at the final merge). Never sorted(): id namespaces of
the two tables overlap, provenance is tracked positionally.
"""

import math
import pickle
import re
from collections import defaultdict

import pandas as pd

CAP = 3000
DIR = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"

# ---------------------------------------------------------------- normalize
ABBREV = {
    "upg": "upgrade", "upgrd": "upgrade",
    "win": "windows", "windows": "windows",
    "mac": "mac", "macintosh": "mac",
    "ed": "edition", "edn": "edition",
    "std": "standard", "pro": "professional", "prof": "professional",
    "acad": "academic", "ac": "academic",
    "lic": "license", "lics": "license", "licence": "license",
    "vol": "volume", "ver": "version",
    "prepaid": "prepaid", "pre-paid": "prepaid",
    "sec": "security", "int": "internet", "intl": "international",
    "dlx": "deluxe", "prem": "premium", "prm": "premium",
    "qckbks": "quickbooks", "rpt": "report", "rpts": "report",
    "wkgp": "workgroup", "trng": "training", "gr": "grade",
    "beginners": "beginning", "producers": "producer",
    "&": "and",
}

STOP = {
    "the", "for", "and", "with", "of", "by", "a", "an", "in", "to", "on",
    "software", "1", "user", "users", "complete", "package", "full",
    "version", "retail", "box", "cd", "dvd", "rom", "jewel", "case",
    "windows", "mac", "s",
}

_num_glue = re.compile(r"\bv\s*(\d+)\s*\.\s*(\d+)")
_tok = re.compile(r"[a-z0-9\.]+")


def normalize(text: str) -> list:
    t = str(text).lower()
    t = _num_glue.sub(r"v\1.\2", t)          # "v3 .0" / "v 3.0" -> "v3.0"
    t = t.replace("-", " ")                   # pc-cillin == pc cillin
    toks = _tok.findall(t)
    out = []
    for tk in toks:
        tk = tk.strip(".")
        if not tk:
            continue
        tk = ABBREV.get(tk, tk)
        if len(tk) > 4 and tk.isalpha() and tk.endswith("s"):
            tk = tk[:-1]                      # light stemming
        out.append(tk)
    # glued bigrams so "resume maker" also matches "resumemaker" -- used
    # for COVERAGE only (extended set), never in the denominator
    glued = [out[i] + out[i + 1] for i in range(len(out) - 1)
             if out[i].isalpha() and out[i + 1].isalpha()]
    return out, glued


def content(tokens):
    return {t for t in tokens if t not in STOP and len(t) > 1}


# ---------------------------------------------------------------- load
dfA = pd.read_csv(f"{DIR}/tableA.csv")
dfB = pd.read_csv(f"{DIR}/tableB.csv")

def build(df):
    core, ext = {}, {}
    for _, r in df.iterrows():
        toks, glued = normalize(f"{r['title']} {r['manufacturer']}")
        c = content(toks)
        core[r["id"]] = c
        ext[r["id"]] = c | set(glued)
    return core, ext


A_core, A_ext = build(dfA)
B_core, B_ext = build(dfB)

# IDF from combined corpus (core tokens only)
df_count = defaultdict(int)
for toks in list(A_core.values()) + list(B_core.values()):
    for t in toks:
        df_count[t] += 1
N = len(A_core) + len(B_core)
idf = {t: math.log(N / (1 + c)) for t, c in df_count.items()}

# inverted index over B's extended tokens (blowup control on frequency)
inv = defaultdict(list)
for bid, toks in B_ext.items():
    for t in toks:
        if df_count.get(t, 0) <= 400:
            inv[t].append(bid)


def containment(core_self, ext_other):
    """IDF-weighted fraction of self's core tokens covered by other."""
    denom = num = 0.0
    for t in core_self:
        w = idf.get(t, 0.1)
        denom += w
        if t in ext_other:
            num += w
    return num / denom if denom > 0 else 0.0


# ---------------------------------------------------------------- score
pair_scores = {}
for aid, atoks in A_core.items():
    if not atoks:
        continue
    hits = defaultdict(int)
    for t in A_ext[aid]:
        for bid in inv.get(t, ()):
            hits[bid] += 1
    for bid in hits:
        c = max(containment(atoks, B_ext[bid]),
                containment(B_core[bid], A_ext[aid]))
        if c >= 0.30:
            # tiny symmetric tiebreak so a B that is *exactly* A beats a
            # B that merely contains A plus a lot of junk
            jac = len(atoks & B_core[bid]) / len(atoks | B_core[bid])
            pair_scores[(aid, bid)] = c + 0.10 * jac

# ---------------------------------------------------------------- select
# Global top-N is the wrong selection for this signal: B contains many
# near-duplicate listings, so a handful of A rows hog thousands of
# score>0.8 pairs. Per-record quotas spread the budget: top-2 B's per A
# row, plus top-1 A per B row (catches B rows whose best A didn't
# reciprocate), then trim lowest scores to the cap. Measured on the 100
# known matches: global-top-2500 = 0.76-0.81, this scheme = ~0.89 @ 3000.
byA, byB = defaultdict(list), defaultdict(list)
for (aid, bid), s in pair_scores.items():
    byA[aid].append((s, bid))
    byB[bid].append((s, aid))
for lst in byA.values():
    lst.sort(reverse=True)
for lst in byB.values():
    lst.sort(reverse=True)

cands = set()
for aid, lst in byA.items():
    cands |= {(aid, bid) for s, bid in lst[:2]}
for bid, lst in byB.items():
    cands |= {(aid, bid) for s, aid in lst[:1]}
if len(cands) > CAP:
    cands = set(sorted(cands, key=lambda p: -pair_scores[p])[:CAP])

# ---------------------------------------------------------------- report
with open(f"{DIR}/100_matches.pkl", "rb") as f:
    known = pickle.load(f)
known = {(a, b) for a, b in known}
hit = len(cands & known)
print(f"candidates: {len(cands)}")
print(f"proxy recall vs 100 known matches: {hit}/100 = {hit/100:.2f}")
missed = known - cands
if missed:
    print("missed known matches:")
    idxA = dfA.set_index("id")
    idxB = dfB.set_index("id")
    for a, b in list(missed)[:20]:
        print("  A:", idxA.loc[a, "title"])
        print("  B:", idxB.loc[b, "title"])

with open(f"{DIR}/candidates_fable.pkl", "wb") as f:
    pickle.dump(cands, f)
print("wrote candidates_fable.pkl")
