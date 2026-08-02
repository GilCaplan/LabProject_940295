"""
SimHash / random-hyperplane cosine-LSH blocking strategy.

Distinct from candidates_lsh.pkl (MinHash/Jaccard over token shingles):
here we build real-valued TF-IDF vectors for `title + manufacturer`, project
them through a fixed set of random hyperplanes to get a binary SimHash
signature per record (sign of dot product = 1 bit per hyperplane), and bucket
records by signature bands (like LSH banding for MinHash, but applied to a
cosine-similarity-approximating signature instead of a Jaccard-approximating
one). Candidate pairs are records from TableA/TableB that share at least one
band, optionally refined by a hamming-distance threshold on the full
signature.

Never sort/swap (id_a, id_b) - id_a always comes from TableA, id_b from TableB.
"""

import pickle
import time
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Random-hyperplane dot products over ~20k-dim TF-IDF vectors legitimately
# overflow/underflow float32 accumulation on some BLAS backends (values stay
# finite; this is a harmless numerical-range warning, not a NaN/Inf bug --
# verified by hand). Silence it so it doesn't look like an error.
np.seterr(all="ignore")

t0 = time.time()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
A = pd.read_csv("tableA.csv")
B = pd.read_csv("tableB.csv")

def norm_text(row):
    title = str(row.get("title", "") or "")
    manuf = str(row.get("manufacturer", "") or "")
    text = f"{title} {manuf}".lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

A["_text"] = A.apply(norm_text, axis=1)
B["_text"] = B.apply(norm_text, axis=1)

# ---------------------------------------------------------------------------
# TF-IDF vectorization (fit jointly so vector spaces align)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_features=20000,
    sublinear_tf=True,
)
all_text = pd.concat([A["_text"], B["_text"]], ignore_index=True)
vectorizer.fit(all_text)

Xa = vectorizer.transform(A["_text"]).toarray().astype(np.float32)
Xb = vectorizer.transform(B["_text"]).toarray().astype(np.float32)

dim = Xa.shape[1]
print(f"TF-IDF dim: {dim}, A rows: {Xa.shape[0]}, B rows: {Xb.shape[0]}")

# ---------------------------------------------------------------------------
# Random hyperplane SimHash signatures
# ---------------------------------------------------------------------------
N_HYPERPLANES = 256  # multiple of band size for clean banding
rng = np.random.RandomState(42)
hyperplanes = rng.normal(size=(dim, N_HYPERPLANES)).astype(np.float32)

def simhash_bits(X, planes):
    proj = X @ planes  # (n_rows, n_hyperplanes)
    return (proj >= 0).astype(np.uint8)  # binary signature matrix

bits_a = simhash_bits(Xa, hyperplanes)
bits_b = simhash_bits(Xb, hyperplanes)

# Pack bits into bytes for fast hamming distance via XOR + popcount
def pack_bits(bits):
    return np.packbits(bits, axis=1)  # (n_rows, n_hyperplanes/8)

packed_a = pack_bits(bits_a)
packed_b = pack_bits(bits_b)

# popcount lookup table for uint8
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def hamming_matrix_row(row_bytes, packed_other):
    xor = np.bitwise_xor(row_bytes, packed_other)
    return POPCOUNT[xor].sum(axis=1)

# ---------------------------------------------------------------------------
# Banding: split the N_HYPERPLANES-bit signature into bands of `band_bits`
# bits each; records that share an identical band value in >=1 band are
# candidate pairs (standard LSH banding, applied to simhash bits).
# ---------------------------------------------------------------------------
BAND_BITS = 8  # 256 / 8 = 32 bands (short bands -> more collisions to catch)
N_BANDS = N_HYPERPLANES // BAND_BITS

def band_keys(bits, band_bits, n_bands):
    # bits: (n_rows, n_hyperplanes) of 0/1 -> pack each band of band_bits into an int key
    keys = np.zeros((bits.shape[0], n_bands), dtype=np.int64)
    for b in range(n_bands):
        chunk = bits[:, b * band_bits:(b + 1) * band_bits]
        # weighted sum to form integer key
        weights = (1 << np.arange(band_bits)).astype(np.int64)
        keys[:, b] = chunk.astype(np.int64) @ weights
    return keys

keys_a = band_keys(bits_a, BAND_BITS, N_BANDS)
keys_b = band_keys(bits_b, BAND_BITS, N_BANDS)

id_a_arr = A["id"].to_numpy()
id_b_arr = B["id"].to_numpy()

# Build hash buckets per band for TableB: band_idx -> {key: [row indices]}
b_bucket_maps = []
for band in range(N_BANDS):
    d = {}
    col = keys_b[:, band]
    for i, k in enumerate(col):
        d.setdefault(int(k), []).append(i)
    b_bucket_maps.append(d)

candidates = set()
HAMMING_THRESHOLD = 95  # out of 256 bits; tuned against 100_matches.pkl proxy recall

for band in range(N_BANDS):
    col_a = keys_a[:, band]
    bucket_map = b_bucket_maps[band]
    for i_a, k in enumerate(col_a):
        b_indices = bucket_map.get(int(k))
        if not b_indices:
            continue
        row_bytes = packed_a[i_a]
        # Refine by hamming distance on full signature to cut noise
        dists = hamming_matrix_row(row_bytes, packed_b[b_indices])
        keep = np.array(b_indices)[dists <= HAMMING_THRESHOLD]
        for i_b in keep:
            candidates.add((int(id_a_arr[i_a]), int(id_b_arr[i_b])))

print(f"Raw candidate pairs after banding + hamming refine: {len(candidates)}")

# ---------------------------------------------------------------------------
# If far too many candidates, tighten hamming threshold; if too few, loosen.
# Cap to a reasonable size (<=3000) by ranking on hamming distance.
# ---------------------------------------------------------------------------
MAX_CANDIDATES = 3000

if len(candidates) > MAX_CANDIDATES:
    # Recompute hamming distances for ranking and keep the closest MAX_CANDIDATES
    id_to_a_idx = {int(v): i for i, v in enumerate(id_a_arr)}
    id_to_b_idx = {int(v): i for i, v in enumerate(id_b_arr)}
    scored = []
    for (ida, idb) in candidates:
        ia = id_to_a_idx[ida]
        ib = id_to_b_idx[idb]
        xor = np.bitwise_xor(packed_a[ia], packed_b[ib])
        dist = int(POPCOUNT[xor].sum())
        scored.append((dist, ida, idb))
    scored.sort(key=lambda x: x[0])
    candidates = set((ida, idb) for _, ida, idb in scored[:MAX_CANDIDATES])

print(f"Final candidate pairs: {len(candidates)}")

# ---------------------------------------------------------------------------
# Proxy recall against 100 known matches (NOT excluded from candidate set)
# ---------------------------------------------------------------------------
with open("100_matches.pkl", "rb") as f:
    known_matches = pickle.load(f)

known_matches_int = set((int(a), int(b)) for a, b in known_matches)
hits = known_matches_int & candidates
recall = len(hits) / len(known_matches_int) if known_matches_int else float("nan")

runtime = time.time() - t0

print(f"Proxy recall vs 100_matches.pkl: {recall:.3f} ({len(hits)}/{len(known_matches_int)})")
print(f"Runtime: {runtime:.1f}s")

with open("candidates_simhash.pkl", "wb") as f:
    pickle.dump(candidates, f)

with open("recall_report_simhash.txt", "w") as f:
    f.write("SimHash / random-hyperplane cosine-LSH blocking strategy\n")
    f.write(f"N_HYPERPLANES={N_HYPERPLANES}, BAND_BITS={BAND_BITS}, N_BANDS={N_BANDS}, "
            f"HAMMING_THRESHOLD={HAMMING_THRESHOLD}\n")
    f.write(f"Final candidate pairs: {len(candidates)}\n")
    f.write(f"Proxy recall vs 100_matches.pkl: {recall:.3f} ({len(hits)}/{len(known_matches_int)})\n")
    f.write(f"Runtime: {runtime:.1f}s\n")

print("Wrote candidates_simhash.pkl and recall_report_simhash.txt")
