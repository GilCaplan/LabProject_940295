"""
DeepBlocker-style self-supervised tuple-embedding blocking.

Approach:
1. Build normalized concatenated text per record (title + manufacturer + price).
2. Fit a TF-IDF vectorizer on the UNION of TableA + TableB text (self-supervised,
   no labels used) to get sparse lexical features.
3. Train a small autoencoder (PyTorch) on these TF-IDF vectors (fit on the union
   of both tables) to learn a dense, compressed "tuple embedding" in an
   unsupervised fashion -- this is the DeepBlocker-style self-supervised
   representation-learning step (structurally different from the
   pretrained-sentence-transformer approach used by the `blocking-embeddings`
   agent).
4. Encode all TableA/TableB records with the trained encoder to get dense
   embeddings.
5. Use faiss for per-record top-K nearest-neighbor search: for each TableA
   record, find its top-K nearest TableB records in embedding space (L2 on
   normalized vectors == cosine ranking). This is DeepBlocker's actual pairing
   strategy (per-record top-K), not a global top-K by score.
6. Union all pairs across TableA records, dedupe via plain set add (no sorted()
   on tuples -- provenance (id_a from A, id_b from B) tracked explicitly
   throughout), drop any accidental self-pairs where ids coincide across the
   shared id namespace but are NOT the same tuple slot (not applicable here
   since we always build (id_a, id_b) explicitly).
7. If the union exceeds the target budget, truncate by nearest-neighbor
   distance (keep closest pairs first).
8. Compute proxy recall against the 100 known matches (before exclusion),
   then EXCLUDE those 100 exact tuples from the saved candidate set (per task
   rules -- exclusion happens at this stage per this agent's instructions,
   even though the standard checklist says do it at final submission; the
   parent ensemble step will also do its own exclusion pass, this is just a
   safe local double-check and reporting step).
9. Save the candidate set (Python set of (id_a, id_b) tuples) to
   candidates_deepblocker.pkl via plain pickle.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import re
import time

import numpy as np
import pandas as pd
import faiss

faiss.omp_set_num_threads(1)

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

torch.set_num_threads(1)

t0 = time.time()

BASE = "/Users/USER/Desktop/University/Semester 8/Lab/Hackathon"

# ---------------------------------------------------------------------------
# 1. Load & normalize text
# ---------------------------------------------------------------------------


def normalize(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


tableA = pd.read_csv(f"{BASE}/TableA.csv")
tableB = pd.read_csv(f"{BASE}/TableB.csv")

tableA_ids = set(tableA["id"])
tableB_ids = set(tableB["id"])
print("id namespace overlap size:", len(tableA_ids & tableB_ids))


def build_text(df):
    title = df["title"].map(normalize)
    manufacturer = df["manufacturer"].map(normalize)
    price = df["price"].map(lambda p: "" if pd.isna(p) else f"price{int(round(float(p)))}")
    return (title + " " + manufacturer + " " + price).str.strip()


tableA["text"] = build_text(tableA)
tableB["text"] = build_text(tableB)

print("Sample A text:", tableA["text"].iloc[0])
print("Sample B text:", tableB["text"].iloc[0])

# ---------------------------------------------------------------------------
# 2. TF-IDF over union of both tables (self-supervised, no labels)
# ---------------------------------------------------------------------------

all_texts = pd.concat([tableA["text"], tableB["text"]], ignore_index=True)

vectorizer = TfidfVectorizer(
    max_features=4000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
tfidf_all = vectorizer.fit_transform(all_texts)  # sparse (nA+nB, F)
print("TF-IDF matrix shape:", tfidf_all.shape)

nA = len(tableA)
nB = len(tableB)

tfidf_dense = tfidf_all.toarray().astype(np.float32)  # small enough (4589 x 4000)

# ---------------------------------------------------------------------------
# 3. Train a small autoencoder on the TF-IDF features (unsupervised)
# ---------------------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")

input_dim = tfidf_dense.shape[1]
hidden_dim = 256
latent_dim = 64


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


model = AutoEncoder(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X = torch.from_numpy(tfidf_dense).to(device)

n_epochs = 30
batch_size = 256
n_samples = X.shape[0]

model.train()
for epoch in range(n_epochs):
    perm = torch.randperm(n_samples)
    total_loss = 0.0
    for i in range(0, n_samples, batch_size):
        idx = perm[i : i + batch_size]
        batch = X[idx]
        optimizer.zero_grad()
        _, recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.shape[0]
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:3d}  recon_loss={total_loss / n_samples:.6f}")

model.eval()
with torch.no_grad():
    Z, _ = model(X)
    Z = Z.numpy()

embA = Z[:nA]
embB = Z[nA:]

# normalize embeddings for cosine-style ranking via L2 on unit vectors
def l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


embA_n = l2_normalize(embA).astype(np.float32)
embB_n = l2_normalize(embB).astype(np.float32)

print("Trained autoencoder embeddings:", embA_n.shape, embB_n.shape)

# ---------------------------------------------------------------------------
# 5. faiss per-record top-K nearest neighbor search (A -> B)
# ---------------------------------------------------------------------------

# K tuned empirically (swept K=4..30, checked proxy recall vs 100_matches.pkl):
# smaller K -> smaller pool but lower recall; K=8 clears the 0.6 recall milestone
# per-record (guaranteeing coverage for every TableA record) while keeping the
# pool a manageable "few thousand"-scale size for the downstream ensemble step.
# We deliberately do NOT do a global top-N-by-score truncation on top of the
# per-record union: empirically that drops recall a lot (truncating by score
# tends to keep many redundant near-duplicate neighbors for popular/generic B
# records while discarding a correct but lower-absolute-score match for a
# harder A record) -- the per-record top-K guarantee is the whole point of
# the DeepBlocker pairing strategy, so we only truncate as a safety net if the
# union ever balloons far past a reasonable budget.
K = 8
TARGET_BUDGET = 12000

index = faiss.IndexFlatIP(embB_n.shape[1])  # inner product on unit vectors = cosine sim
index.add(embB_n)

sims, neighbor_idx = index.search(embA_n, K)  # shape (nA, K)

a_ids = tableA["id"].to_numpy()
b_ids = tableB["id"].to_numpy()

pairs_with_score = []  # list of (sim, id_a, id_b)
candidate_set = set()
seen = set()

for i in range(nA):
    id_a = a_ids[i]
    for k in range(K):
        j = neighbor_idx[i, k]
        if j < 0:
            continue
        id_b = b_ids[j]
        sim = float(sims[i, k])
        pair = (id_a, id_b)
        if pair not in seen:
            seen.add(pair)
            pairs_with_score.append((sim, id_a, id_b))

print("Union of per-record top-K pairs (before truncation):", len(pairs_with_score))

# ---------------------------------------------------------------------------
# 6/7. Truncate to budget by similarity if needed
# ---------------------------------------------------------------------------

if len(pairs_with_score) > TARGET_BUDGET:
    pairs_with_score.sort(key=lambda x: x[0], reverse=True)
    pairs_with_score = pairs_with_score[:TARGET_BUDGET]

candidate_set = {(id_a, id_b) for (_, id_a, id_b) in pairs_with_score}

print("Final candidate set size (pre-exclusion):", len(candidate_set))

# ---------------------------------------------------------------------------
# 8. Proxy recall against the 100 known matches (BEFORE exclusion)
# ---------------------------------------------------------------------------

with open(f"{BASE}/100_matches.pkl", "rb") as f:
    known_matches = pickle.load(f)

overlap = candidate_set & known_matches
proxy_recall = len(overlap) / len(known_matches)
print(f"Proxy recall vs 100_matches.pkl (before exclusion): {proxy_recall:.3f} "
      f"({len(overlap)}/{len(known_matches)})")

# Explicitly exclude the 100 known matches from the saved set (per task rules)
candidate_set -= known_matches

print("Final candidate set size (post-exclusion):", len(candidate_set))

# sanity: no self/invalid pairs, no sorted() shenanigans -- verify provenance
bad = [p for p in candidate_set if p[0] not in tableA_ids or p[1] not in tableB_ids]
assert not bad, f"Found invalid pairs: {bad[:5]}"
assert len(candidate_set) <= TARGET_BUDGET + 1  # intermediate strategy pool, not final submission cap

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------

with open(f"{BASE}/candidates_deepblocker.pkl", "wb") as f:
    pickle.dump(candidate_set, f)

runtime = time.time() - t0

print("=" * 70)
print("DeepBlocker-style strategy summary")
print("=" * 70)
print(f"Embedding approach: custom autoencoder (unsupervised) trained on TF-IDF "
      f"features of the UNION of TableA+TableB text (title+manufacturer+price).")
print(f"Pairing strategy: per-record top-{K} nearest-neighbor (faiss IndexFlatIP, "
      f"cosine via inner product on L2-normalized embeddings), not a global threshold.")
print(f"Candidate set size (final, post-exclusion): {len(candidate_set)}")
print(f"Proxy recall vs 100 known matches (pre-exclusion): {proxy_recall:.3f}")
print(f"Runtime: {runtime:.1f}s")
print(f"Saved to: {BASE}/candidates_deepblocker.pkl")
