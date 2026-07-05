# Blocking Strategies Cheat Sheet

The hackathon task is: **given two (or one) datasets, output a size-limited set of candidate record pairs that contains as many true matches as possible.** This is a pure blocking/recall problem under a hard budget `K` on the number of output pairs. Read this alongside [[02 - Evaluation Metrics]] and [[03 - Python Blocking Toolkit]].

## Reframe the problem correctly
This is NOT "classify every pair as match/non-match." It's:
1. Generate a superset of candidate pairs (blocking).
2. Score every candidate pair with *some* similarity signal.
3. Keep the top-K scoring pairs, where K is the size limit.

So the real objective is **ranking**, not binary blocking. Even a coarse blocking key that keeps 50k candidates is fine as long as you can score and truncate to K afterward. Always build the pipeline so blocking output feeds a scorer, and the scorer output feeds a top-K truncation — never stop at "did the blocking key match."

## Core blocking techniques (cheapest → most powerful)

### 1. Standard / Token (Attribute) Blocking
Group records that share an exact value (or normalized value) on some attribute (last name, zip code, category, brand). Cross product within each block = candidate pairs.
- Pros: trivial, O(n) to build blocks, fast.
- Cons: misses typos/variants entirely (a single character difference = no match).
- Fix: normalize first (lowercase, strip punctuation/whitespace, remove stopwords) before using as a key.

### 2. Sorted Neighborhood
Sort all records by a blocking key (concatenation of normalized fields), slide a fixed-size window of size `w`, pair every record with the others in its window.
- Pros: tolerant to small key errors if sort key is chosen well; tunable cost via window size.
- Cons: sensitive to which field you sort on; errors in the *first few characters* of the sort key are fatal.
- Tip: run it multiple times with different sort keys (e.g., sort by title, then separately by brand+model) and union the pairs — multi-pass blocking.

### 3. Q-gram / Token-set overlap blocking
Break each record's key string into character n-grams (q-grams, e.g., q=3) or word tokens, build an inverted index (token → list of record IDs), and generate candidate pairs for any two records sharing at least one token/q-gram (or at least `t` of them).
- Pros: robust to word reordering and partial typos; easy to implement with a dict.
- Cons: common tokens (stopwords, "the", "inc", numbers like "2023") create huge blocks — blow past your budget. Always drop very frequent tokens (near-stopwords for your domain) before indexing.

### 4. LSH (MinHash / SimHash) blocking
Compute MinHash signatures over token sets (Jaccard) or SimHash over TF-IDF vectors (cosine), bucket by LSH bands, pair records that collide in ≥1 band.
- Pros: scales to very large datasets, approximate but tunable recall/precision via bands×rows.
- Cons: more setup code; needs a library (`datasketch`) or manual implementation.
- Use when token-blocking blocks are too large/slow (record count > ~50k–100k).

### 5. Semantic / Embedding blocking
Embed each record's serialized text with a sentence embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`), then use approximate nearest neighbor (FAISS, or even brute-force cosine sim if data is small) to get each record's top-`m` nearest neighbors as candidates.
- Pros: catches paraphrases/synonyms token overlap can't ("printing calculator" vs "calc, printer style"); usually the best recall for a fixed pair budget when there's time to run it.
- Cons: needs a model + compute; embedding all records is O(n); ANN search still needed at scale.
- **This is very likely your best single technique for maximizing matches within a hard pair budget**, because you can directly ask for "top-K nearest neighbor pairs by cosine similarity" — which *is* the top-K ranking the task wants.

### 6. Hybrid (recommended default)
1. Cheap high-recall blocking (token/q-gram or sorted neighborhood) to cut n×m down to a manageable candidate set.
2. Score every surviving candidate with a stronger signal: string similarity (Jaccard/cosine over tokens, edit distance on key fields) and/or embedding cosine similarity.
3. Sort by score, take top-K, dedupe (pairs are unordered — `(a,b) == (b,a)`, and a pair can arise from multiple blocks/passes).

## Formal framing: this task *is* Meta-Blocking / Progressive Entity Resolution
This isn't just an ad-hoc "score and truncate" trick — it's two well-established subfields of ER research, and knowing the real names/techniques is worth having in your back pocket:

**Meta-blocking** (Papadakis et al., TKDE'13) takes an oversized blocking collection, builds a "blocking graph" (nodes = records, edges = candidate pairs that co-occurred in ≥1 block, edge weight = number/strength of shared blocks), and *prunes* the graph down to a smaller, higher-value edge set while trying to keep the edges connecting true matches. The pruning schemes map directly onto "pick your best K pairs":
- **Cardinality Edge Pruning (CEP)** — keep the **global top-K weighted edges** across the whole graph. This is exactly "take the top-K scoring pairs overall" from the Hybrid recipe above.
- **Cardinality Node Pruning (CNP)** — keep the **top-k edges per node** (per record), instead of a single global cutoff. This tends to preserve recall better when match-worthiness varies a lot by record (some records naturally have many plausible candidates, others very few) — worth trying if global top-K underperforms, since it guarantees every record gets *some* candidates rather than being crowded out by high-degree hub records.
- **Weighted Edge/Node Pruning (WEP/WNP)** — threshold-based instead of cardinality-based (keep edges above the average weight, globally or per-node). Useful if you're not hard-capped at exactly K but want a quality bar.

**Progressive Entity Resolution** is the subfield studying exactly the "budget-constrained, maximize recall as fast/early as possible" setting — algorithms are explicitly evaluated on *recall achieved for a given number of allowed comparisons*, which is precisely the hackathon's scoring setup. If you want one extra idea beyond top-K-by-score: **process/emit pairs in decreasing order of estimated match likelihood** (not just filter-then-cut) — e.g., process records with rarer/more discriminative blocking keys first, since generic/common-key records are cheap to defer without losing much recall.

**Practical takeaway:** the `finalize()` function in [[03 - Python Blocking Toolkit]] already implements global Cardinality Edge Pruning. Also implement a **per-record top-k variant (CNP-style)** and compare PC on a validation sample — whichever wins, use it. See the toolkit for both.

## Picking the right blocking key(s)
- Prefer fields that are **discriminative and low-noise**: product IDs/model numbers, ISBNs, phone last-4-digits, brand+model, title. Avoid free-text fields with lots of noise as sort/exact keys — use them for scoring instead.
- Normalize before anything else: lowercase, strip punctuation, collapse whitespace, remove very common stopwords ("inc", "the", "edition", "version"), normalize digits (see DITTO's span normalization idea).
- Multiple blocking passes on different keys + union of candidate pairs consistently beats a single perfect key. Budget allowing, always union ≥2 blocking strategies before scoring/truncating.

## Common pitfalls (these are exactly what cost points)
- **Self-pairs**: never include (r, r).
- **Duplicate pairs**: the same pair generated by two different blocks/passes — dedupe with a plain `set` add.
- **Order matters in this hackathon**: unlike generic within-table dedup, our output must be `(id_a, id_b)` with `id_a` always from TableA and `id_b` always from TableB (see [[05 - Official Task Instructions (Confirmed)]]). **Do not** use `tuple(sorted((id1, id2)))` to dedupe — if the two tables' id namespaces overlap, sorting can silently swap which id lands in which slot and produce an invalid/misattributed pair. Always build pairs by tracking provenance (which table each id came from), never by sorting.
- **Ignoring the size limit precisely**: know the exact K from the grading doc and never exceed it — truncate deterministically by score, not arbitrarily.
- **Over-reliance on one attribute**: if the key attribute is NULL/missing for a chunk of records (as in the lecture's example table with `–` phone values), those records get orphaned from every block. Always have a fallback pass using another field for records where the primary key is missing/null.
- **Not normalizing before hashing/sorting**: "R&D" vs "Research and Development" won't share tokens unless you expand abbreviations or fall back to a fuzzy signal.
- **Runtime blowup**: q-gram/token blocking on unfiltered stopwords can produce blocks with thousands of records → cross product explodes. Cap block size or drop over-frequent tokens.

## If grading rewards precision too (read the brief Saturday night)
If the metric isn't pure recall/pairs-completeness but also penalizes non-matching pairs (e.g., F1-style or pairs quality), don't just dump every candidate — do a real scoring/threshold step so lower-confidence pairs get dropped before hitting the cap, and prefer precision-boosting signals (numeric field exact-match, embedding similarity threshold) over raw blocking overlap.

## Off-the-shelf tools worth having installed (verified, not just hand-rolled code)
Writing everything from scratch is a good fallback, but these libraries directly solve this exact problem and can save real time — see [[03 - Python Blocking Toolkit]] for usage:

- **[recordlinkage](https://recordlinkage.readthedocs.io/)** — has a built-in `Index` class with `.block()`, `.sortedneighbourhood()`, and `.full()` methods that do exactly the "attribute blocking" / "sorted neighborhood" techniques above in a couple of lines instead of hand-rolled loops. Good default if you want to move fast on the classic techniques.
- **[DeepBlocker](https://github.com/qcri/DeepBlocker)** (QCRI, VLDB'21 "Deep Learning for Blocking in Entity Matching: A Design Space Exploration") — **self-supervised, no labeled data required**, embeds tuples and retrieves candidates via a `VectorPairingModel`. Crucially, it ships an **`ExactTopKVectorPairing(K=...)`** strategy that literally returns the top-K nearest-neighbor pairs per record — this is almost exactly the "size-limited candidate set" framing of the hackathon task. Strong option if you have it installed and working ahead of time.
- **[BlockingPy](https://arxiv.org/abs/2504.04266)** (2025) — ANN-based blocking on character n-grams or embeddings (supports `model2vec` model IDs directly), single pandas-style call. Newer/less battle-tested than the above but worth knowing it exists.
- **[Splink](https://moj-analytical-services.github.io/splink/)** — built for probabilistic record linkage at scale (millions of rows), with `block_on(...)` rules including phonetic blocking (Soundex/Metaphone). Overkill for small/medium datasets but the right call if the hackathon dataset turns out to be huge.

Decision rule: if the dataset is small-to-medium and you're comfortable with the custom code in [[03 - Python Blocking Toolkit]], hand-rolled gives you the most control over ranking/truncation to hit the exact metric. If you're short on time, `recordlinkage` for classic blocking + your own top-K scoring/truncation step is the fastest path to a correct baseline.

## Sources
- [DITTO paper (arXiv 2004.00584)](https://arxiv.org/abs/2004.00584) and [official code](https://github.com/megagonlabs/ditto)
- [A Survey of Blocking and Filtering Techniques for Entity Resolution (arXiv 1905.06167)](https://arxiv.org/pdf/1905.06167)
- [DeepBlocker repo](https://github.com/qcri/DeepBlocker) / [paper (VLDB'21)](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- [BlockingPy paper (arXiv 2504.04266)](https://arxiv.org/abs/2504.04266)
- [recordlinkage docs](https://recordlinkage.readthedocs.io/)
- [Meta-Blocking: Taking Entity Resolution to the Next Level (TKDE'13)](https://helios2.mi.parisdescartes.fr/~themisp/publications/tkde13-metablocking.pdf) — CEP/CNP/WEP/WNP pruning schemes
- [Progressive Entity Resolution: A Design Space Exploration (arXiv 2503.08298)](https://arxiv.org/html/2503.08298v1)
- [Entity Matching using Large Language Models (arXiv 2310.11244)](https://arxiv.org/pdf/2310.11244) — zero-shot/few-shot LLM matching, relevant if you want an LLM-based final rescoring pass
