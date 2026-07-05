# Hackathon Game Plan (Sunday)

Study order: [[Lecture 6 - Data Integration]] → [[01 - Blocking Strategies Cheat Sheet]] → [[02 - Evaluation Metrics]] → [[03 - Python Blocking Toolkit]] → this file.

**The official instructions have arrived — see [[05 - Official Task Instructions (Confirmed)]] first.** It resolves every "identify the metric/size limit/format" step below with the real numbers (K=2000, pure recall grading with a 0.6 threshold, output is a Python `set`, and a 100-pair labeled sample you must use but not leak into your output). `solution_starter.py` in this folder is a ready-to-adapt end-to-end script implementing steps 1–9 below against the real file names (`TableA.csv`, `TableB.csv`, `100_matches.pkl`). The sections below are now historical planning context — still useful for technique choices, but treat [[05 - Official Task Instructions (Confirmed)]] as authoritative wherever they conflict.

## Before Saturday night's instructions doc arrives
- [ ] Re-read the lecture md, especially **Blocking** and **Pipeline** sections — that's the exact syllabus scope for this hackathon.
- [ ] Pre-install everything in [[03 - Python Blocking Toolkit]]'s setup section on your laptop (`pip install ...`). Confirm `sentence-transformers` downloads a model successfully at least once (first run pulls weights from the internet — do this before you might lose signal Sunday).
- [ ] Skim your tutorial materials/exercises on blocking (if you have them) — the hackathon is explicitly tied to "the tutorial," so any code you wrote there is your fastest starting point.
- [ ] Set up a scratch project folder with the toolkit snippets already saved as a `.py` file so Sunday is copy-paste-adapt, not type-from-scratch.
- [ ] Test-drive `recordlinkage`'s `Index.block()`/`.sortedneighbourhood()` once on any small CSV — it's the fastest correct baseline and worth being fluent in before Sunday (see [[03 - Python Blocking Toolkit]] section 0).
- [ ] If time allows, clone [DeepBlocker](https://github.com/qcri/DeepBlocker) and get its `requirements.txt` installed + a smoke-test run working — it's self-supervised (no labels needed) and its `ExactTopKVectorPairing(K=...)` maps directly onto "size-limited candidate pairs." Don't attempt this for the first time mid-hackathon if setup is flaky.

## The moment you get Saturday's document
1. Identify **the exact metric** (PC only? PC + PQ? F1? something else) — this single fact determines whether you should "fill the budget with your best top-K" or "threshold conservatively." See [[02 - Evaluation Metrics]].
2. Identify **the exact size limit** `K` and the **required output format** (columns, file type, ordering, whether pairs are ordered or unordered).
3. Identify **dataset scale** (row counts of D1/D2) — this determines whether embeddings-on-everything is feasible in the time limit or whether you need to pre-filter with cheap token blocking first.
4. Identify **time limit** for the hackathon itself and any runtime constraint on the submission/script.

## Day-of sequence (adapt once you know the real numbers)
1. **Load + explore data** (10–15 min): check schema, missing values, obvious duplicate/near-duplicate columns, id columns.
2. **Normalize** all text fields (lowercase, strip punctuation, expand obvious abbreviations you spot by eye — e.g., "R&D" → "research and development").
3. **Cheap high-recall blocking first** (token/q-gram + sorted neighborhood on 2–3 different keys). Get *a* candidate set working end-to-end before optimizing — a mediocre full pipeline beats a perfect blocking step with no scoring/output.
4. **Add a stronger signal** (rapidfuzz token_sort_ratio and/or TF-IDF cosine) to score all candidates from step 3.
5. **If time/compute allows**, run semantic/embedding blocking and union its candidates into the pool before final scoring — this is usually the single highest-leverage step for recall.
6. **Dedupe + truncate to K** — try both global top-K (`finalize()`, i.e. Cardinality Edge Pruning) and per-record top-k (`finalize_cnp()`, Cardinality Node Pruning) if you have a validation sample; CNP tends to win when a few generic/hub records would otherwise dominate the budget. See [[03 - Python Blocking Toolkit]] section 7. Write output in the required format.
7. **Validate**: if any labeled sample is provided, compute PC/PQ locally. If not, spot-check 20–30 pairs by eye for sanity.
8. **Iterate only if time remains**: try alternate blocking keys, tune `top_k` in embeddings, try lowering/raising the score threshold, re-check PC/PQ each time. If there's real time to spare and the pair count is small enough, consider a final LLM rescoring pass (zero-shot "are these the same entity?" prompt) on your borderline pairs only — recent work (ComEM, BATCHER) shows simple zero-shot prompts work surprisingly well for EM, but this is a bonus, not a baseline dependency.
9. **Freeze and submit early** — don't tune until the deadline; leave 15–20 min buffer for format/submission issues.

## Time-boxing (adjust to actual event length)
- If it's a half-day event (~4 hrs): ~30 min explore/plan, ~90 min build baseline pipeline (steps 1–4), ~60 min add embeddings + tune, ~30 min validate/iterate, ~30 min buffer.
- Whatever the real length, reserve the **last 10–15% of total time** purely for output formatting + submission — a correct pipeline with a malformed output file scores zero.

## Mental checklist while coding (catches the point-losing mistakes)
- Am I deduping unordered pairs `(a,b)==(b,a)`?
- Am I excluding self-pairs?
- Am I strictly respecting the size cap?
- Did I normalize before blocking (not after)?
- Do I have a fallback blocking pass for records with missing/null key fields?
- Is my output schema exactly what's asked (column names, id types, delimiter)?

## If things go wrong
- Blocking step too slow / blocks too big → drop over-frequent tokens, add `max_block_size` guard (see toolkit), or subsample before embeddings.
- Embedding model fails to load (no internet) → fall back to token/TF-IDF/rapidfuzz-only pipeline; still solid without semantic blocking.
- Running low on time → submit the simplest working pipeline (token blocking + rapidfuzz scoring + truncate) rather than risk a half-finished advanced version. A working baseline beats a broken advanced pipeline.
