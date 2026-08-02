# Official Task Instructions (Confirmed)

This is the real brief — it replaces all the "figure this out Sunday" placeholders in [[04 - Hackathon Game Plan]]. Read this first, then jump straight to building; [[01 - Blocking Strategies Cheat Sheet]], [[02 - Evaluation Metrics]], and [[03 - Python Blocking Toolkit]] still apply, with the corrections noted below.

## The confirmed facts (resolves the open questions from the game plan)

| Question from the game plan | Answer |
|---|---|
| Exact metric? | **Recall (Pairs Completeness) only**, computed against a hidden ground-truth set. No PQ/precision term in the grade. |
| Exact size limit `K`? | **2000 pairs, hard cap.** `validate_candidate_set` asserts `len(candidate_set) <= 2000` — don't touch `max_pairs`. |
| Output format? | A **Python `set`** of `(id_a, id_b)` tuples — not a list, not a DataFrame, not sorted pairs. `id_a` must come from TableA, `id_b` from TableB, in that order. |
| Given labeled data? | `100_matches.pkl` — 100 known true matching pairs, same `(id_a, id_b)` format. |
| Files to submit | `<sid>.zip` containing (1) your full code (script or notebook) and (2) `<sid>.pkl` (the candidate set, produced by `save_submission`). |

## Grading formula (this changes strategy vs. a pure "maximize recall" framing)

```
if recall < 0.6:
    grade = recall * 150       # e.g. recall 0.5 -> grade 75
else:
    grade = max(100 - rank, 90)  # rank 0 = best in cohort, rank 1 = second, ...
```

Implications:
- **Below 0.6 recall, every extra bit of recall is worth a lot** (linear, ×150 — e.g. going from 0.55→0.60 alone is +7.5 points). Getting over the 0.6 line is the single highest-leverage thing you can do.
- **Above 0.6, the grade compresses to a tight 90–100 band** based on relative rank, not absolute recall. Once you're comfortably over 0.6, squeezing out the last few % of recall only matters if you're near the top of the pack — a big, robust pipeline that reliably clears 0.6 is worth more than a fragile one aiming for the theoretical ceiling.
- Practically: **treat 0.6 recall as a hard milestone to validate you've cleared before doing anything else**, then spend remaining time pushing recall higher for rank, but don't panic-tune at the very end in a way that risks dropping back under 0.6.

## Critical gotcha: the 100 known matches are dual-purpose, and the exclusion is exact-tuple, not "similar"

- **You may use `100_matches.pkl` freely** for feature engineering, choosing a similarity threshold, training a small classifier/reranker, or validating recall locally (it's your only labeled proxy for the hidden test set).
- **But none of those 100 exact pairs may appear in your final candidate set.** This is checked as exact `(id_a, id_b)` membership, not "don't match this product" — so after building your candidate set, explicitly subtract the 100 pairs:
  ```python
  with open("100_matches.pkl", "rb") as f:
      known_matches = pickle.load(f)   # set of (id_a, id_b)

  candidate_set -= known_matches
  ```
- Do this **right before validation/saving**, not earlier — if you filter them out too early you lose them as a validation signal while you're still tuning the pipeline. Filter-last, exclude-once.
- Since removing 100 pairs frees up 100 slots under the 2000 cap, re-top-up from your ranked leftover pool after subtracting, rather than just submitting 1900 pairs.

## Second gotcha: do NOT `sorted()` your pair tuples here

[[01 - Blocking Strategies Cheat Sheet]] and [[03 - Python Blocking Toolkit]] both suggest `tuple(sorted((id1, id2)))` for deduping pairs — that advice is for single-table dedup and **does not apply directly here**. In this task:
- `id_a` must always be the TableA identifier and `id_b` always the TableB identifier — `validate_candidate_set` checks `ida in tableA_ids` and `idb in tableB_ids` positionally.
- If TableA and TableB ids share a namespace (e.g. both are small integers, or both start at 0), `sorted()` can silently swap which one lands in position 0 vs 1, flipping a valid pair into an invalid one that fails validation (or silently passes validation but scores against the wrong table).
- **Fix:** always construct pairs as `(id_from_tableA, id_from_tableB)` explicitly by keeping track of provenance (which table a candidate came from) instead of sorting. Dedupe with a plain `set` add — since the tuple is already canonical (A-id, B-id), no sorting is needed or safe.
- Check this once early: `set(tableA['id']) & set(tableB['id'])` — if non-empty, this gotcha is live and you must be careful; if empty, the risk is lower but the discipline is still good practice.

## Recommended local validation loop (since you have labels)

```python
import pickle
import pandas as pd
from helpers import validate_candidate_set, save_submission

tableA = pd.read_csv("TableA.csv")
tableB = pd.read_csv("TableB.csv")
tableA_ids, tableB_ids = set(tableA["id"]), set(tableB["id"])

with open("100_matches.pkl", "rb") as f:
    known_matches = pickle.load(f)

def local_recall(candidate_set):
    # proxy for the hidden metric: what fraction of the 100 known pairs would you have kept?
    return len(candidate_set & known_matches) / len(known_matches)

# ... build candidate_set via blocking + scoring ...
print("proxy recall before exclusion:", local_recall(candidate_set))

candidate_set -= known_matches   # required: none of the 100 may appear in the final output

validate_candidate_set(candidate_set, tableA_ids, tableB_ids)
save_submission(candidate_set, "<sid>")
```

This proxy recall is optimistic (you tuned against these exact 100 pairs), so treat it as an upper bound, not a guarantee — but it's the best signal available before submission. See `solution_starter.py` for the full end-to-end script that implements this checklist, including the exclude-then-top-up step.

## Submission checklist
- [ ] Zip is named `<sid>.zip` (your actual ID — will be added right before submitting).
- [ ] Zip contains the code (script/notebook) **and** `<sid>.pkl`.
- [ ] `<sid>.pkl` was produced via `save_submission`, not a manual `pickle.dump`.
- [ ] `validate_candidate_set` was called and passed before saving.
- [ ] None of the 100 given matches are in the final set (`candidate_set & known_matches == set()`).
- [ ] `len(candidate_set) <= 2000`.
- [ ] `max_pairs` parameter in `validate_candidate_set` was left untouched at 2000.
