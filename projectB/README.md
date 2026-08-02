# Project B — Active Learning & Graph Neural Networks

## Team
Guy Dukas, Gil Caplan, Murad Rahimli

## [Video Presentation — Click Here](https://youtu.be/hNOz35dc70M)

This repo includes the instructor-given assignment harness (config, evaluation script,
entry point, framework utilities, and the blank starter files) alongside our completed
solution, so anyone can set it up and attempt the assignment themselves. See
[Repo layout](#repo-layout) below. The actual datasets are **not** included (see that
section for why) — you'll need your own copy of the course data to run anything.

---

## Section A — Active Learning for Employee Attrition

**Problem.** Predict employee attrition with a fixed Random Forest classifier, starting
from 500 free labeled samples per seed and an oracle budget of 5,000 queries, under a
60-second runtime limit per seed. Scored on F1 of the minority "Left" class at the
model's fixed 0.5 threshold, averaged over 3 seeds.

**Approach** ([`strategy.py`](strategy.py)):
1. **Margin sampling** — 9 rounds × 500 queries: retrain, then query the pool samples
   with the smallest `|P(Left) − 0.5|` (closest to the decision boundary).
2. **Positive hunt** — the final 500 queries go to the highest `P(Left)` samples instead,
   adding real minority-class positives rather than more boundary cases.
3. **Minority oversampling** — duplicate every queried "Left" row ×3 before the final
   fit only (not inside the query loop, where it would skew the probabilities margin
   sampling depends on). This shifts the fixed 0.5 threshold's effective recall/precision
   trade-off; F1 peaks at ×3 oversampling.

**Results.**

| Seed 1 | Seed 2 | Seed 3 | Mean |
|--------|--------|--------|------|
| 0.6383 | 0.6493 | 0.6535 | **0.6470** |

Runtime: 20–30s of the 60s budget per seed. Ablation: no-query baseline 0.407 → random
sampling (4,500 queries) 0.551 → margin sampling 0.589 → + oversampling ×3 0.642 → +
positive hunt 0.647.

We also tried committee-based querying, SMOTE-style synthetic positives, and
self-distillation; none beat this pipeline. Diagnosis: after active querying the training
set is ~51% positive vs. ~33% in the pool, so validating hyperparameters on queried data
picks the wrong values — this ruled out several fancier variants. See
[`presentation/presentation.pdf`](presentation/presentation.pdf) for the full writeup and
figures.

---

## Section B — Graph Neural Network Node Classification

**Problem.** Given a citation-style graph (node features, edges, subject labels), train a
GNN to classify node subjects.

**Approach** ([`gnn.py`](gnn.py)): a 2-layer GraphSAGE (`SAGEConv`) network with a ReLU
+ dropout (p=0.5) between layers, trained with cross-entropy on the labeled training
mask. The best checkpoint (by validation accuracy) is saved during training and used for
the final test-set evaluation.

---

## Repo layout

```
strategy.py          our completed Section A solution
gnn.py                our completed Section B solution
parta/
  constants.yaml      instructor config (budget, seeds, RF hyperparameters)
  evaluation.py        instructor evaluation harness (do not modify)
  run.py                instructor entry point (do not modify) — imports strategy.py
  utils.py              instructor framework utilities (do not modify) — oracle, data loading
  strategy_skeleton.py  the blank starter file as given, with TODOs
partb/
  evaluation.py, run.py, utils.py   instructor harness, same rules as above
  gnn_skeleton.py                    the blank starter file as given, with TODOs
```

To attempt either section yourself: copy `strategy_skeleton.py` → `strategy.py` (or
`gnn_skeleton.py` → `gnn.py`) into the matching `parta/`/`partb/` folder, fill in the
TODOs, and run `evaluation.py`.

**Data is not included.** `parta/data/` (employee attrition pool + per-seed splits) and
`partb/data/` (citation graph nodes/edges/labels) are course-provided datasets and are
gitignored here — you'll need the course's copy to actually run the harness. Note also
that `partb/`'s `constants.yaml` (referenced by its `utils.py`) wasn't available in our
local copy, so it isn't included; you'll need the course's version of that file too.
