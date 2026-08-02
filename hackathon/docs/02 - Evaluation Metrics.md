# Evaluation Metrics for Blocking

These are the standard metrics for evaluating a blocking/candidate-pair-generation step. The hackathon is almost certainly graded with one or more of these — know them cold before Sunday. See [[01 - Blocking Strategies Cheat Sheet]] for how to optimize toward them and [[03 - Python Blocking Toolkit]] for ready-made functions.

Let:
- `M` = set of all true matching pairs that exist between the datasets (ground truth)
- `C` = set of candidate pairs you output
- `|C ∩ M|` = true matches you actually included in your output

These correspond to the standard entity-resolution literature terms (Papadakis et al. survey; see Sources below), where blocking output is often written `B`: PC(B) = |matches in B| / |all matches|, PQ(B) = |matches in B| / |all pairs in B|, RR(B) = 1 − |pairs in B| / |all possible pairs|. Below, `C` plays the role of `B`.

## Pairs Completeness (PC) — this is almost certainly the headline metric
```
PC = |C ∩ M| / |M|
```
Fraction of *all true matches* that survived into your candidate set. This is exactly "how many matching pairs did you capture" — the task description ("as many matching pairs as possible") is describing PC. **Maximize this first.**

## Pairs Quality (PQ) — precision of the candidate set
```
PQ = |C ∩ M| / |C|
```
Fraction of your output pairs that are actually true matches. High PQ with low PC = you were too conservative. If the brief only limits *size* (not requiring a minimum PQ), PQ mostly matters as a tiebreaker or if the grading formula combines both (e.g., F1 of PC/PQ).

## Reduction Ratio (RR)
```
RR = 1 - |C| / (|D1| * |D2|)
```
How much you cut down from the full cross product. Since your `|C|` is capped by the size limit, RR will naturally be close to 1 and is not something you need to actively optimize — it falls out of the budget constraint.

## F1 over PC/PQ
If grading combines recall and precision:
```
F1 = 2 * PC * PQ / (PC + PQ)
```
Same harmonic-mean logic as standard F1. If this is the metric, don't just fill the entire budget with low-confidence pairs — a smaller, high-precision set can beat a maxed-out noisy one. Check the actual formula in Saturday's document before assuming pure PC.

## Practical checklist for Sunday
1. **Find out exactly which formula is used** (PC alone? PC capped by budget? F1? weighted?) — this changes strategy materially (dump-everything-up-to-K vs. threshold-and-stop-early).
2. **Compute PC/PQ locally** on any labeled sample/dev set you're given, before submitting — don't submit blind.
3. If only PC (with a hard cap) is used: your only job is memory recall — cast the widest net your budget allows, rank by best signal, take top-K. Precision doesn't matter beyond fitting inside K.
4. If PQ/F1 also matters: build a validation loop — try a few thresholds/top-K cutoffs below the max size, measure PC and PQ on a held-out labeled slice, pick the sweet spot.
5. Watch for a **separate runtime budget** — some blocking hackathons also grade wall-clock time or require the whole pipeline to finish within a time limit; if so, prefer fast blocking (token/q-gram/sorted neighborhood) over expensive embedding search on large datasets, or subsample before embedding.

## Sources
- [A Survey of Blocking and Filtering Techniques for Entity Resolution (arXiv 1905.06167)](https://arxiv.org/pdf/1905.06167) — formal PC/PQ/RR definitions.
- [Entity Resolution Benchmarking: Best Datasets + Metrics Beyond F1](https://www.minimalistinnovation.com/post/benchmarking-datasets-metrics-entity-resolution) — accessible overview of ER metrics beyond plain F1.
