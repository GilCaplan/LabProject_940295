# Section A video script — 6 slides, 3 speakers, ~95 seconds

Target pace: ~2.5 words/sec. Each speaker has ~32s. Rehearse with a timer —
the 10% time-management points are the cheapest points in the rubric.

---

## Slide 1 — Problem & constraints (Speaker 1, ~14s)

**On slide:** task diagram: pool (~14,900 unlabeled) → oracle (5,000 budget) →
fixed RandomForest → F1(Left). Constraints box: 500 free labels/seed, 60s/seed.

**Script (S1):**
> "Our task: predict employee attrition with a fixed Random Forest, starting
> from 500 labeled samples and an oracle budget of 5,000 queries, under 60
> seconds per seed. The metric is F1 on the minority 'Left' class, at the
> model's fixed 0.5 threshold."

*(43 words)*

---

## Slide 2 — Our pipeline (Speaker 1, ~18s)

**On slide:** 3-step flow: ① margin sampling, 9 rounds × 500
(query |P(Left)−0.5| smallest) → ② positive hunt, last 500 queries = top
P(Left) → ③ duplicate Left rows ×3, final refit.

**Script (S1):**
> "Our pipeline has three parts. First, nine rounds of margin sampling:
> retrain, then query the 500 pool samples the model is most uncertain about.
> Second, the last 500 queries hunt likely positives — highest P-of-Left —
> adding real minority samples. Third, we duplicate every positive training
> row three times before the final fit."

*(52 words)*

---

## Slide 3 — Ablation plot: what each part was worth (Speaker 2, ~17s)

**On slide:** bar chart, mean F1 over the 3 seeds:
baseline (no queries) 0.407 → random 4,500 0.551 → margin 4,500 0.589 →
+ oversample ×3 0.642 → + positive hunt 0.647. Dashed line at 0.55
(guarantee threshold). Value labels on every bar so the figure self-interprets.

**Script (S2):**
> "This plot shows what each component was worth. The no-query baseline scores
> 0.41. Spending the same budget randomly reaches 0.55; margin sampling, 0.59.
> Oversampling the positives adds five points — our single biggest win — and
> the positive hunt brings us to 0.647."

*(43 words)*

---

## Slide 4 — Tradeoff plot: the oversampling factor (Speaker 2, ~16s)

**On slide:** line plot, mean F1 vs oversample factor:
×1 0.589, ×2 0.636, **×3 0.642 (peak, marked)**, ×4 0.641, ×5 0.637, ×6 0.635.
Annotate: "recall ↑ / precision ↓ — F1 peaks at ×3". Caption: "RF
hyperparameters and encoding are fixed by the framework — the training
distribution is our only tuning lever."

**Script (S2):**
> "Why duplicate exactly three times? F1 rises, then falls: too little leaves
> the minority class under-predicted; too much trades precision past the F1
> optimum. With the model, its hyperparameters, and the threshold all fixed,
> duplication is effectively our only tuning dial — and times-three is its peak."

*(46 words)*

---

## Slide 5 — What didn't work (Speaker 3, ~19s)

**On slide:** dot/strip plot of the rejected variants' mean F1 (0.50–0.646),
grouped by axis: query selection / batch & budget / augmentation / validation
& calibration / distillation. Horizontal line at 0.647 labeled "ours".
Callout box: "queried training set = 51% positive vs ~33% in pool →
self-validation picks wrong hyperparameters".

**Script (S3):**
> "We tried many alternatives — committee-based querying, SMOTE-style
> synthetic positives, self-distillation, and more. Every dot here fell short
> of the simple pipeline. One key finding explains several failures: after
> active querying, our training set is fifty-one percent positive versus a
> third in the pool, so validating on queried data picks the wrong
> hyperparameters."

*(54 words)*

---

## Slide 6 — Results & conclusion (Speaker 3, ~12s)

**On slide:** table: seed 1 / 2 / 3 = 0.6383 / 0.6493 / 0.6535, mean **0.6470**;
runtime 20–30s of 60s. One line: "top alternatives within seed noise (±0.005–0.01)
→ stopped to avoid overfitting local test sets."

**Script (S3):**
> "Final result: mean F1 0.647, all three seeds above 0.638, well inside the
> runtime limit. With every alternative inside seed-level noise, we stopped
> rather than overfit the local test sets. Thank you."

*(34 words)*

---

## Totals

| Speaker | Slides | Words | ~Time |
|---|---|---|---|
| S1 | 1–2 | 95 | ~34s |
| S2 | 3–4 | 89 | ~33s |
| S3 | 5–6 | 88 | ~32s |
| **Total** | 6 | **272** | **~99s** |

This is right at the limit. If a rehearsal runs long, cut in this order:
1. Slide 1: drop "at the model's fixed 0.5 threshold" (−7 words)
2. Slide 5: drop "SMOTE-style synthetic positives," (−4 words)
3. Slide 6: drop "well inside the runtime limit" (−5 words)

## Rubric mapping
- **Empirical evaluation (7.5 pts):** slides 3–5 are all plots with legends,
  value labels, and a stated takeaway per figure — the ablation, the tradeoff
  curve, and the negative results with an explanation (the 51%-positive
  finding is the "critical discussion" they ask for).
- **Clarity (3 pts):** one idea per slide; pipeline stated in 3 steps.
- **Participation (3 pts):** exactly 2 slides per speaker; S2 and S3 own the
  quantitative content, so all members speak "meaningfully".
- **Time (1.5 pts):** script is 99s at 2.7 w/s — rehearse at least twice.
