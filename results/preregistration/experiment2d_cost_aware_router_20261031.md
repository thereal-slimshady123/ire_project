# Experiment 2d (exploratory, pre-declared): cost-aware router

Written and hashed on 2026-10-31, before this script was run. It is **not blind**: a diagnostic of the
Experiment 2 router on the same folds was already seen (below), and this design was chosen in response to
it. It is run once and reported whatever the result. It does not change the pre-registered Experiment 2
verdict on H1.

## Why

The pre-registered Experiment 2 GBM router lost to tuned always-hybrid (pooled −0.0061 [−0.0091,
−0.0032]). Diagnosis on its held-out predictions:

- Leaving hybrid is costly and asymmetric: BM25 beats hybrid on 7% of queries (mean +0.20) but loses on
  28% (mean −0.31); dense wins 12% (+0.17) and loses 18% (−0.22).
- The router was trained with `class_weight: "balanced"`, which multiplies the weight of the rare BM25 /
  dense labels by roughly 6–15×, so it predicts them far more often than their expected value justifies.
  Of 190 held-out queries it sent to BM25, 16 improved and 63 got worse (net −14.8 nDCG points).
- The margin labeler compares the top route with the runner-up, not with hybrid, so a query on which
  BM25 and dense are close to each other but both far above hybrid is labelled "hybrid" (1–2 queries per
  dataset).
- Scratch runs on the same folds: the same GBM without class balancing scored −0.0008 [−0.0023, +0.0008];
  routers that fall back to hybrid unless inner CV shows a gain scored ≈ 0.0000.

## Fixed design

- **Queries, folds, α\*:** as Experiment 2 (five datasets, 5-fold CV within each dataset, seed 42, α\*
  chosen on the training fold).
- **Primary router ("cost-aware GBM"):** the Experiment 2 LightGBM classifier and hyperparameters
  (`configs/router.yaml`) with three changes:
  1. `class_weight = None` (no class balancing); regret-spread sample weights as in Experiment 2;
  2. labels from `MarginLabeler(ε, margin_to="default")`: a non-hybrid label only if that route beats
     hybrid by ≥ ε;
  3. **hybrid fallback:** ε is chosen by Experiment 2's inner 3-fold CV over the training fold from the
     `labels.yaml` grid, with "always hybrid" added as a candidate. A router is used only if its inner-CV
     routed nDCG@10 is strictly higher than always-hybrid's on the same inner folds; otherwise the fold
     routes every query to hybrid.
- **Secondary routers** (same folds, α\*, inner CV and fallback unless stated):
  - S1, **blended target** (`improvements.md` §2.3, `problems.md` item 5): labels and inner-CV utility
    from 0.6·nDCG@10 + 0.4·MRR@10 per route, computed from the saved runs;
  - S2, **+ query embedding**: the 18 features + PCA-32 of the e5-small-v2 query embedding (the
    Experiment 2c features, PCA fitted on the outer training fold);
  - S3, **no fallback**: the primary router without the always-hybrid candidate (isolates the effect of
    removing class balancing and fixing the margin);
  - S4, **Experiment 2 router refit** (balanced classes, runner-up margin) as a reference.

## Tests

- **Primary (superiority, the H1 test):** pooled held-out nDCG@10, cost-aware GBM − always-hybrid,
  paired bootstrap (10,000 resamples). Supported iff the 95% CI lower bound > 0.
- **Co-primary (non-inferiority):** the same difference with margin δ = 0.005 nDCG@10. Non-inferior iff
  the 95% CI lower bound > −0.005.
- **Secondary:** per dataset; MRR@10 and MAP on the chosen route; each secondary router vs
  always-hybrid; primary vs random routing at the same route mix (expected value); oracle bound. Raw CIs
  only, no confirmatory claims.
