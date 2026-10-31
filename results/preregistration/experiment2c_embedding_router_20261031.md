# Experiment 2c (exploratory, pre-declared): query-embedding router

Written and hashed on 2026-10-31, before any held-out result for this router was computed. This is an
exploratory follow-up to the pre-registered Experiment 2 (improvements.md §3), whose GBM router on 18
hand-crafted features did not beat always-hybrid. Its outcome does not change the Experiment 2 verdict
on H1. It is run once and reported whatever the result.

## Question

Does adding the query's dense embedding (a latent representation, as in Query-Adaptive Hybrid Search)
to the 18 Experiment 2 features give a router that beats tuned always-hybrid on held-out queries?

## Fixed design (no part of it is tuned on held-out queries)

- **Queries, folds:** the five Experiment 2 datasets (NFCorpus, SciFact, FiQA, CQADupStack Unix, Quora
  subset), 5-fold CV within each dataset, seed 42: the same folds as Experiment 2.
- **Embedding:** `intfloat/e5-small-v2`, prefix `"query: "`, L2-normalized: the encoder the dense route
  already runs, so the feature adds no encoding latency.
- **Reduction:** PCA to 32 components, fitted on the outer training fold's queries only. 32 is fixed
  here and not tuned.
- **Features:** the 18 Experiment 2 features + the 32 PCA components (50 in all).
- **Router:** the Experiment 2 GBM router unchanged: `configs/router.yaml` hyperparameters, margin labels,
  regret weights, α* chosen on the training fold, ε chosen by inner 3-fold CV on the training fold
  (`labels.yaml` grid) using the 50 features. During ε tuning the inner folds reuse the outer-fold PCA
  (unsupervised, training queries only); the outer test fold is never seen.
- **Secondary model:** the logistic router on the same 50 features (standardized), same labels and ε.

## Tests

- **Primary:** pooled held-out nDCG@10, embedding GBM router − always-hybrid, paired bootstrap
  (10,000 resamples). Supported iff the 95% CI lower bound > 0.
- **Secondary (H1 as worded in the proposal):** per dataset, and for MRR@10 and MAP: embedding GBM
  router vs always-hybrid and vs RRF (k = 60, computed from the same BM25/dense runs). Embedding GBM
  vs random routing with the same route mix (expected value); embedding GBM vs the 18-feature GBM
  router refit on the same folds; the logistic variant vs always-hybrid. Routes are chosen with
  nDCG@10 labels; MRR@10 and MAP are measured on the chosen route's ranking.
- **Multiplicity:** per-dataset and per-metric tests are reported with raw CIs and a Holm-adjusted
  p-value over the 5 datasets × 3 metrics for router − hybrid. Only the primary test is confirmatory.
