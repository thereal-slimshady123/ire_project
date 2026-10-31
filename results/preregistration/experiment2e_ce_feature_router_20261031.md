# Experiment 2e (exploratory, pre-declared): router with document-aware cross-encoder features

Written and hashed on 2026-10-31, before any held-out result for this router was computed. It is
exploratory and not blind: it was designed after Experiments 2, 2c and 2d and a held-out diagnostic on the
same folds showed that the 18 features plus the query embedding predict "BM25 / dense beats hybrid" with
AUC ≈ 0.67, but with only 29–46% precision among non-tied queries, against the 56–61% needed to break even.
It is run once and reported whatever the result. It does not change the Experiment 2 verdict on H1.

## Question

The existing features describe the query and the score distributions, not whether the retrieved
documents are relevant. Does adding a cheap relevance check of each route's top documents give a router
that beats tuned always-hybrid on held-out queries?

## Fixed design

- **Queries, folds, α\*:** as Experiments 2 and 2d (five datasets, 5-fold CV within each dataset,
  seed 42, α\* chosen on the training fold).
- **New features (10):** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores, taken from the cached Experiment 4
  scores (`results/runs/{dataset}/ce_*.tsv`), of the top-3 documents of each route: BM25, dense, and hybrid
  at the fold's α\*. Per route: the top-1 document's score and the mean over the top 3 (6 features), plus
  the differences BM25 − hybrid and dense − hybrid for both (4 features). Pairs absent from the cache
  (17 of 17,618) are missing values, which LightGBM handles natively; nothing is re-scored.
- **Primary router:** the Experiment 2d cost-aware GBM unchanged (no class balancing, margins against
  hybrid, inner-CV ε with an always-hybrid fallback) on the 18 Experiment 2 features + the 10 new ones.
- **Secondary:** (S1) the same router on the 10 new features only; (S2) the Experiment 2b advantage router
  (`configs/router.yaml` `advantage` block, unchanged) on the 28 features.
- **Cost:** a query needs the union of the three top-3 lists scored by the cross-encoder, 5.3 pairs per
  query on average. Its latency is estimated from the measured Experiment 3.5 cross-encoder cost per
  pair (50-pair rerank time / 50) and reported next to the quality result. It counts against H4.

## Tests

- **Primary (superiority):** pooled held-out nDCG@10, primary router − always-hybrid, paired bootstrap
  (10,000 resamples). Supported iff the 95% CI lower bound > 0.
- **Co-primary (non-inferiority):** the same difference with margin δ = 0.005; met iff the lower bound
  > −0.005.
- **Secondary:** per dataset; MRR@10 and MAP on the chosen route; S1 and S2 vs always-hybrid; primary vs
  random routing at the same route mix; primary vs the Experiment 2d router (same folds). Raw CIs only.
