# Experiment 4b (exploratory, pre-declared): interpolated cross-encoder escalation

Written and hashed on 2026-10-31, before any held-out result for this backend was computed. This is an
exploratory follow-up to the pre-registered Experiment 4 (improvements.md §4.2), where escalation could not
help because the MiniLM-L6 reranker on its own scored below always-hybrid. Its outcome does not change the
Experiment 4 verdict. It is run once and reported whatever the result.

## Question

If the cross-encoder score is blended with the hybrid score instead of replacing it, does the blended
reranker beat always-hybrid, and does budgeted escalation then retain most of that gain (H3) at a small
fraction of its latency (H4)?

## Fixed design (no part of it is tuned on held-out queries)

- **Queries, folds, pools:** the five Experiment 4 datasets, 5-fold CV within each dataset, seed 42, α* per
  training fold, the hybrid top-50 pool, and the cached `cross-encoder/ms-marco-MiniLM-L-6-v2` scores:
  all identical to Experiment 4. Ranks 51–100 keep their hybrid order.
- **Blended backend:** within the top-50, score = λ·mm(CE) + (1 − λ)·mm(hybrid), where mm is min-max over
  the 50 candidates of that query. Exact ties keep hybrid order. λ ∈ {0.0, 0.1, …, 1.0} is chosen per
  outer fold as the value maximizing the training queries' mean nDCG@10 under always-blended reranking.
- **Escalation policies at budget B ∈ {5, 10, 15, 20, 30, 50}%,** applied per (dataset, fold) as in
  Experiment 4: predicted gain (LightGBM regressor of blended − hybrid nDCG@10 trained on the training
  fold, Experiment 4's features and hyperparameters), router confidence (Experiment 2 GBM router, lowest
  maximum probability first), random (expected value), oracle (upper bound).

## Tests

- **Prerequisite:** pooled always-blended − always-hybrid nDCG@10, paired bootstrap (10,000). Met iff the
  95% CI lower bound > 0. If it is not met, H3 is reported as not testable.
- **H3 (primary, pooled, B = 10%):** supported iff the prerequisite is met, predicted-gain@10% − always-hybrid
  has a CI lower bound > 0, and retained gain = (policy − hybrid) / (always-blended − hybrid) ≥ 50%.
- **Secondary:** predicted-gain and router-confidence vs random at each budget; per-dataset results;
  per-dataset always-blended − hybrid.
- **H4 (latency):** on NFCorpus, SciFact and FiQA, per-query latency is composed from the measured per-query
  component timings in `results/policy_latency/*_components.csv` (hybrid + gain-model features and
  inference for every query + cross-encoder for escalated queries), reported as mean and p95 next to
  always-hybrid and always-rerank.
