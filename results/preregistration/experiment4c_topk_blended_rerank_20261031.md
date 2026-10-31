# Experiment 4c (exploratory, pre-declared): blended reranking of every query's top-k

Written and hashed on 2026-10-31, before any held-out result for this policy was computed. It is
exploratory and not blind: it was designed after Experiment 4b showed that the blended cross-encoder
backend beats always-hybrid (+0.0067 [+0.0035, +0.0098]) but that budgeted per-query escalation retains
only 16% of that gain at a 10% budget, because no selector identifies the queries that benefit. It is run
once and reported whatever the result. It does not change the Experiment 4 / 4b verdicts on H3.

## Question and link to the proposal

The proposal's H3 states the budget as a share of queries ("≤B% of queries escalated"). Experiment 4b
showed the per-query selection is the weak step. This experiment spends the same cross-encoder budget
differently: instead of 50 pairs for 10% of queries (5 pairs per query on average), it reranks the
top-k of **every** query. The question is H3 with the budget measured in cross-encoder pairs:

**H3′:** With a cross-encoder budget of 5 pairs per query (the average cost of escalating 10% of queries
over a 50-document pool), reranking retains most (≥ 50%) of the always-blended gain and beats always-hybrid.

## Fixed design (no part of it is tuned on held-out queries)

- **Queries, folds, pools, scores:** as Experiment 4b: five datasets, 5-fold CV within each dataset,
  seed 42, α\* per training fold, hybrid candidates, cached `cross-encoder/ms-marco-MiniLM-L-6-v2` scores
  (`results/runs/{dataset}/ce_*.tsv`; the model is never loaded).
- **Policy top-k:** within the hybrid top-k, score = λ·mm(CE) + (1 − λ)·mm(hybrid), mm = min-max over
  those k candidates; exact ties keep hybrid order; ranks below k keep hybrid order. λ ∈ {0.0, 0.1, …, 1.0}
  is chosen per outer fold and per k as the value maximizing the training queries' mean nDCG@10.
- **k ∈ {3, 5, 10, 20, 50}.** k = 5 is primary (matched cost). k = 50 is always-blended, identical to
  Experiment 4b's backend, and is the reference for retained gain.

## Tests

- **H3′ (primary, pooled, k = 5):** supported iff top-5 − always-hybrid nDCG@10 has a 95% paired bootstrap
  CI (10,000 resamples) lower bound > 0 **and** retained gain = (top-5 − hybrid) / (always-blended − hybrid)
  ≥ 50%. If always-blended − hybrid is not > 0 (CI lower bound), H3′ is reported as not testable.
- **Matched-cost comparison (secondary):** top-5 vs Experiment 4b's predicted-gain and router-confidence
  escalation at B = 10% (same average CE pairs per query), paired per query.
- **Secondary:** every k vs always-hybrid; per dataset; MRR@10 is not computed (nDCG@10 only, as in 4b).
- **H4 (latency):** on NFCorpus, SciFact and FiQA, per-query latency = measured hybrid path + k × the
  query's measured cross-encoder time per pair (50-pair rerank / 50), mean and p95, next to always-hybrid
  and the 4b escalation policies at 10%. Batching overhead makes the per-pair cost of a 5-pair batch higher
  than 1/50 of a 50-pair batch, so these estimates are lower bounds and are labelled as such.
