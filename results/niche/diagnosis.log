# Router lock diagnosis and Track 2 separability

No retrieval re-run; all inputs are cached features/nDCG. Router = Experiment 2d cost-aware GBM (`scripts/run_experiment2d_cost_aware_router.py`), reproduced fold-by-fold with instrumentation.

## 1. Why the router locks onto hybrid (94–100%)

Real queries instrumented: 5163 (8 Track datasets + 3 BRIGHT); synthetic reference: 288.

**Mechanism (two stacked gates, both in `run_experiment2d_cost_aware_router.py`):**

1. **Fallback gate** — `choose_epsilon(...)`: `if fallback and utility[best] <= float(R["hybrid"].mean()): return HYBRID`. The GBM's inner 3-fold CV nDCG@10 must beat always-hybrid on the training folds, or the whole fold is served by hybrid. It fired in **41/55 real folds**, covering **4132/5163 (80.0%) real queries**. Mean inner margin (router − hybrid) on real folds: -0.0023.
2. **Label/margin gate** — `MarginLabeler(epsilon, margin_to="default")`: a query is labelled bm25/dense only if it beats hybrid by ≥ ε, else `hybrid`. Hybrid ties or wins most real queries, so in the 1031 real queries whose fold passed the gate the GBM still chose hybrid for 910 (88.3%).

The gate is not the root cause, it is the symptom: when it is removed (`ungated`), the GBM routes away from hybrid more often but gains nothing (column `ungated − hybrid`: negative on most datasets, ≤ +0.005 on the rest, against 2–7 points of oracle headroom), i.e. the 18 features do not predict *which* queries are in the oracle headroom. On synthetic data the same code never falls back.

| Dataset | n | Folds → fallback | Hybrid share (this run) | Hybrid share (published 2d) | Hybrid share if ungated | Inner margin | Non-hybrid train labels | Ungated − hybrid | Oracle − hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| arguana | 1401 | 4/5 | 97.1% | 97.1% | 87.6% | -0.0011 | 23.9% | -0.0022 | +0.0649 |
| bright-biology | 103 | 3/5 | 94.2% | 94.2% | 92.2% | -0.0017 | 9.5% | +0.0002 | +0.0239 |
| bright-economics | 103 | 4/5 | 96.1% | 96.1% | 80.6% | -0.0037 | 16.0% | +0.0005 | +0.0493 |
| bright-psychology | 101 | 3/5 | 95.0% | 95.0% | 82.2% | -0.0033 | 12.6% | -0.0023 | +0.0396 |
| cosqa | 500 | 4/5 | 99.0% | 99.0% | 96.2% | -0.0011 | 9.4% | -0.0022 | +0.0393 |
| cqadupstack-android | 699 | 4/5 | 98.0% | 98.0% | 91.0% | -0.0048 | 16.2% | -0.0018 | +0.0524 |
| cqadupstack-webmasters | 506 | 4/5 | 96.8% | 96.8% | 88.1% | -0.0028 | 16.0% | -0.0071 | +0.0402 |
| fiqa | 648 | 5/5 | 100.0% | 100.0% | 92.9% | -0.0027 | 16.5% | -0.0018 | +0.0436 |
| nfcorpus | 323 | 3/5 | 99.1% | 99.1% | 96.9% | +0.0000 | 10.2% | -0.0027 | +0.0288 |
| scifact | 300 | 2/5 | 90.7% | 90.7% | 86.0% | +0.0004 | 18.3% | +0.0050 | +0.0647 |
| synthetic-nfcorpus | 138 | 0/5 | 55.8% | — | 55.8% | +0.0750 | 30.6% | +0.0718 | +0.1161 |
| synthetic-scifact | 150 | 0/5 | 51.3% | — | 51.3% | +0.0829 | 26.8% | +0.1055 | +0.1242 |
| wands | 479 | 5/5 | 100.0% | 100.0% | 93.9% | -0.0047 | 14.0% | -0.0018 | +0.0387 |

**GBM TreeSHAP (ungated models, held-out queries).** `push to hybrid` = SHAP for the hybrid class minus the mean of the other classes (positive → pushes the decision to hybrid). Bias term push to hybrid: real +3.549, synthetic +2.348 (log-odds).

| Feature | Real mean abs SHAP | Real push to hybrid | Synthetic mean abs SHAP | Synthetic push to hybrid |
| :--- | :---: | :---: | :---: | :---: |
| dense_score_entropy | 0.650 | -0.002 | 0.400 | -0.039 |
| bm25_top1_score | 0.571 | +0.000 | 0.465 | +0.029 |
| bm25_score_entropy | 0.568 | +0.001 | 0.895 | -0.045 |
| dense_top1_top2_gap | 0.546 | -0.003 | 0.765 | -0.029 |
| dense_top1_score | 0.544 | -0.007 | 0.378 | +0.002 |
| mean_idf | 0.453 | -0.005 | 0.297 | +0.009 |
| dense_top10_query_term_coverage | 0.451 | -0.004 | 0.431 | -0.039 |
| bm25_top1_top2_gap | 0.444 | -0.002 | 0.742 | +0.000 |
| bm25_top10_query_term_coverage | 0.438 | +0.002 | 0.376 | +0.009 |
| bm25_top1_top2_ratio | 0.431 | -0.004 | 0.591 | -0.006 |
| jaccard_overlap_top20 | 0.392 | -0.002 | 0.701 | +0.005 |
| rank_correlation_spearman | 0.367 | -0.005 | 0.044 | +0.004 |
| max_idf | 0.346 | +0.004 | 0.157 | +0.041 |
| query_length | 0.336 | -0.000 | 1.510 | -0.102 |
| entity_density | 0.293 | -0.001 | 0.425 | -0.009 |
| rare_term_density | 0.253 | -0.004 | 0.674 | -0.011 |
| jaccard_overlap_top10 | 0.211 | -0.004 | 0.407 | +0.035 |
| oov_fraction | 0.106 | +0.001 | 0.009 | +0.000 |

## 2. What separates Track 2's 314 dense-favored vs 314 BM25-favored queries

Test A: Track 2 semantic (n=314) vs lexical (n=314); AUC > 0.5 means higher in semantic (dense-favored). Test B: all 628 polarized vs 628 neutral (|nDCG_dense − nDCG_bm25| < 0.10, random sample). `sep` = |AUC − 0.5| × 2 (0 = useless, 1 = perfect).

| Feature | Group | AUC A (sem vs lex) | sep A | AUC B (polar vs neutral) | Semantic P10/P50/P90 | Lexical P10/P50/P90 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| zdiff_entropy_dense_minus_bm25 | within-dataset z | 0.364 | 0.27 | 0.509 | -1.01 / -0.101 / 0.637 | -0.612 / 0.167 / 1.12 |
| z_dense_score_entropy | within-dataset z | 0.382 | 0.24 | 0.560 | -0.916 / 0.202 / 0.834 | -0.555 / 0.493 / 0.88 |
| idf_weighted_overlap_ratio | post-hoc (uses qrels) | 0.386 | 0.23 | 0.469 | 0.117 / 0.289 / 0.842 | 0.144 / 0.447 / 1 |
| dense_score_entropy | 18 features | 0.387 | 0.23 | 0.519 | 0.968 / 0.986 / 0.995 | 0.975 / 0.99 / 0.996 |
| zdiff_gap_bm25_minus_dense | within-dataset z | 0.394 | 0.21 | 0.488 | -1.76 / -0.118 / 1.01 | -1.14 / 0.152 / 1.93 |
| bm25_top1_top2_ratio | 18 features | 0.394 | 0.21 | 0.471 | 1 / 1.05 / 1.25 | 1 / 1.11 / 1.35 |
| z_bm25_top1_top2_gap | within-dataset z | 0.395 | 0.21 | 0.415 | -1.07 / -0.577 / 0.638 | -1.06 / -0.181 / 1.27 |
| bm25_top1_top2_gap | 18 features | 0.400 | 0.20 | 0.460 | 0.00644 / 0.101 / 0.329 | 0.00751 / 0.165 / 0.417 |
| zdiff_top1_bm25_minus_dense | within-dataset z | 0.424 | 0.15 | 0.530 | -1.33 / -0.0337 / 1.19 | -1.02 / 0.163 / 1.5 |
| z_bm25_top1_score | within-dataset z | 0.428 | 0.14 | 0.474 | -1.03 / -0.355 / 1 | -0.854 / -0.206 / 1.2 |
| raw_token_overlap | post-hoc (uses qrels) | 0.431 | 0.14 | 0.661 | 2 / 7 / 20 | 3 / 7.5 / 26.7 |
| bm25_score_entropy | 18 features | 0.565 | 0.13 | 0.492 | 0.953 / 0.984 / 0.996 | 0.944 / 0.981 / 0.995 |
| z_bm25_score_entropy | within-dataset z | 0.564 | 0.13 | 0.550 | -0.625 / 0.367 / 0.761 | -0.686 / 0.276 / 0.669 |
| dense_top1_score | 18 features | 0.547 | 0.09 | 0.402 | 0.85 / 0.874 / 0.902 | 0.844 / 0.872 / 0.895 |
| z_dense_top1_score | within-dataset z | 0.544 | 0.09 | 0.439 | -1.24 / -0.119 / 1.07 | -1.49 / -0.246 / 0.931 |
| dense_top1_top2_gap | 18 features | 0.542 | 0.08 | 0.465 | 0.00853 / 0.129 / 0.334 | 0.0134 / 0.114 / 0.28 |
| z_dense_top1_top2_gap | within-dataset z | 0.540 | 0.08 | 0.433 | -1.07 / -0.284 / 1.27 | -1.04 / -0.43 / 0.781 |
| entity_density | 18 features | 0.539 | 0.08 | 0.603 | 0 / 0.189 / 0.404 | 0 / 0.155 / 0.416 |
| max_idf | 18 features | 0.476 | 0.05 | 0.584 | 4.18 / 7.13 / 8.5 | 4.27 / 7.2 / 8.5 |
| bm25_top1_score | 18 features | 0.477 | 0.05 | 0.635 | 13.7 / 33.6 / 185 | 15.5 / 32.5 / 201 |
| bm25_top10_query_term_coverage | 18 features | 0.478 | 0.04 | 0.364 | 0.172 / 0.393 / 0.697 | 0.176 / 0.38 / 0.75 |
| mean_idf | 18 features | 0.487 | 0.03 | 0.475 | 2.4 / 3.23 / 4.12 | 2.53 / 3.22 / 4.44 |
| dense_top10_query_term_coverage | 18 features | 0.489 | 0.02 | 0.383 | 0.144 / 0.254 / 0.585 | 0.148 / 0.254 / 0.611 |
| rare_term_density | 18 features | 0.492 | 0.02 | 0.489 | 0 / 0.108 / 0.25 | 0 / 0.111 / 0.333 |
| jaccard_overlap_top10 | 18 features | 0.508 | 0.02 | 0.531 | 0.0526 / 0.176 / 0.429 | 0.0526 / 0.176 / 0.429 |
| jaccard_overlap_top20 | 18 features | 0.506 | 0.01 | 0.534 | 0.0526 / 0.176 / 0.429 | 0.0526 / 0.212 / 0.379 |
| oov_fraction | 18 features | 0.505 | 0.01 | 0.588 | 0 / 0 / 0.0355 | 0 / 0 / 0.0349 |
| rank_correlation_spearman | 18 features | 0.504 | 0.01 | 0.455 | -0.4 / 0.3 / 0.8 | -0.196 / 0.215 / 0.781 |
| query_length | 18 features | 0.503 | 0.01 | 0.651 | 5 / 15 / 220 | 5 / 14 / 260 |

### Why Track 1's rules picked Easy vs Hard instead

Pass rate of each Track 1 condition on the Track 2 buckets (a useful condition passes one bucket and fails the other):

| Track 1 condition | Track 2 semantic | Track 2 lexical | Neutral |
| :--- | :---: | :---: | :---: |
| sem: query_length ≥ 5 | 95.2% | 93.3% | 80.9% |
| sem: idf_overlap ≤ 0.35 | 56.4% | 45.5% | 44.7% |
| sem: jaccard10 ≤ 0.15 | 42.7% | 40.4% | 45.2% |
| lex: rare anchor | 56.7% | 58.9% | 55.9% |
| lex: idf_overlap ≥ 0.70 | 16.6% | 33.8% | 34.9% |
| lex: bm25 ratio ≥ P75 | 14.3% | 24.8% | 31.1% |
| lex: jaccard10 ≤ 0.20 | 55.7% | 59.2% | 62.3% |

`idf_weighted_overlap_ratio` measures whether the query's informative words are in the relevant document — that drives BM25 *and* the e5 dense model together (both find documents that share the query's rare words), so low overlap = both fail (hard), high overlap = both succeed (easy). It does not encode *which* retriever wins. See its AUC A above vs its AUC on hard-vs-easy below.

AUC of `idf_weighted_overlap_ratio` for easy (both > 0.7, n=1011) vs hard (both < 0.2, n=1207): **0.805**; for Track 2 semantic vs lexical: **0.386**.

## 3. Leave-one-dataset-out routing vs fixed hybrid (α = 0.6)

Routes: BM25 / dense / hybrid α=0.6. Labels: best route (ties → hybrid), sample weight = nDCG spread (as in Experiment 2). (a) = trained on the 288 synthetic queries only (2d recipe: eps=0); (b) = trained on the 7 other real datasets, evaluated on the 8th. `z-feat` = the 18 features z-scored within each dataset (unsupervised, uses the held-out dataset's own feature distribution, no labels). Cells: mean nDCG@10; pooled rows add Δ vs hybrid [95% CI].

| Dataset | n | Hybrid 0.6 | (a) Synthetic-trained 2d GBM | (a) Synthetic-trained depth-2 tree | (b) LODO 2d GBM recipe | (b) LODO depth-2 tree | (b) LODO L1 logistic | (b) LODO depth-2 tree (z-feat) | (b) LODO L1 logistic (z-feat) | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cosqa | 500 | 0.2595 | 0.2571 | 0.2531 | 0.2593 | 0.2595 | 0.2595 | 0.1793 | 0.2595 | 0.3057 |
| wands | 479 | 0.7651 | 0.7465 | 0.7630 | 0.7638 | 0.7650 | 0.7651 | 0.7651 | 0.7651 | 0.8038 |
| cqadupstack-android | 699 | 0.4693 | 0.4509 | 0.4598 | 0.4708 | 0.4693 | 0.4693 | 0.4624 | 0.4693 | 0.5216 |
| cqadupstack-webmasters | 506 | 0.3832 | 0.3702 | 0.3772 | 0.3821 | 0.3832 | 0.3832 | 0.3832 | 0.3832 | 0.4228 |
| nfcorpus | 323 | 0.3526 | 0.3446 | 0.3554 | 0.3497 | 0.3526 | 0.3526 | 0.3526 | 0.3526 | 0.3872 |
| scifact | 300 | 0.7348 | 0.7089 | 0.7212 | 0.7348 | 0.7348 | 0.7348 | 0.7120 | 0.7348 | 0.7922 |
| fiqa | 648 | 0.4029 | 0.3760 | 0.3867 | 0.3977 | 0.4029 | 0.4029 | 0.4029 | 0.4029 | 0.4465 |
| arguana | 1401 | 0.6078 | 0.5468 | 0.6020 | 0.6078 | 0.6078 | 0.6078 | 0.5971 | 0.6078 | 0.7055 |
| **pooled (4,856)** | 4856 | 0.5076 | 0.4782 | 0.5003 | 0.5067 | 0.5076 | 0.5076 | 0.4939 | 0.5076 | 0.5677 |
| **Track 2 subset (628)** | 628 | 0.7366 | 0.6768 | 0.7134 | 0.7374 | 0.7366 | 0.7366 | 0.7093 | 0.7366 | 0.9009 |

| Subset | (a) Synthetic-trained 2d GBM | (a) Synthetic-trained depth-2 tree | (b) LODO 2d GBM recipe | (b) LODO depth-2 tree | (b) LODO L1 logistic | (b) LODO depth-2 tree (z-feat) | (b) LODO L1 logistic (z-feat) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Δ vs hybrid, pooled | -0.0294 [-0.0337, -0.0252] | -0.0073 [-0.0098, -0.0048] | -0.0009 [-0.0019, -0.0000] | -0.0000 [-0.0002, +0.0001] | +0.0000 [+0.0000, +0.0000] | -0.0137 [-0.0165, -0.0110] | +0.0000 [+0.0000, +0.0000] |
| Δ vs hybrid, Track 2 | -0.0597 [-0.0804, -0.0393] | -0.0232 [-0.0361, -0.0111] | +0.0008 [-0.0033, +0.0050] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0272 [-0.0410, -0.0140] | +0.0000 [+0.0000, +0.0000] |

Route shares (pooled, BM25 / dense / hybrid):

- (a) Synthetic-trained 2d GBM: all 3.7% / 62.1% / 34.1%; Track 2 2.2% / 64.3% / 33.4%
- (a) Synthetic-trained depth-2 tree: all 4.1% / 17.5% / 78.4%; Track 2 2.5% / 14.6% / 82.8%
- (b) LODO 2d GBM recipe: all 1.7% / 2.1% / 96.2%; Track 2 0.3% / 1.8% / 97.9%
- (b) LODO depth-2 tree: all 0.1% / 0.0% / 99.9%; Track 2 0.0% / 0.0% / 100.0%
- (b) LODO L1 logistic: all 0.0% / 0.0% / 100.0%; Track 2 0.0% / 0.0% / 100.0%
- (b) LODO depth-2 tree (z-feat): all 12.7% / 1.9% / 85.4%; Track 2 9.4% / 2.7% / 87.9%
- (b) LODO L1 logistic (z-feat): all 0.0% / 0.0% / 100.0%; Track 2 0.0% / 0.0% / 100.0%

LODO 2d-recipe gate per held-out dataset: cosqa: eps=0.1, wands: eps=0.1, cqadupstack-android: eps=0.05, cqadupstack-webmasters: eps=0.05, nfcorpus: eps=0.1, scifact: eps=0.1, fiqa: eps=0.1, arguana: fallback

Example LODO depth-2 tree (trained without arguana):

```
|--- bm25_top10_query_term_coverage <= 0.61
|   |--- bm25_score_entropy <= 0.99
|   |   |--- class: hybrid
|   |--- bm25_score_entropy >  0.99
|   |   |--- class: hybrid
|--- bm25_top10_query_term_coverage >  0.61
|   |--- mean_idf <= 3.09
|   |   |--- class: hybrid
|   |--- mean_idf >  3.09
|   |   |--- class: hybrid
```

## 4. Experiment 2d on Track 2

**(i) Published Experiment 2d run, restricted to Track 2 queries** (router trained on each dataset's full query set, n = 628): cost-aware − hybrid = -0.0021 [-0.0080, +0.0039]; hybrid share 97.6%; oracle − hybrid = +0.1704 [+0.1538, +0.1879].

**(ii) Stress test: 2d recipe trained by 5-fold CV on the 628 Track 2 queries themselves** (optimistic: the set was selected on nDCG, so every query has a clear winner): router − hybrid = +0.0000 [+0.0000, +0.0000]; oracle − hybrid = +0.1615 [+0.1461, +0.1774]; route share BM25 / dense / hybrid = 0.0% / 0.0% / 100.0%; gate outcome across the 5 folds: fallback.

