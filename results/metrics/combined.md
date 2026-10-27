# Combined held-out evaluation

Primary metric: mean nDCG@10.
Baseline metrics come from BM25/dense/hybrid/RRF runs; adaptive systems come from held-out Experiment 2–4 artifacts. Matched-pool reranker results are included when a benchmark summary is supplied. Missing systems are listed as unavailable.
95% CIs are bootstrap intervals over held-out queries where per-query values are saved. Latency is measured per query (results/policy_latency) where the dataset is available locally.

| Dataset | System | nDCG@10 | 95% CI | Mean / p95 ms | Queries |
| :--- | :--- | ---: | :---: | ---: | ---: |
| cqadupstack-unix | always_hybrid | 0.3764 | [0.3517, 0.4015] | — | 1072 |
| cqadupstack-unix | best_single | 0.3597 | [0.3357, 0.3840] | — | 1072 |
| cqadupstack-unix | gbm_router | 0.3659 | [0.3416, 0.3910] | — | 1072 |
| cqadupstack-unix | logistic_router | 0.3515 | [0.3272, 0.3768] | — | 1072 |
| cqadupstack-unix | random_route | 0.3656 | [0.3418, 0.3902] | — | 1072 |
| cqadupstack-unix | oracle_route | 0.4182 | [0.3930, 0.4440] | — | 1072 |
| cqadupstack-unix | always_rerank_minilm_l6 | 0.3683 | [0.3436, 0.3936] | — | 1072 |
| cqadupstack-unix | predicted_gain_escalation_10pct | 0.3786 | [0.3538, 0.4037] | — | 1072 |
| cqadupstack-unix | confidence_escalation_10pct | 0.3798 | [0.3549, 0.4053] | — | 1072 |
| cqadupstack-unix | random_escalation_10pct | 0.3755 | [0.3510, 0.4003] | — | 1072 |
| cqadupstack-unix | oracle_escalation_10pct | 0.4176 | [0.3924, 0.4435] | — | 1072 |
| cqadupstack-unix | router_tau_escalation_10pct | 0.3693 | [0.3444, 0.3947] | — | 1072 |
| cqadupstack-unix | router_random_escalation_10pct | 0.3676 | [0.3434, 0.3923] | — | 1072 |
| cqadupstack-unix | hybrid_tau_escalation_10pct | 0.3758 | [0.3507, 0.4013] | — | 1072 |
| fiqa | always_bm25 | 0.2528 | — | 1.1 / 3.1 | 648 |
| fiqa | always_dense | 0.3747 | — | — | 648 |
| fiqa | always_hybrid | 0.4029 | [0.3751, 0.4315] | 27.1 / 103.9 | 648 |
| fiqa | rrf | 0.3741 | — | — | 648 |
| fiqa | best_single | 0.3747 | [0.3472, 0.4030] | — | 648 |
| fiqa | gbm_router | 0.3925 | [0.3646, 0.4215] | 39.8 / 153.5 | 648 |
| fiqa | logistic_router | 0.3542 | [0.3261, 0.3826] | — | 648 |
| fiqa | random_route | 0.3889 | [0.3621, 0.4167] | — | 648 |
| fiqa | oracle_route | 0.4465 | [0.4187, 0.4754] | — | 648 |
| fiqa | always_rerank_minilm_l6 | 0.3759 | [0.3490, 0.4037] | 367.8 / 505.6 | 648 |
| fiqa | predicted_gain_escalation_10pct | 0.4010 | [0.3728, 0.4301] | 76.5 / 352.2 | 648 |
| fiqa | confidence_escalation_10pct | 0.4042 | [0.3763, 0.4330] | 74.1 / 353.8 | 648 |
| fiqa | random_escalation_10pct | 0.4002 | [0.3729, 0.4283] | 61.3 / 346.9 | 648 |
| fiqa | oracle_escalation_10pct | 0.4399 | [0.4117, 0.4692] | 61.3 / 347.6 | 648 |
| fiqa | router_tau_escalation_10pct | 0.3947 | [0.3668, 0.4239] | 67.8 / 351.4 | 648 |
| fiqa | router_random_escalation_10pct | 0.3923 | [0.3648, 0.4209] | — | 648 |
| fiqa | hybrid_tau_escalation_10pct | 0.4024 | [0.3744, 0.4313] | — | 648 |
| fiqa | reranker_ms_marco_minilm_l_12_v2 | 0.3856 | — | 372.2 / 479.7 | 648 |
| fiqa | reranker_bge_reranker_base | 0.3508 | — | 1737.1 / 2347.2 | 648 |
| nfcorpus | always_bm25 | 0.3223 | — | 0.4 / 0.5 | 323 |
| nfcorpus | always_dense | 0.3255 | — | — | 323 |
| nfcorpus | always_hybrid | 0.3589 | [0.3234, 0.3948] | 7.0 / 8.9 | 323 |
| nfcorpus | rrf | 0.3509 | — | — | 323 |
| nfcorpus | best_single | 0.3156 | [0.2819, 0.3498] | — | 323 |
| nfcorpus | gbm_router | 0.3548 | [0.3194, 0.3906] | 12.0 / 15.8 | 323 |
| nfcorpus | logistic_router | 0.3387 | [0.3037, 0.3740] | — | 323 |
| nfcorpus | random_route | 0.3549 | [0.3196, 0.3905] | — | 323 |
| nfcorpus | oracle_route | 0.3877 | [0.3516, 0.4230] | — | 323 |
| nfcorpus | always_rerank_minilm_l6 | 0.3546 | [0.3199, 0.3901] | 405.6 / 439.8 | 323 |
| nfcorpus | predicted_gain_escalation_10pct | 0.3599 | [0.3249, 0.3954] | 50.4 / 414.2 | 323 |
| nfcorpus | confidence_escalation_10pct | 0.3558 | [0.3201, 0.3913] | 48.9 / 406.3 | 323 |
| nfcorpus | random_escalation_10pct | 0.3585 | [0.3232, 0.3942] | 44.4 / 409.5 | 323 |
| nfcorpus | oracle_escalation_10pct | 0.3751 | [0.3395, 0.4114] | 43.7 / 404.7 | 323 |
| nfcorpus | router_tau_escalation_10pct | 0.3556 | [0.3199, 0.3913] | 44.0 / 404.5 | 323 |
| nfcorpus | router_random_escalation_10pct | 0.3550 | [0.3197, 0.3907] | — | 323 |
| nfcorpus | hybrid_tau_escalation_10pct | 0.3566 | [0.3209, 0.3922] | — | 323 |
| nfcorpus | reranker_ms_marco_minilm_l_12_v2 | 0.3272 | — | 620.7 / 708.9 | 323 |
| nfcorpus | reranker_bge_reranker_base | 0.3257 | — | 3145.9 / 3543.6 | 323 |
| quora | always_hybrid | 0.9674 | [0.9595, 0.9748] | — | 1000 |
| quora | best_single | 0.9587 | [0.9497, 0.9670] | — | 1000 |
| quora | gbm_router | 0.9651 | [0.9569, 0.9730] | — | 1000 |
| quora | logistic_router | 0.9553 | [0.9455, 0.9643] | — | 1000 |
| quora | random_route | 0.9662 | [0.9584, 0.9735] | — | 1000 |
| quora | oracle_route | 0.9805 | [0.9747, 0.9858] | — | 1000 |
| quora | always_rerank_minilm_l6 | 0.9491 | [0.9396, 0.9578] | — | 1000 |
| quora | predicted_gain_escalation_10pct | 0.9630 | [0.9548, 0.9708] | — | 1000 |
| quora | confidence_escalation_10pct | 0.9639 | [0.9556, 0.9717] | — | 1000 |
| quora | random_escalation_10pct | 0.9656 | [0.9579, 0.9727] | — | 1000 |
| quora | oracle_escalation_10pct | 0.9781 | [0.9716, 0.9838] | — | 1000 |
| quora | router_tau_escalation_10pct | 0.9623 | [0.9535, 0.9702] | — | 1000 |
| quora | router_random_escalation_10pct | 0.9628 | [0.9545, 0.9705] | — | 1000 |
| quora | hybrid_tau_escalation_10pct | 0.9644 | [0.9560, 0.9721] | — | 1000 |
| scifact | always_bm25 | 0.6827 | — | 0.5 / 0.6 | 300 |
| scifact | always_dense | 0.6875 | — | — | 300 |
| scifact | always_hybrid | 0.7264 | [0.6843, 0.7685] | 7.3 / 8.4 | 300 |
| scifact | rrf | 0.7318 | — | — | 300 |
| scifact | best_single | 0.6680 | [0.6230, 0.7137] | — | 300 |
| scifact | gbm_router | 0.7300 | [0.6869, 0.7725] | 12.6 / 14.6 | 300 |
| scifact | logistic_router | 0.7060 | [0.6613, 0.7493] | — | 300 |
| scifact | random_route | 0.7162 | [0.6749, 0.7572] | — | 300 |
| scifact | oracle_route | 0.7911 | [0.7511, 0.8297] | — | 300 |
| scifact | skip_dense_gbm | 0.7237 | [0.6813, 0.7656] | 6.7 / 10.2 | 300 |
| scifact | skip_dense_random | 0.7039 | [0.6619, 0.7456] | — | 300 |
| scifact | skip_dense_oracle | 0.7505 | [0.7087, 0.7905] | — | 300 |
| scifact | skip_dense_top_ratio | 0.7255 | [0.6829, 0.7672] | — | 300 |
| scifact | always_rerank_minilm_l6 | 0.6986 | [0.6550, 0.7410] | 403.0 / 454.9 | 300 |
| scifact | predicted_gain_escalation_10pct | 0.7272 | [0.6848, 0.7693] | 53.3 / 400.6 | 300 |
| scifact | confidence_escalation_10pct | 0.7229 | [0.6803, 0.7650] | 52.2 / 397.7 | 300 |
| scifact | random_escalation_10pct | 0.7236 | [0.6823, 0.7648] | 47.0 / 399.5 | 300 |
| scifact | oracle_escalation_10pct | 0.7680 | [0.7279, 0.8068] | 46.7 / 397.5 | 300 |
| scifact | router_tau_escalation_10pct | 0.7302 | [0.6877, 0.7725] | 48.5 / 399.0 | 300 |
| scifact | router_random_escalation_10pct | 0.7273 | [0.6857, 0.7689] | — | 300 |
| scifact | hybrid_tau_escalation_10pct | 0.7247 | [0.6820, 0.7666] | — | 300 |
| scifact | reranker_ms_marco_minilm_l_12_v2 | 0.6964 | — | 621.5 / 732.2 | 300 |
| scifact | reranker_bge_reranker_base | 0.7178 | — | 3246.6 / 3753.5 | 300 |

## Paired held-out comparisons

Bootstrap confidence intervals and paired Wilcoxon tests use per-query nDCG@10 values.
Baseline artifacts also report MRR@10, MAP, and Recall@100.

| Dataset | Comparison | Δ nDCG@10 [95% bootstrap CI] | Bootstrap p | Wilcoxon p | n |
| :--- | :--- | :--- | ---: | ---: | ---: |
| cqadupstack-unix | gbm_router_vs_hybrid | -0.0105 [-0.0169, -0.0042] | 0.0016 | 0.0014 | 1072 |
| cqadupstack-unix | gbm_router_vs_random | +0.0003 [-0.0055, +0.0059] | 0.9284 | 0.0001 | 1072 |
| cqadupstack-unix | predicted_gain_vs_random_10pct | +0.0030 [-0.0010, +0.0072] | 0.1417 | 0.0277 | 1072 |
| cqadupstack-unix | confidence_vs_random_10pct | +0.0043 [+0.0003, +0.0085] | 0.0408 | 0.0151 | 1072 |
| cqadupstack-unix | always_rerank_vs_hybrid | -0.0081 [-0.0206, +0.0048] | 0.2092 | 0.1560 | 1072 |
| cqadupstack-unix | router_tau_vs_hybrid_10pct | -0.0070 [-0.0145, +0.0003] | 0.0638 | 0.0828 | 1072 |
| cqadupstack-unix | router_tau_vs_random_10pct | +0.0018 [-0.0032, +0.0068] | 0.4871 | 0.1401 | 1072 |
| fiqa | reranker_ms_marco_minilm_l_12_v2_vs_hybrid | -0.0173 [-0.0337, -0.0005] | 0.0411 | 0.0118 | 648 |
| fiqa | reranker_bge_reranker_base_vs_hybrid | -0.0521 [-0.0704, -0.0340] | 0.0001 | 0.0000 | 648 |
| fiqa | gbm_router_vs_hybrid | -0.0104 [-0.0172, -0.0043] | 0.0018 | 0.0038 | 648 |
| fiqa | gbm_router_vs_random | +0.0036 [-0.0031, +0.0096] | 0.2651 | 0.0000 | 648 |
| fiqa | predicted_gain_vs_random_10pct | +0.0008 [-0.0033, +0.0051] | 0.6963 | 0.0139 | 648 |
| fiqa | confidence_vs_random_10pct | +0.0040 [-0.0011, +0.0089] | 0.1170 | 0.0002 | 648 |
| fiqa | always_rerank_vs_hybrid | -0.0270 [-0.0441, -0.0102] | 0.0019 | 0.0009 | 648 |
| fiqa | router_tau_vs_hybrid_10pct | -0.0082 [-0.0156, -0.0010] | 0.0288 | 0.0328 | 648 |
| fiqa | router_tau_vs_random_10pct | +0.0024 [-0.0024, +0.0071] | 0.3133 | 0.0105 | 648 |
| nfcorpus | reranker_ms_marco_minilm_l_12_v2_vs_hybrid | -0.0317 [-0.0443, -0.0196] | 0.0001 | 0.0000 | 323 |
| nfcorpus | reranker_bge_reranker_base_vs_hybrid | -0.0332 [-0.0473, -0.0198] | 0.0001 | 0.0000 | 323 |
| nfcorpus | gbm_router_vs_hybrid | -0.0041 [-0.0087, +0.0000] | 0.0680 | 0.0768 | 323 |
| nfcorpus | gbm_router_vs_random | -0.0001 [-0.0046, +0.0040] | 0.9737 | 0.0000 | 323 |
| nfcorpus | predicted_gain_vs_random_10pct | +0.0014 [-0.0030, +0.0064] | 0.5449 | 0.6203 | 323 |
| nfcorpus | confidence_vs_random_10pct | -0.0027 [-0.0057, -0.0000] | 0.0604 | 0.2500 | 323 |
| nfcorpus | always_rerank_vs_hybrid | -0.0043 [-0.0153, +0.0063] | 0.4311 | 0.5462 | 323 |
| nfcorpus | router_tau_vs_hybrid_10pct | -0.0033 [-0.0074, +0.0007] | 0.1046 | 0.1414 | 323 |
| nfcorpus | router_tau_vs_random_10pct | +0.0006 [-0.0024, +0.0043] | 0.7436 | 0.6485 | 323 |
| quora | gbm_router_vs_hybrid | -0.0023 [-0.0058, +0.0014] | 0.2078 | 0.0959 | 1000 |
| quora | gbm_router_vs_random | -0.0011 [-0.0044, +0.0025] | 0.5428 | 0.0000 | 1000 |
| quora | predicted_gain_vs_random_10pct | -0.0026 [-0.0059, +0.0006] | 0.1281 | 0.0944 | 1000 |
| quora | confidence_vs_random_10pct | -0.0017 [-0.0056, +0.0024] | 0.4137 | 0.0541 | 1000 |
| quora | always_rerank_vs_hybrid | -0.0183 [-0.0260, -0.0108] | 0.0001 | 0.0000 | 1000 |
| quora | router_tau_vs_hybrid_10pct | -0.0052 [-0.0101, -0.0002] | 0.0403 | 0.0352 | 1000 |
| quora | router_tau_vs_random_10pct | -0.0005 [-0.0047, +0.0037] | 0.8124 | 0.0215 | 1000 |
| scifact | reranker_ms_marco_minilm_l_12_v2_vs_hybrid | -0.0300 [-0.0572, -0.0034] | 0.0304 | 0.0233 | 300 |
| scifact | reranker_bge_reranker_base_vs_hybrid | -0.0086 [-0.0343, +0.0175] | 0.5120 | 0.3157 | 300 |
| scifact | gbm_router_vs_hybrid | +0.0036 [-0.0100, +0.0182] | 0.6245 | 0.9288 | 300 |
| scifact | gbm_router_vs_random | +0.0138 [+0.0020, +0.0265] | 0.0290 | 0.0099 | 300 |
| scifact | skip_router_vs_hybrid | -0.0027 [-0.0093, +0.0044] | 0.4279 | 0.1335 | 300 |
| scifact | skip_router_vs_random | +0.0198 [+0.0104, +0.0296] | 0.0001 | 0.0001 | 300 |
| scifact | predicted_gain_vs_random_10pct | +0.0036 [-0.0035, +0.0111] | 0.3284 | 0.0610 | 300 |
| scifact | confidence_vs_random_10pct | -0.0008 [-0.0083, +0.0064] | 0.8354 | 0.3057 | 300 |
| scifact | always_rerank_vs_hybrid | -0.0278 [-0.0551, -0.0019] | 0.0422 | 0.0420 | 300 |
| scifact | router_tau_vs_hybrid_10pct | +0.0038 [-0.0099, +0.0188] | 0.6059 | 0.8448 | 300 |
| scifact | router_tau_vs_random_10pct | +0.0029 [-0.0047, +0.0106] | 0.4571 | 0.1240 | 300 |

## Pairwise win/loss/tie counts

Counts compare per-query nDCG@10; each label is system A versus system B.

| Dataset | Comparison | Wins | Losses | Ties | n |
| :--- | :--- | ---: | ---: | ---: | ---: |
| cqadupstack-unix | gbm_router_vs_hybrid | 47 | 73 | 952 | 1072 |
| cqadupstack-unix | gbm_router_vs_random | 283 | 150 | 639 | 1072 |
| cqadupstack-unix | predicted_gain_vs_random_10pct | 196 | 148 | 728 | 1072 |
| cqadupstack-unix | confidence_vs_random_10pct | 197 | 147 | 728 | 1072 |
| cqadupstack-unix | always_rerank_vs_hybrid | 155 | 189 | 728 | 1072 |
| cqadupstack-unix | router_tau_vs_hybrid_10pct | 67 | 79 | 926 | 1072 |
| cqadupstack-unix | router_tau_vs_random_10pct | 195 | 166 | 711 | 1072 |
| fiqa | reranker_ms_marco_minilm_l_12_v2_vs_hybrid | 148 | 190 | 310 | 648 |
| fiqa | reranker_bge_reranker_base_vs_hybrid | 132 | 247 | 269 | 648 |
| fiqa | gbm_router_vs_hybrid | 22 | 41 | 585 | 648 |
| fiqa | gbm_router_vs_random | 318 | 110 | 220 | 648 |
| fiqa | predicted_gain_vs_random_10pct | 199 | 149 | 300 | 648 |
| fiqa | confidence_vs_random_10pct | 207 | 141 | 300 | 648 |
| fiqa | always_rerank_vs_hybrid | 147 | 201 | 300 | 648 |
| fiqa | router_tau_vs_hybrid_10pct | 29 | 49 | 570 | 648 |
| fiqa | router_tau_vs_random_10pct | 202 | 152 | 294 | 648 |
| nfcorpus | reranker_ms_marco_minilm_l_12_v2_vs_hybrid | 75 | 123 | 125 | 323 |
| nfcorpus | reranker_bge_reranker_base_vs_hybrid | 70 | 122 | 131 | 323 |
| nfcorpus | gbm_router_vs_hybrid | 7 | 15 | 301 | 323 |
| nfcorpus | gbm_router_vs_random | 148 | 61 | 114 | 323 |
| nfcorpus | predicted_gain_vs_random_10pct | 87 | 95 | 141 | 323 |
| nfcorpus | confidence_vs_random_10pct | 79 | 103 | 141 | 323 |
| nfcorpus | always_rerank_vs_hybrid | 95 | 87 | 141 | 323 |
| nfcorpus | router_tau_vs_hybrid_10pct | 11 | 19 | 293 | 323 |
| nfcorpus | router_tau_vs_random_10pct | 80 | 100 | 143 | 323 |
| quora | gbm_router_vs_hybrid | 10 | 16 | 974 | 1000 |
| quora | gbm_router_vs_random | 164 | 48 | 788 | 1000 |
| quora | predicted_gain_vs_random_10pct | 99 | 62 | 839 | 1000 |
| quora | confidence_vs_random_10pct | 102 | 59 | 839 | 1000 |
| quora | always_rerank_vs_hybrid | 50 | 111 | 839 | 1000 |
| quora | router_tau_vs_hybrid_10pct | 26 | 35 | 939 | 1000 |
| quora | router_tau_vs_random_10pct | 107 | 56 | 837 | 1000 |
| scifact | reranker_ms_marco_minilm_l_12_v2_vs_hybrid | 41 | 66 | 193 | 300 |
| scifact | reranker_bge_reranker_base_vs_hybrid | 43 | 62 | 195 | 300 |
| scifact | gbm_router_vs_hybrid | 14 | 19 | 267 | 300 |
| scifact | gbm_router_vs_random | 86 | 52 | 162 | 300 |
| scifact | skip_router_vs_hybrid | 4 | 12 | 284 | 300 |
| scifact | skip_router_vs_random | 61 | 30 | 209 | 300 |
| scifact | predicted_gain_vs_random_10pct | 66 | 42 | 192 | 300 |
| scifact | confidence_vs_random_10pct | 63 | 45 | 192 | 300 |
| scifact | always_rerank_vs_hybrid | 41 | 67 | 192 | 300 |
| scifact | router_tau_vs_hybrid_10pct | 17 | 22 | 261 | 300 |
| scifact | router_tau_vs_random_10pct | 67 | 46 | 187 | 300 |

## Router confusion matrices

Rows are observed best-route labels using each held-out fold's α and ε; columns are router predictions.
Label order: BM25, dense, hybrid.

- **cqadupstack-unix, gbm:** `[[3, 10, 42], [10, 39, 70], [76, 207, 615]]`
- **cqadupstack-unix, logistic:** `[[15, 24, 16], [39, 46, 34], [245, 279, 374]]`
- **fiqa, gbm:** `[[4, 5, 30], [4, 10, 54], [35, 67, 439]]`
- **fiqa, logistic:** `[[13, 13, 13], [17, 27, 24], [168, 157, 216]]`
- **nfcorpus, gbm:** `[[1, 1, 13], [1, 2, 14], [12, 18, 261]]`
- **nfcorpus, logistic:** `[[2, 5, 8], [6, 5, 6], [61, 58, 172]]`
- **quora, gbm:** `[[3, 3, 17], [5, 5, 16], [18, 18, 915]]`
- **quora, logistic:** `[[10, 11, 2], [8, 11, 7], [98, 158, 695]]`
- **scifact, gbm:** `[[2, 6, 13], [3, 10, 21], [13, 48, 184]]`
- **scifact, logistic:** `[[6, 7, 8], [10, 17, 7], [56, 60, 129]]`

Unavailable from current artifacts: BM25/dense/RRF baseline metrics, MRR@10/MAP/Recall@100, advantage_regression_router, measured policy latency, skip_dense, stronger-reranker Experiment 4.
