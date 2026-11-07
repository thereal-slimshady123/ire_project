# Measured policy latency matched to held-out quality

Device: `cuda` (dense encoder and cross-encoder). Every component is timed for real on every test query (one pass after warm-up). Policy latency = sum of the components it runs; the cross-encoder (MiniLM-L6, hybrid top-50) is added only on queries the policy escalates. Escalation decisions are replayed from the saved held-out artifacts. Random escalation uses one seeded sample, so its nDCG here differs slightly from Experiment 4's expected value.

## Component latency (ms)

| Dataset | Component | Mean | p50 | p95 |
| :--- | :--- | ---: | ---: | ---: |
| cqadupstack-unix | bm25_search | 0.69 | 0.66 | 0.96 |
| cqadupstack-unix | dense_encode | 5.96 | 5.87 | 6.48 |
| cqadupstack-unix | faiss_search | 7.52 | 6.93 | 12.15 |
| cqadupstack-unix | hybrid_fusion | 0.73 | 0.74 | 0.81 |
| cqadupstack-unix | feature_extraction | 4.28 | 4.06 | 6.07 |
| cqadupstack-unix | skip_feature_extraction | 1.09 | 1.04 | 1.52 |
| cqadupstack-unix | router_inference | 1.87 | 1.70 | 2.80 |
| cqadupstack-unix | gain_model_inference | 3.40 | 3.48 | 4.52 |
| cqadupstack-unix | cross_encoder_rerank | 333.16 | 332.61 | 367.26 |
| fiqa | bm25_search | 1.14 | 0.81 | 3.13 |
| fiqa | dense_encode | 10.44 | 5.95 | 28.20 |
| fiqa | faiss_search | 14.59 | 5.33 | 67.95 |
| fiqa | hybrid_fusion | 0.89 | 0.67 | 1.61 |
| fiqa | feature_extraction | 8.29 | 4.16 | 32.92 |
| fiqa | skip_feature_extraction | 2.28 | 1.26 | 9.00 |
| fiqa | router_inference | 4.49 | 1.47 | 20.13 |
| fiqa | gain_model_inference | 7.62 | 2.52 | 36.82 |
| fiqa | cross_encoder_rerank | 340.76 | 335.38 | 397.54 |
| nfcorpus | bm25_search | 0.37 | 0.38 | 0.53 |
| nfcorpus | dense_encode | 5.81 | 5.56 | 7.40 |
| nfcorpus | faiss_search | 0.42 | 0.38 | 0.57 |
| nfcorpus | hybrid_fusion | 0.40 | 0.42 | 0.51 |
| nfcorpus | feature_extraction | 3.65 | 3.74 | 4.70 |
| nfcorpus | skip_feature_extraction | 1.03 | 1.16 | 1.50 |
| nfcorpus | router_inference | 1.40 | 1.22 | 2.09 |
| nfcorpus | gain_model_inference | 2.64 | 2.31 | 4.47 |
| nfcorpus | cross_encoder_rerank | 398.57 | 399.35 | 433.16 |
| quora | bm25_search | 0.62 | 0.62 | 0.75 |
| quora | dense_encode | 6.08 | 5.96 | 6.82 |
| quora | faiss_search | 8.16 | 7.35 | 13.43 |
| quora | hybrid_fusion | 0.80 | 0.79 | 0.95 |
| quora | feature_extraction | 3.19 | 3.08 | 4.48 |
| quora | skip_feature_extraction | 0.56 | 0.54 | 0.71 |
| quora | router_inference | 2.65 | 2.46 | 4.37 |
| quora | gain_model_inference | 3.57 | 3.26 | 5.16 |
| quora | cross_encoder_rerank | 29.12 | 27.33 | 41.53 |
| scifact | bm25_search | 0.45 | 0.43 | 0.58 |
| scifact | dense_encode | 5.91 | 5.75 | 6.77 |
| scifact | faiss_search | 0.53 | 0.51 | 0.65 |
| scifact | hybrid_fusion | 0.44 | 0.44 | 0.48 |
| scifact | feature_extraction | 3.95 | 3.79 | 4.64 |
| scifact | skip_feature_extraction | 1.24 | 1.20 | 1.57 |
| scifact | router_inference | 1.34 | 1.22 | 1.45 |
| scifact | gain_model_inference | 2.44 | 2.14 | 3.72 |
| scifact | cross_encoder_rerank | 395.68 | 391.86 | 447.78 |

## Policies (nDCG@10 and per-query latency)

| Dataset | System | nDCG@10 | Mean ms | p50 ms | p95 ms |
| :--- | :--- | ---: | ---: | ---: | ---: |
| nfcorpus | always_bm25 | 0.3223 | 0.37 | 0.38 | 0.53 |
| nfcorpus | always_hybrid | 0.3589 | 7.00 | 6.73 | 8.95 |
| nfcorpus | gbm_router | 0.3550 | 12.05 | 11.68 | 15.76 |
| nfcorpus | always_rerank_minilm_l6 | 0.3546 | 405.56 | 406.13 | 439.83 |
| nfcorpus | predicted_gain_escalation_10% | 0.3599 | 50.41 | 12.80 | 414.23 |
| nfcorpus | router_confidence_escalation_10% | 0.3558 | 48.88 | 11.76 | 406.33 |
| nfcorpus | random_escalation_10% | 0.3610 | 44.41 | 6.75 | 409.50 |
| nfcorpus | oracle_escalation_10% | 0.3751 | 43.72 | 6.76 | 404.75 |
| nfcorpus | router_tau_escalation_10% | 0.3556 | 44.02 | 11.76 | 404.51 |
| nfcorpus | skip_dense_gbm | 0.3537 | 6.97 | 8.98 | 11.09 |
| scifact | always_bm25 | 0.6827 | 0.45 | 0.43 | 0.58 |
| scifact | always_hybrid | 0.7264 | 7.33 | 7.13 | 8.42 |
| scifact | gbm_router | 0.7239 | 12.62 | 12.19 | 14.62 |
| scifact | always_rerank_minilm_l6 | 0.6986 | 403.01 | 399.17 | 454.91 |
| scifact | predicted_gain_escalation_10% | 0.7272 | 53.28 | 13.27 | 400.64 |
| scifact | router_confidence_escalation_10% | 0.7229 | 52.16 | 12.28 | 397.74 |
| scifact | random_escalation_10% | 0.7251 | 46.96 | 7.16 | 399.47 |
| scifact | oracle_escalation_10% | 0.7680 | 46.73 | 7.17 | 397.48 |
| scifact | router_tau_escalation_10% | 0.7247 | 44.29 | 12.26 | 391.61 |
| scifact | skip_dense_gbm | 0.7237 | 6.67 | 9.18 | 10.20 |
| fiqa | always_bm25 | 0.2528 | 1.14 | 0.81 | 3.13 |
| fiqa | always_hybrid | 0.4029 | 27.05 | 12.66 | 103.92 |
| fiqa | gbm_router | 0.3909 | 39.83 | 18.42 | 153.54 |
| fiqa | always_rerank_minilm_l6 | 0.3759 | 367.81 | 348.04 | 505.59 |
| fiqa | predicted_gain_escalation_10% | 0.4010 | 76.49 | 20.00 | 352.21 |
| fiqa | router_confidence_escalation_10% | 0.4042 | 74.11 | 18.77 | 353.77 |
| fiqa | random_escalation_10% | 0.3987 | 61.29 | 12.78 | 346.92 |
| fiqa | oracle_escalation_10% | 0.4399 | 61.28 | 12.80 | 347.59 |
| fiqa | router_tau_escalation_10% | 0.3940 | 67.74 | 18.68 | 351.42 |
| fiqa | skip_dense_gbm | 0.3974 | 32.47 | 15.49 | 130.47 |
| cqadupstack-unix | always_bm25 | 0.2937 | 0.69 | 0.66 | 0.96 |
| cqadupstack-unix | always_hybrid | 0.3764 | 14.89 | 14.22 | 19.77 |
| cqadupstack-unix | gbm_router | 0.3658 | 21.03 | 20.06 | 27.21 |
| cqadupstack-unix | always_rerank_minilm_l6 | 0.3683 | 348.05 | 347.43 | 381.80 |
| cqadupstack-unix | predicted_gain_escalation_10% | 0.3786 | 55.76 | 22.31 | 355.50 |
| cqadupstack-unix | router_confidence_escalation_10% | 0.3798 | 54.38 | 20.25 | 353.83 |
| cqadupstack-unix | random_escalation_10% | 0.3764 | 48.52 | 14.37 | 351.42 |
| cqadupstack-unix | oracle_escalation_10% | 0.4176 | 47.73 | 14.35 | 347.09 |
| cqadupstack-unix | router_tau_escalation_10% | 0.3715 | 56.14 | 20.28 | 353.51 |
| cqadupstack-unix | skip_dense_gbm | 0.3697 | 15.49 | 16.77 | 22.84 |
| quora | always_bm25 | 0.9333 | 0.62 | 0.62 | 0.75 |
| quora | always_hybrid | 0.9674 | 15.65 | 14.70 | 21.53 |
| quora | gbm_router | 0.9643 | 21.49 | 20.55 | 28.00 |
| quora | always_rerank_minilm_l6 | 0.9491 | 44.77 | 43.18 | 58.15 |
| quora | predicted_gain_escalation_10% | 0.9630 | 25.31 | 21.90 | 49.50 |
| quora | router_confidence_escalation_10% | 0.9639 | 24.33 | 20.93 | 48.63 |
| quora | random_escalation_10% | 0.9662 | 18.58 | 14.94 | 43.60 |
| quora | oracle_escalation_10% | 0.9781 | 18.54 | 15.02 | 43.57 |
| quora | router_tau_escalation_10% | 0.9623 | 24.39 | 20.94 | 48.93 |
| quora | skip_dense_gbm | 0.9630 | 9.41 | 4.55 | 21.82 |

All budgets are in `policy_latency.csv`.
