# Measured policy latency matched to held-out quality

Device: `cuda` (dense encoder and cross-encoder). Every component is timed for real on every test query (one pass after warm-up). Policy latency = sum of the components it runs; the cross-encoder (MiniLM-L6, hybrid top-50) is added only on queries the policy escalates. Escalation decisions are replayed from the saved held-out artifacts. Random escalation uses one seeded sample, so its nDCG here differs slightly from Experiment 4's expected value.

## Component latency (ms)

| Dataset | Component | Mean | p50 | p95 |
| :--- | :--- | ---: | ---: | ---: |
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
| nfcorpus | gbm_router | 0.3548 | 12.05 | 11.68 | 15.76 |
| nfcorpus | always_rerank_minilm_l6 | 0.3546 | 405.56 | 406.13 | 439.83 |
| nfcorpus | predicted_gain_escalation_10% | 0.3599 | 50.41 | 12.80 | 414.23 |
| nfcorpus | router_confidence_escalation_10% | 0.3558 | 48.88 | 11.76 | 406.33 |
| nfcorpus | random_escalation_10% | 0.3610 | 44.41 | 6.75 | 409.50 |
| nfcorpus | oracle_escalation_10% | 0.3751 | 43.72 | 6.76 | 404.75 |
| nfcorpus | router_tau_escalation_10% | 0.3556 | 44.02 | 11.76 | 404.51 |
| scifact | always_bm25 | 0.6827 | 0.45 | 0.43 | 0.58 |
| scifact | always_hybrid | 0.7264 | 7.33 | 7.13 | 8.42 |
| scifact | gbm_router | 0.7300 | 12.62 | 12.19 | 14.62 |
| scifact | always_rerank_minilm_l6 | 0.6986 | 403.01 | 399.17 | 454.91 |
| scifact | predicted_gain_escalation_10% | 0.7272 | 53.28 | 13.27 | 400.64 |
| scifact | router_confidence_escalation_10% | 0.7229 | 52.16 | 12.28 | 397.74 |
| scifact | random_escalation_10% | 0.7251 | 46.96 | 7.16 | 399.47 |
| scifact | oracle_escalation_10% | 0.7680 | 46.73 | 7.17 | 397.48 |
| scifact | router_tau_escalation_10% | 0.7302 | 48.47 | 12.28 | 399.04 |
| scifact | skip_dense_gbm | 0.7237 | 6.67 | 9.18 | 10.20 |
| fiqa | always_bm25 | 0.2528 | 1.14 | 0.81 | 3.13 |
| fiqa | always_hybrid | 0.4029 | 27.05 | 12.66 | 103.92 |
| fiqa | gbm_router | 0.3925 | 39.83 | 18.42 | 153.54 |
| fiqa | always_rerank_minilm_l6 | 0.3759 | 367.81 | 348.04 | 505.59 |
| fiqa | predicted_gain_escalation_10% | 0.4010 | 76.49 | 20.00 | 352.21 |
| fiqa | router_confidence_escalation_10% | 0.4042 | 74.11 | 18.77 | 353.77 |
| fiqa | random_escalation_10% | 0.3987 | 61.29 | 12.78 | 346.92 |
| fiqa | oracle_escalation_10% | 0.4399 | 61.28 | 12.80 | 347.59 |
| fiqa | router_tau_escalation_10% | 0.3947 | 67.81 | 18.68 | 351.42 |

All budgets are in `policy_latency.csv`.
