# Experiment 3 results: skipping the dense retriever

Pre-registered in improvements.md §4.1. Held-out nDCG@10 (5-fold CV per dataset); thresholds fitted on training folds.

**Verdict: SUCCESS** — non-inferior (CI low ≥ −0.01): PASS; skip rate ≥ 20%: PASS (46.7%); beats random skipping: PASS.

| Dataset | n | Always-hybrid | Always-BM25 | Router | Router skip rate | Router − hybrid | Router − random (same rate) | H1 skip / − random | H2 skip / − random | Oracle skip rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| scifact | 300 | 0.7264 | 0.6827 | 0.7237 | 46.7% | -0.0027 [-0.0083, +0.0037] | +0.0198 [+0.0101, +0.0280] | 49.7% / +0.0206 [+0.0109, +0.0301] | 9.3% / -0.0009 [-0.0062, +0.0041] | 77.0% |
| **pooled** | 300 | 0.7264 | 0.6827 | 0.7237 | 46.7% | -0.0027 [-0.0083, +0.0037] | +0.0198 [+0.0101, +0.0280] | 49.7% / +0.0206 [+0.0109, +0.0301] | 9.3% / -0.0009 [-0.0062, +0.0041] | 77.0% |

## Transfer (train on the other four datasets)

| Target | Skip rate | Router − hybrid | Router − random |
| :--- | :---: | :---: | :---: |

## Pooled curve: mean nDCG@10 at matched skip rates

| Skip rate | Router | H1 | H2 | Random | Oracle |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0% | 0.7264 | 0.7264 | 0.7264 | 0.7264 | 0.7264 |
| 10% | 0.7260 | 0.7264 | 0.7167 | 0.7220 | 0.7470 |
| 20% | 0.7283 | 0.7287 | 0.7122 | 0.7177 | 0.7505 |
| 30% | 0.7276 | 0.7280 | 0.7035 | 0.7133 | 0.7505 |
| 40% | 0.7295 | 0.7270 | 0.7010 | 0.7089 | 0.7505 |
| 50% | 0.7281 | 0.7246 | 0.6884 | 0.7045 | 0.7505 |
| 60% | 0.7297 | 0.7205 | 0.6824 | 0.7002 | 0.7505 |
| 70% | 0.7170 | 0.7179 | 0.6814 | 0.6958 | 0.7505 |
| 80% | 0.7088 | 0.7093 | 0.6839 | 0.6914 | 0.7487 |
| 90% | 0.7015 | 0.6942 | 0.6857 | 0.6870 | 0.7268 |
| 100% | 0.6827 | 0.6827 | 0.6827 | 0.6827 | 0.6827 |
