# Experiment 2b: Dual Advantage Regression router (held-out)

5-fold CV within each dataset (Experiment 2 folds). alpha*, targets, theta, temperature and regressors are fitted on training folds only; synthetic queries (if any) join the training side only. Differences are nDCG@10, router minus always-hybrid, [95% paired bootstrap CI].

**Primary (pooled, n = 3343, 0 (natural only)): advantage router − always-hybrid = -0.0001 [-0.0007, +0.0003]** → NOT a significant gain.

Reference, pre-registered Experiment 2 (3-class GBM, pooled): -0.0061 [-0.0091, -0.0032].

## Scaling with synthetic training queries

| Synthetic per fold | nfcorpus | scifact | fiqa | cqadupstack-unix | quora | **pooled** | Pooled route shares B/D/H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (natural only) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0003, +0.0000] | -0.0003 [-0.0020, +0.0010] | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0007, +0.0003] | 0%/1%/99% |

## Detail for the largest setting (0 (natural only))

| Dataset | n | Hybrid | BM25 | Dense | Router | Random (same mix) | Oracle | Win/Loss/Tie |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 0.3589 | 0.3223 | 0.3255 | 0.3589 | 0.3589 | 0.3877 | 0/0/323 |
| scifact | 300 | 0.7264 | 0.6827 | 0.6875 | 0.7264 | 0.7264 | 0.7911 | 0/0/300 |
| fiqa | 648 | 0.4029 | 0.2528 | 0.3747 | 0.4028 | 0.4024 | 0.4465 | 0/1/647 |
| cqadupstack-unix | 1072 | 0.3764 | 0.2937 | 0.3597 | 0.3760 | 0.3757 | 0.4182 | 4/3/1065 |
| quora | 1000 | 0.9674 | 0.9333 | 0.9587 | 0.9674 | 0.9674 | 0.9805 | 0/0/1000 |
| pooled | 3343 | 0.5880 | 0.5148 | 0.5679 | 0.5879 | 0.5877 | 0.6224 | 4/4/3335 |

Router vs random routing at the same route mix (pooled): +0.0002 [-0.0003, +0.0006]; oracle − hybrid: +0.0344 [+0.0311, +0.0378]; share of oracle headroom captured: -0.4%.
