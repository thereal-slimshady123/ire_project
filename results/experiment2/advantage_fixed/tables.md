# Experiment 2b: Dual Advantage Regression router (held-out)

5-fold CV within each dataset (Experiment 2 folds). alpha*, targets, theta, temperature and regressors are fitted on training folds only; synthetic queries (if any) join the training side only. Differences are nDCG@10, router minus always-hybrid, [95% paired bootstrap CI].

**Primary (pooled, n = 3343, 0 (natural only)): advantage router − always-hybrid = -0.0009 [-0.0028, +0.0009]** → NOT a significant gain.

Reference, pre-registered Experiment 2 (3-class GBM, pooled): -0.0061 [-0.0091, -0.0032].

## Scaling with synthetic training queries

| Synthetic per fold | nfcorpus | scifact | fiqa | cqadupstack-unix | quora | **pooled** | Pooled route shares B/D/H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (natural only) | -0.0034 [-0.0078, +0.0003] | +0.0002 [-0.0096, +0.0102] | -0.0000 [-0.0040, +0.0040] | -0.0026 [-0.0060, +0.0008] | +0.0008 [-0.0016, +0.0036] | -0.0009 [-0.0028, +0.0009] | 3%/11%/87% |

## Detail for the largest setting (0 (natural only))

| Dataset | n | Hybrid | BM25 | Dense | Router | Random (same mix) | Oracle | Win/Loss/Tie |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 0.3589 | 0.3223 | 0.3255 | 0.3555 | 0.3553 | 0.3877 | 8/12/303 |
| scifact | 300 | 0.7264 | 0.6827 | 0.6875 | 0.7266 | 0.7149 | 0.7911 | 11/15/274 |
| fiqa | 648 | 0.4029 | 0.2528 | 0.3747 | 0.4029 | 0.3984 | 0.4465 | 20/19/609 |
| cqadupstack-unix | 1072 | 0.3764 | 0.2937 | 0.3597 | 0.3738 | 0.3724 | 0.4182 | 21/30/1021 |
| quora | 1000 | 0.9674 | 0.9333 | 0.9587 | 0.9682 | 0.9663 | 0.9805 | 9/8/983 |
| pooled | 3343 | 0.5880 | 0.5148 | 0.5679 | 0.5871 | 0.5842 | 0.6224 | 69/84/3190 |

Router vs random routing at the same route mix (pooled): +0.0030 [+0.0011, +0.0047]; oracle − hybrid: +0.0344 [+0.0311, +0.0378]; share of oracle headroom captured: -2.6%.
