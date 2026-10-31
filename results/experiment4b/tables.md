# Experiment 4b results: escalation to a blended cross-encoder (exploratory)

Design fixed before running in `results/preregistration/experiment4b_interpolated_rerank_20261031.md`. Held-out nDCG@10, 5-fold CV within each dataset (Experiment 4's folds, α* and top-50 pools). Blended backend = λ·mm(CE) + (1 − λ)·mm(hybrid), λ chosen on training folds.

**Prerequisite (pooled): always-blended − always-hybrid = +0.0067 [+0.0035, +0.0098] → MET.**

**H3 (pooled, B = 10%): predicted-gain − always-hybrid = +0.0011 [-0.0001, +0.0023], retained = 16% → NOT SUPPORTED.**

## Always-on systems

| Dataset | n | Always-hybrid | Always-rerank (pure CE) | Always-blended | Blended − hybrid | λ per fold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 0.3589 | 0.3546 | 0.3616 | +0.0027 [-0.0022, +0.0078] | 0.3, 0.3, 0.4, 0.4, 0.3 |
| scifact | 300 | 0.7264 | 0.6986 | 0.7327 | +0.0063 [-0.0037, +0.0166] | 0.5, 0.2, 0.5, 0.3, 0.3 |
| fiqa | 648 | 0.4029 | 0.3759 | 0.4198 | +0.0169 [+0.0075, +0.0266] | 0.6, 0.6, 0.6, 0.6, 0.4 |
| cqadupstack-unix | 1072 | 0.3764 | 0.3683 | 0.3832 | +0.0068 [+0.0002, +0.0135] | 0.6, 0.6, 0.3, 0.6, 0.6 |
| quora | 1000 | 0.9674 | 0.9491 | 0.9687 | +0.0013 [-0.0020, +0.0050] | 0.6, 0.6, 0.4, 0.4, 0.4 |
| **pooled** | 3343 | 0.5880 | 0.5718 | 0.5947 | +0.0067 [+0.0035, +0.0098] |  |

## Escalation at B = 10%

| Dataset | Pred-gain | Router-conf | Random | Oracle | Pred-gain − hybrid | Pred-gain − random | Retained (pred-gain) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 0.3599 | 0.3592 | 0.3592 | 0.3684 | +0.0010 [-0.0010, +0.0031] | +0.0008 [-0.0011, +0.0027] | 38% |
| scifact | 0.7243 | 0.7263 | 0.7270 | 0.7431 | -0.0021 [-0.0055, +0.0005] | -0.0028 [-0.0060, -0.0002] | -34% |
| fiqa | 0.4082 | 0.4069 | 0.4046 | 0.4300 | +0.0053 [+0.0018, +0.0092] | +0.0036 [+0.0003, +0.0072] | 31% |
| cqadupstack-unix | 0.3772 | 0.3793 | 0.3770 | 0.3990 | +0.0009 [-0.0012, +0.0032] | +0.0002 [-0.0018, +0.0023] | 13% |
| quora | 0.9670 | 0.9680 | 0.9676 | 0.9725 | -0.0004 [-0.0026, +0.0016] | -0.0006 [-0.0025, +0.0013] | -32% |
| **pooled** | 0.5891 | 0.5900 | 0.5887 | 0.6045 | +0.0011 [-0.0001, +0.0023] | +0.0004 [-0.0007, +0.0016] | 16% |

## Pooled, every budget: mean nDCG@10 (retained share of the always-blended gain)

| Budget | Predicted gain | Router confidence | Random | Oracle | Pred-gain − random |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 0.5887 (9%) | 0.5896 (23%) | 0.5884 (5%) | 0.6001 (179%) | +0.0003 [-0.0007, +0.0012] |
| 10% | 0.5891 (16%) | 0.5900 (29%) | 0.5887 (10%) | 0.6045 (246%) | +0.0004 [-0.0007, +0.0016] |
| 15% | 0.5894 (20%) | 0.5905 (37%) | 0.5890 (15%) | 0.6063 (272%) | +0.0003 [-0.0009, +0.0015] |
| 20% | 0.5900 (29%) | 0.5904 (36%) | 0.5894 (20%) | 0.6070 (283%) | +0.0006 [-0.0007, +0.0020] |
| 30% | 0.5904 (36%) | 0.5912 (47%) | 0.5900 (30%) | 0.6073 (288%) | +0.0004 [-0.0011, +0.0019] |
| 50% | 0.5919 (58%) | 0.5928 (70%) | 0.5914 (50%) | 0.6073 (288%) | +0.0005 [-0.0011, +0.0022] |

## Latency, composed from measured per-query components (ms, mean / p95)

| Dataset | Always-hybrid | Pred-gain @10% | Router-conf @10% | Always-blended |
| :--- | :---: | :---: | :---: | :---: |
| nfcorpus | 7.0 / 8.9 | 50.4 / 413.4 | 48.9 / 406.3 | 405.6 / 439.8 |
| scifact | 7.3 / 8.4 | 53.2 / 405.3 | 52.2 / 397.7 | 403.0 / 454.9 |
| fiqa | 27.1 / 103.9 | 76.5 / 352.5 | 74.1 / 353.8 | 367.8 / 505.6 |
