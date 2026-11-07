# Experiment 2 results: held-out routing vs tuned always-hybrid

Pre-registered in improvements.md §3. All numbers are nDCG@10 on held-out queries (5-fold CV within each dataset); α, ε, best single retriever and routers fitted on training folds only.

**Primary test (pooled, n = 3969): GBM router − always-hybrid = -0.0099 [-0.0130, -0.0068], p = 1.0e-04 → NEGATIVE (CI includes or is below 0).**

## Mean held-out nDCG@10

| Dataset | n | Always-hybrid | Best single | GBM router | Logistic router | Random (GBM mix) | Oracle (bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | 804 | 0.2748 | 0.2517 | 0.2643 | 0.2525 | 0.2649 | 0.3156 |
| cqadupstack-english | 1570 | 0.4567 | 0.4206 | 0.4463 | 0.4195 | 0.4435 | 0.5003 |
| cqadupstack-gaming | 1595 | 0.5770 | 0.5571 | 0.5679 | 0.5477 | 0.5656 | 0.6248 |
| **pooled** | 3969 | 0.4682 | 0.4412 | 0.4583 | 0.4372 | 0.4564 | 0.5129 |

## Paired differences [95% bootstrap CI]

| Dataset | GBM − hybrid | Logistic − hybrid | GBM − best single | GBM − random (same mix) | Oracle − hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | -0.0106 [-0.0168, -0.0048] | -0.0223 [-0.0330, -0.0120] | +0.0126 [+0.0031, +0.0218] | -0.0006 [-0.0064, +0.0047] | +0.0408 [+0.0330, +0.0489] |
| cqadupstack-english | -0.0104 [-0.0158, -0.0054] | -0.0372 [-0.0464, -0.0284] | +0.0257 [+0.0162, +0.0351] | +0.0028 [-0.0021, +0.0076] | +0.0436 [+0.0380, +0.0495] |
| cqadupstack-gaming | -0.0091 [-0.0143, -0.0043] | -0.0293 [-0.0374, -0.0215] | +0.0108 [+0.0025, +0.0189] | +0.0022 [-0.0025, +0.0068] | +0.0478 [+0.0424, +0.0535] |
| **pooled** | -0.0099 [-0.0130, -0.0068] | -0.0310 [-0.0363, -0.0258] | +0.0170 [+0.0119, +0.0224] | +0.0019 [-0.0010, +0.0048] | +0.0447 [+0.0412, +0.0483] |

## Route shares chosen by the GBM router (held-out, mean over folds)

| Dataset | BM25 | Dense | Hybrid | ε chosen per fold | α* per fold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | 16.2% | 6.5% | 77.4% | 0.05, 0.05, 0.1, 0.1, 0.1 | 0.7, 0.7, 0.7, 0.7, 0.7 |
| cqadupstack-english | 8.2% | 13.2% | 78.6% | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.6, 0.6, 0.6, 0.6, 0.6 |
| cqadupstack-gaming | 10.2% | 15.3% | 74.5% | 0.05, 0.1, 0.1, 0.1, 0.1 | 0.6, 0.6, 0.6, 0.7, 0.6 |

## Cross-dataset transfer (train on the other four datasets, test on this one)

| Target | n | Hybrid (source α) | Hybrid (in-domain CV α) | GBM router | GBM − hybrid(source α) | GBM − hybrid(in-domain α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | 804 | 0.2735 | 0.2748 | 0.2580 | -0.0155 [-0.0236, -0.0074] | -0.0168 [-0.0256, -0.0083] |
| cqadupstack-english | 1570 | 0.4567 | 0.4567 | 0.4413 | -0.0154 [-0.0214, -0.0098] | -0.0154 [-0.0214, -0.0098] |
| cqadupstack-gaming | 1595 | 0.5783 | 0.5770 | 0.5573 | -0.0210 [-0.0281, -0.0144] | -0.0197 [-0.0267, -0.0131] |
