# Experiment 2 results: held-out routing vs tuned always-hybrid

Pre-registered in improvements.md §3. All numbers are nDCG@10 on held-out queries (5-fold CV within each dataset); α, ε, best single retriever and routers fitted on training folds only.

**Primary test (pooled, n = 2800): GBM router − always-hybrid = -0.0142 [-0.0186, -0.0102], p = 1.0e-04 → NEGATIVE (CI includes or is below 0).**

## Mean held-out nDCG@10

| Dataset | n | Always-hybrid | Best single | GBM router | Logistic router | Random (GBM mix) | Oracle (bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-physics | 1039 | 0.4218 | 0.4047 | 0.4040 | 0.3864 | 0.4094 | 0.4771 |
| cqadupstack-programmers | 876 | 0.4077 | 0.3947 | 0.3933 | 0.3769 | 0.3979 | 0.4481 |
| cqadupstack-gis | 885 | 0.3725 | 0.3468 | 0.3625 | 0.3482 | 0.3612 | 0.4255 |
| **pooled** | 2800 | 0.4018 | 0.3833 | 0.3876 | 0.3714 | 0.3906 | 0.4517 |

## Paired differences [95% bootstrap CI]

| Dataset | GBM − hybrid | Logistic − hybrid | GBM − best single | GBM − random (same mix) | Oracle − hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-physics | -0.0178 [-0.0259, -0.0098] | -0.0354 [-0.0467, -0.0243] | -0.0007 [-0.0117, +0.0102] | -0.0054 [-0.0126, +0.0017] | +0.0553 [+0.0472, +0.0636] |
| cqadupstack-programmers | -0.0144 [-0.0201, -0.0091] | -0.0308 [-0.0406, -0.0211] | -0.0014 [-0.0104, +0.0078] | -0.0046 [-0.0102, +0.0006] | +0.0404 [+0.0331, +0.0479] |
| cqadupstack-gis | -0.0099 [-0.0175, -0.0026] | -0.0243 [-0.0356, -0.0133] | +0.0157 [+0.0028, +0.0286] | +0.0014 [-0.0056, +0.0084] | +0.0530 [+0.0441, +0.0624] |
| **pooled** | -0.0142 [-0.0186, -0.0102] | -0.0304 [-0.0368, -0.0242] | +0.0043 [-0.0022, +0.0106] | -0.0030 [-0.0070, +0.0008] | +0.0499 [+0.0451, +0.0548] |

## Route shares chosen by the GBM router (held-out, mean over folds)

| Dataset | BM25 | Dense | Hybrid | ε chosen per fold | α* per fold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-physics | 13.4% | 14.6% | 72.0% | 0.1, 0.1, 0.1, 0.05, 0.01 | 0.5, 0.7, 0.7, 0.7, 0.6 |
| cqadupstack-programmers | 7.9% | 11.9% | 80.2% | 0.1, 0.05, 0.05, 0.05, 0.1 | 0.7, 0.7, 0.7, 0.7, 0.7 |
| cqadupstack-gis | 9.8% | 20.0% | 70.2% | 0.01, 0.02, 0.1, 0.1, 0.02 | 0.5, 0.5, 0.6, 0.6, 0.6 |

## Cross-dataset transfer (train on the other four datasets, test on this one)

| Target | n | Hybrid (source α) | Hybrid (in-domain CV α) | GBM router | GBM − hybrid(source α) | GBM − hybrid(in-domain α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-physics | 1039 | 0.4271 | 0.4218 | 0.4115 | -0.0155 [-0.0230, -0.0081] | -0.0103 [-0.0184, -0.0019] |
| cqadupstack-programmers | 876 | 0.4032 | 0.4077 | 0.3872 | -0.0160 [-0.0243, -0.0080] | -0.0205 [-0.0305, -0.0112] |
| cqadupstack-gis | 885 | 0.3721 | 0.3725 | 0.3565 | -0.0157 [-0.0233, -0.0086] | -0.0160 [-0.0251, -0.0072] |
