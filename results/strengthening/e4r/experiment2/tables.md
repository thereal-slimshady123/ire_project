# Experiment 2 results: held-out routing vs tuned always-hybrid

Pre-registered in improvements.md §3. All numbers are nDCG@10 on held-out queries (5-fold CV within each dataset); α, ε, best single retriever and routers fitted on training folds only.

**Primary test (pooled, n = 1535): GBM router − always-hybrid = -0.0075 [-0.0113, -0.0039], p = 3.0e-04 → NEGATIVE (CI includes or is below 0).**

## Mean held-out nDCG@10

| Dataset | n | Always-hybrid | Best single | GBM router | Logistic router | Random (GBM mix) | Oracle (bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| climate-fever | 1535 | 0.4119 | 0.3802 | 0.4044 | 0.3807 | 0.4000 | 0.4608 |
| **pooled** | 1535 | 0.4119 | 0.3802 | 0.4044 | 0.3807 | 0.4000 | 0.4608 |

## Paired differences [95% bootstrap CI]

| Dataset | GBM − hybrid | Logistic − hybrid | GBM − best single | GBM − random (same mix) | Oracle − hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| climate-fever | -0.0075 [-0.0113, -0.0039] | -0.0312 [-0.0382, -0.0244] | +0.0242 [+0.0171, +0.0315] | +0.0044 [+0.0008, +0.0079] | +0.0489 [+0.0444, +0.0534] |
| **pooled** | -0.0075 [-0.0113, -0.0039] | -0.0312 [-0.0382, -0.0244] | +0.0242 [+0.0171, +0.0315] | +0.0044 [+0.0008, +0.0079] | +0.0489 [+0.0444, +0.0534] |

## Route shares chosen by the GBM router (held-out, mean over folds)

| Dataset | BM25 | Dense | Hybrid | ε chosen per fold | α* per fold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| climate-fever | 6.7% | 13.2% | 80.1% | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.6, 0.6, 0.6, 0.6, 0.6 |

## Cross-dataset transfer (train on the other four datasets, test on this one)

| Target | n | Hybrid (source α) | Hybrid (in-domain CV α) | GBM router | GBM − hybrid(source α) | GBM − hybrid(in-domain α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
