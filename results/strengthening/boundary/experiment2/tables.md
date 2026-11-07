# Experiment 2 results: held-out routing vs tuned always-hybrid

Pre-registered in improvements.md §3. All numbers are nDCG@10 on held-out queries (5-fold CV within each dataset); α, ε, best single retriever and routers fitted on training folds only.

**Primary test (pooled, n = 5817): GBM router − always-hybrid = -0.0021 [-0.0039, -0.0002], p = 2.7e-02 → NEGATIVE (CI includes or is below 0).**

## Mean held-out nDCG@10

| Dataset | n | Always-hybrid | Best single | GBM router | Logistic router | Random (GBM mix) | Oracle (bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| musique | 2417 | 0.5374 | 0.4726 | 0.5340 | 0.5208 | 0.5126 | 0.5925 |
| 2wikimultihopqa | 3000 | 0.7598 | 0.7507 | 0.7592 | 0.7488 | 0.7581 | 0.7758 |
| dbpedia | 400 | 0.5409 | 0.5416 | 0.5356 | 0.4861 | 0.5321 | 0.5706 |
| **pooled** | 5817 | 0.6523 | 0.6208 | 0.6503 | 0.6360 | 0.6406 | 0.6855 |

## Paired differences [95% bootstrap CI]

| Dataset | GBM − hybrid | Logistic − hybrid | GBM − best single | GBM − random (same mix) | Oracle − hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| musique | -0.0034 [-0.0075, +0.0007] | -0.0165 [-0.0222, -0.0108] | +0.0614 [+0.0541, +0.0685] | +0.0214 [+0.0175, +0.0253] | +0.0552 [+0.0514, +0.0590] |
| 2wikimultihopqa | -0.0006 [-0.0014, +0.0002] | -0.0109 [-0.0135, -0.0084] | +0.0085 [+0.0060, +0.0111] | +0.0011 [+0.0003, +0.0019] | +0.0160 [+0.0143, +0.0178] |
| dbpedia | -0.0054 [-0.0105, -0.0011] | -0.0548 [-0.0695, -0.0406] | -0.0061 [-0.0122, -0.0004] | +0.0035 [-0.0016, +0.0078] | +0.0296 [+0.0234, +0.0363] |
| **pooled** | -0.0021 [-0.0039, -0.0002] | -0.0163 [-0.0192, -0.0134] | +0.0295 [+0.0260, +0.0329] | +0.0097 [+0.0080, +0.0114] | +0.0332 [+0.0313, +0.0352] |

## Route shares chosen by the GBM router (held-out, mean over folds)

| Dataset | BM25 | Dense | Hybrid | ε chosen per fold | α* per fold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| musique | 17.7% | 18.7% | 63.6% | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.5, 0.5, 0.5, 0.5, 0.5 |
| 2wikimultihopqa | 2.8% | 1.8% | 95.4% | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.7, 0.6, 0.7, 0.6, 0.6 |
| dbpedia | 5.5% | 1.0% | 93.5% | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.9, 0.9, 0.9, 0.9, 1 |

## Cross-dataset transfer (train on the other four datasets, test on this one)

| Target | n | Hybrid (source α) | Hybrid (in-domain CV α) | GBM router | GBM − hybrid(source α) | GBM − hybrid(in-domain α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| musique | 2417 | 0.5097 | 0.5374 | 0.5117 | +0.0020 [-0.0016, +0.0056] | -0.0257 [-0.0307, -0.0207] |
| 2wikimultihopqa | 3000 | 0.7583 | 0.7598 | 0.7562 | -0.0021 [-0.0037, -0.0006] | -0.0036 [-0.0055, -0.0017] |
| dbpedia | 400 | 0.5036 | 0.5409 | 0.5147 | +0.0111 [-0.0017, +0.0237] | -0.0262 [-0.0371, -0.0157] |
