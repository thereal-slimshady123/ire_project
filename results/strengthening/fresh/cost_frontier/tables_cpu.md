# Cost–quality frontier of skipping dense retrieval (cpu)

Datasets: cqadupstack-stats, cqadupstack-wordpress, cqadupstack-tex, scidocs, nq, hotpotqa. Dense model intfloat/e5-small-v2; FAISS flat search timed over random unit vectors of the corpus size. Quality: held-out nDCG@10 at matched skip rates (run_e4_matched_rate.py).

## Mean component latency (ms)

| Dataset | bm25_search | dense_encode | faiss_search | hybrid_fusion | skip_feature_extraction | router_inference | h1_score | Break-even skip rate: router / H1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-stats | 0.64 | 11.51 | 5.63 | 0.71 | 2.05 | 4.55 | 0.01 | 37.0% / 0.06% |
| cqadupstack-wordpress | 0.82 | 13.16 | 6.56 | 1.13 | 2.35 | 5.32 | 0.01 | 36.8% / 0.06% |
| cqadupstack-tex | 1.01 | 14.88 | 8.16 | 0.88 | 2.68 | 5.97 | 0.01 | 36.1% / 0.06% |
| scidocs | 0.66 | 14.77 | 6.17 | 0.86 | 2.78 | 5.60 | 0.01 | 38.4% / 0.06% |
| nq | 0.67 | 13.21 | 6.93 | 0.85 | 2.15 | 5.51 | 0.01 | 36.5% / 0.06% |
| hotpotqa | 0.69 | 474.30 | 6.92 | 0.85 | 1.97 | 5.54 | 0.01 | 1.6% / 0.00% |
| pooled | 0.75 | 227.53 | 7.04 | 0.86 | 2.21 | 5.57 | 0.01 | 3.3% / 0.01% |

## Pooled frontier

| Skip rate | Hybrid ms | Router ms | H1 ms | Random ms | Router nDCG | H1 nDCG | Random nDCG | Hybrid nDCG |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0% | 236.17 | 243.94 | 236.18 | 236.17 | 0.7076 | 0.7076 | 0.7076 | 0.7076 |
| 10% | 236.17 | 220.40 | 212.64 | 212.63 | 0.7063 | 0.7036 | 0.6993 | 0.7076 |
| 20% | 236.17 | 196.86 | 189.10 | 189.09 | 0.7038 | 0.6991 | 0.6910 | 0.7076 |
| 30% | 236.17 | 173.32 | 165.56 | 165.54 | 0.7002 | 0.6944 | 0.6827 | 0.7076 |
| 40% | 236.17 | 149.77 | 142.01 | 142.00 | 0.6958 | 0.6878 | 0.6744 | 0.7076 |
| 50% | 236.17 | 126.23 | 118.47 | 118.46 | 0.6893 | 0.6798 | 0.6661 | 0.7076 |

## Quality at 20% skip rate (paired bootstrap)

| Dataset | H1 − hybrid | H1 − random | Router − hybrid | H1 latency saving vs hybrid |
| :--- | :---: | :---: | :---: | :---: |
| cqadupstack-stats | -0.0047 [-0.0102, +0.0005] | +0.0054 [+0.0002, +0.0105] | -0.0062 [-0.0124, -0.0007] | 19.3% |
| cqadupstack-wordpress | -0.0015 [-0.0092, +0.0061] | +0.0113 [+0.0042, +0.0184] | -0.0105 [-0.0177, -0.0038] | 19.2% |
| cqadupstack-tex | -0.0011 [-0.0038, +0.0015] | +0.0069 [+0.0044, +0.0095] | -0.0046 [-0.0077, -0.0015] | 19.1% |
| scidocs | -0.0058 [-0.0089, -0.0029] | +0.0005 [-0.0023, +0.0032] | -0.0069 [-0.0101, -0.0040] | 19.4% |
| nq | -0.0033 [-0.0051, -0.0017] | +0.0352 [+0.0326, +0.0378] | -0.0049 [-0.0069, -0.0030] | 19.3% |
| hotpotqa | -0.0148 [-0.0164, -0.0133] | -0.0029 [-0.0043, -0.0015] | -0.0017 [-0.0022, -0.0012] | 20.0% |
| pooled | -0.0084 [-0.0094, -0.0074] | +0.0082 [+0.0071, +0.0092] | -0.0037 [-0.0046, -0.0029] | 19.9% |
