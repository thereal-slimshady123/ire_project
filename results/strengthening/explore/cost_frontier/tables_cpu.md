# Cost–quality frontier of skipping dense retrieval (cpu)

Datasets: cqadupstack-physics, cqadupstack-programmers, cqadupstack-gis. Dense model intfloat/e5-small-v2; FAISS flat search timed over random unit vectors of the corpus size. Quality: held-out nDCG@10 at matched skip rates (run_e4_matched_rate.py).

## Mean component latency (ms)

| Dataset | bm25_search | dense_encode | faiss_search | hybrid_fusion | skip_feature_extraction | router_inference | h1_score | Break-even skip rate: router / H1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-physics | 0.64 | 12.82 | 6.06 | 0.78 | 2.13 | 5.08 | 0.01 | 36.7% / 0.06% |
| cqadupstack-programmers | 0.75 | 14.62 | 6.47 | 0.87 | 2.64 | 5.75 | 0.01 | 38.2% / 0.06% |
| cqadupstack-gis | 0.77 | 14.55 | 6.67 | 0.86 | 2.49 | 5.68 | 0.01 | 37.0% / 0.06% |
| pooled | 0.72 | 13.93 | 6.38 | 0.83 | 2.40 | 5.48 | 0.01 | 37.3% / 0.06% |

## Pooled frontier

| Skip rate | Hybrid ms | Router ms | H1 ms | Random ms | Router nDCG | H1 nDCG | Random nDCG | Hybrid nDCG |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0% | 21.86 | 29.74 | 21.87 | 21.86 | 0.4018 | 0.4018 | 0.4018 | 0.4018 |
| 10% | 21.86 | 27.63 | 19.76 | 19.75 | 0.3992 | 0.4006 | 0.3938 | 0.4018 |
| 20% | 21.86 | 25.51 | 17.64 | 17.63 | 0.3918 | 0.3947 | 0.3862 | 0.4018 |
| 30% | 21.86 | 23.40 | 15.53 | 15.52 | 0.3848 | 0.3865 | 0.3784 | 0.4018 |
| 40% | 21.86 | 21.28 | 13.42 | 13.40 | 0.3787 | 0.3775 | 0.3705 | 0.4018 |
| 50% | 21.86 | 19.17 | 11.30 | 11.29 | 0.3685 | 0.3691 | 0.3626 | 0.4018 |

## Quality at 20% skip rate (paired bootstrap)

| Dataset | H1 − hybrid | H1 − random | Router − hybrid | H1 latency saving vs hybrid |
| :--- | :---: | :---: | :---: | :---: |
| cqadupstack-physics | -0.0036 [-0.0088, +0.0016] | +0.0110 [+0.0059, +0.0160] | -0.0087 [-0.0146, -0.0030] | 19.3% |
| cqadupstack-programmers | -0.0095 [-0.0158, -0.0037] | +0.0118 [+0.0056, +0.0177] | -0.0163 [-0.0229, -0.0103] | 19.3% |
| cqadupstack-gis | -0.0088 [-0.0150, -0.0029] | +0.0025 [-0.0033, +0.0079] | -0.0054 [-0.0104, -0.0007] | 19.3% |
| pooled | -0.0071 [-0.0104, -0.0038] | +0.0086 [+0.0053, +0.0117] | -0.0100 [-0.0133, -0.0068] | 19.3% |
