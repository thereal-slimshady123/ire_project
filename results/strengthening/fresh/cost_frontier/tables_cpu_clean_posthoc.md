# Cost table (cpu) without the suspended measurement — POST HOC

Recomputed from `results/strengthening/fresh/cost_frontier/timings_cpu.csv` by `scripts/posthoc_cost_frontier_clean.py` (not pre-registered). Dropped 1 row(s) with a component above 60,000 ms: hotpotqa query 5abfe1115542997d64295909 (56.7 min). Quality numbers are unchanged; see `tables_cpu.md`.

## Means, corrupted row excluded (same statistic as the original table)

| Dataset | bm25_search | dense_encode | faiss_search | hybrid_fusion | skip_feature_extraction | router_inference | h1_score | Hybrid ms | Break-even: router / H1 | H1 saving @20% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-stats | 0.64 | 11.51 | 5.63 | 0.71 | 2.05 | 4.55 | 0.01 | 18.49 | 37.0% / 0.06% | 19.3% |
| cqadupstack-wordpress | 0.82 | 13.16 | 6.56 | 1.13 | 2.35 | 5.32 | 0.01 | 21.68 | 36.8% / 0.06% | 19.2% |
| cqadupstack-tex | 1.01 | 14.88 | 8.16 | 0.88 | 2.68 | 5.97 | 0.01 | 24.94 | 36.1% / 0.06% | 19.1% |
| scidocs | 0.66 | 14.77 | 6.17 | 0.86 | 2.78 | 5.60 | 0.01 | 22.45 | 38.4% / 0.06% | 19.4% |
| nq | 0.67 | 13.21 | 6.93 | 0.85 | 2.15 | 5.51 | 0.01 | 21.66 | 36.5% / 0.06% | 19.3% |
| hotpotqa | 0.69 | 14.99 | 6.90 | 0.85 | 1.97 | 5.53 | 0.01 | 23.44 | 33.0% / 0.05% | 19.4% |
| pooled | 0.75 | 14.37 | 7.03 | 0.86 | 2.21 | 5.56 | 0.01 | 23.00 | 34.9% / 0.06% | 19.3% |

## Medians, all rows (robustness check)

| Dataset | bm25_search | dense_encode | faiss_search | hybrid_fusion | skip_feature_extraction | router_inference | h1_score | Hybrid ms | Break-even: router / H1 | H1 saving @20% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-stats | 0.62 | 11.46 | 5.54 | 0.72 | 2.06 | 4.43 | 0.01 | 18.35 | 36.6% / 0.06% | 19.3% |
| cqadupstack-wordpress | 0.78 | 12.96 | 6.65 | 0.78 | 2.32 | 5.36 | 0.01 | 21.17 | 37.6% / 0.05% | 19.2% |
| cqadupstack-tex | 0.92 | 14.79 | 8.09 | 0.91 | 2.66 | 5.91 | 0.01 | 24.72 | 36.0% / 0.06% | 19.2% |
| scidocs | 0.65 | 14.60 | 6.15 | 0.88 | 2.83 | 5.54 | 0.01 | 22.28 | 38.7% / 0.06% | 19.4% |
| nq | 0.66 | 13.25 | 6.89 | 0.88 | 2.21 | 5.46 | 0.01 | 21.68 | 36.5% / 0.06% | 19.3% |
| hotpotqa | 0.67 | 14.58 | 6.74 | 0.86 | 1.97 | 5.44 | 0.01 | 22.85 | 33.4% / 0.05% | 19.4% |
| pooled | 0.69 | 14.11 | 6.84 | 0.87 | 2.19 | 5.50 | 0.01 | 22.51 | 35.2% / 0.05% | 19.3% |

## Pooled frontier (means, corrupted row excluded)

| Skip rate | Hybrid ms | Router ms | H1 ms |
| :---: | :---: | :---: | :---: |
| 0% | 23.00 | 30.77 | 23.01 |
| 10% | 23.00 | 28.55 | 20.79 |
| 20% | 23.00 | 26.32 | 18.56 |
| 30% | 23.00 | 24.09 | 16.34 |
| 40% | 23.00 | 21.87 | 14.11 |
| 50% | 23.00 | 19.64 | 11.89 |
