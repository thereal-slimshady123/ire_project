# Learned-fusion latency (CPU, descriptive)

Medians (robust to the one suspended measurement in the hybrid timings).

| Dataset | Hybrid ms | + query features | + candidate features | + LambdaRank | Learned fusion ms | Overhead p95 | Candidates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-stats | 18.39 | 2.19 | 2.22 | 1.22 | 23.35 | 6.87 | 171 |
| cqadupstack-wordpress | 21.26 | 2.23 | 2.25 | 1.24 | 26.25 | 8.01 | 175 |
| cqadupstack-tex | 24.74 | 2.52 | 2.26 | 1.24 | 29.99 | 8.90 | 178 |
| scidocs | 22.24 | 2.71 | 2.54 | 1.39 | 28.01 | 8.68 | 168 |
| nq | 21.72 | 2.04 | 2.61 | 1.40 | 26.90 | 8.46 | 184 |
| hotpotqa | 22.96 | 1.97 | 2.64 | 1.41 | 28.14 | 8.69 | 182 |
| pooled | 22.71 | 2.10 | 2.59 | 1.39 | 27.88 | 8.63 | 181 |
