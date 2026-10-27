# Cross-encoder checkpoint benchmark

Exploratory matched-candidate-pool comparison. The registered Experiment 4 remains the primary result.

| Dataset | Model | Queries | Hybrid nDCG@10 | Reranker nDCG@10 | Δ [95% paired bootstrap CI] | Bootstrap p | Wilcoxon p | Wins / losses / ties | Mean / p95 ms |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | :--- | :--- |
| nfcorpus | cross-encoder/ms-marco-MiniLM-L-12-v2 | 323 | 0.3589 | 0.3272 | -0.0317 [-0.0443, -0.0196] | 0.0001 | 0.0000 | 75 / 123 / 125 | 620.7 / 708.9 |
| nfcorpus | BAAI/bge-reranker-base | 323 | 0.3589 | 0.3257 | -0.0332 [-0.0473, -0.0198] | 0.0001 | 0.0000 | 70 / 122 / 131 | 3145.9 / 3543.6 |
| scifact | cross-encoder/ms-marco-MiniLM-L-12-v2 | 300 | 0.7264 | 0.6964 | -0.0300 [-0.0572, -0.0034] | 0.0304 | 0.0233 | 41 / 66 / 193 | 621.5 / 732.2 |
| scifact | BAAI/bge-reranker-base | 300 | 0.7264 | 0.7178 | -0.0086 [-0.0343, +0.0175] | 0.5120 | 0.3157 | 43 / 62 / 195 | 3246.6 / 3753.5 |
| fiqa | cross-encoder/ms-marco-MiniLM-L-12-v2 | 648 | 0.4029 | 0.3856 | -0.0173 [-0.0337, -0.0005] | 0.0411 | 0.0118 | 148 / 190 / 310 | 372.2 / 479.7 |
| fiqa | BAAI/bge-reranker-base | 648 | 0.4029 | 0.3508 | -0.0521 [-0.0704, -0.0340] | 0.0001 | 0.0000 | 132 / 247 / 269 | 1737.1 / 2347.2 |
