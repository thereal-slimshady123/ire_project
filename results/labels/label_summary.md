# Held-out route labels (training-fold α* and ε)

Outcome-derived and descriptive: each held-out query labelled with its training fold's α* and ε (margin labeler; near-ties → hybrid). Routers are trained on labels recomputed inside training folds.

| Dataset | n | BM25 | Dense | Hybrid | Mean max regret |
| :--- | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 4.6% | 5.3% | 90.1% | 0.1166 |
| scifact | 300 | 7.0% | 11.3% | 81.7% | 0.1972 |
| fiqa | 648 | 6.0% | 10.5% | 83.5% | 0.2447 |
| cqadupstack-unix | 1072 | 5.1% | 11.1% | 83.8% | 0.1700 |
| quora | 1000 | 2.3% | 2.6% | 95.1% | 0.0666 |
