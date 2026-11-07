# Label reliability: is the per-query winner a stable property of the query?

Datasets: musique, 2wikimultihopqa, dbpedia. Base label: nDCG@10, e5-small-v2, official qrels. Share of queries with a single relevant document: 0.3%.

Pooled (query-weighted mean over datasets; qrels-half averaged over 5 random splits):

| Perturbation | Datasets | n | κ (3-way) | 3-way agreement | Binary retained | Binary flipped | Binary lost | Reference AUC′ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| alpha-0.1 | 3 | 5817 | 0.839 | 0.932 | 0.886 | 0.000 | 0.114 | — |
| alpha+0.1 | 3 | 5817 | 0.800 | 0.917 | 0.861 | 0.000 | 0.139 | — |
| ndcg@5 | 3 | 5817 | 0.814 | 0.926 | 0.780 | 0.008 | 0.212 | 0.974 |
| ndcg@20 | 3 | 5817 | 0.854 | 0.931 | 0.953 | 0.008 | 0.040 | 0.996 |
| mrr@10 | 3 | 5817 | 0.416 | 0.835 | 0.350 | 0.006 | 0.644 | 0.835 |
| qrels-half | 3 | 5801 | -0.053 | 0.615 | 0.087 | 0.111 | 0.802 | 0.714 |

Reference AUC′ pooled over queries (perturbed dense − BM25 margin → base binary label):

| Perturbation | queries | AUC′ [95% CI] |
| :--- | :---: | :---: |
| ndcg@5 | 1596 | 0.982 [0.978, 0.987] |
| ndcg@20 | 1596 | 0.996 [0.995, 0.998] |
| mrr@10 | 1596 | 0.890 [0.878, 0.902] |
| qrels-half | 2420 | 0.662 [0.640, 0.685] |

## Per dataset

| Dataset | α* | Perturbation | n | κ (3-way) | Binary retained | Binary lost | Reference AUC′ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| musique | 0.5 | alpha-0.1 | 2417.0 | 0.838 | 0.853 | 0.147 | — |
| musique | 0.5 | alpha+0.1 | 2417.0 | 0.818 | 0.920 | 0.080 | — |
| musique | 0.5 | ndcg@5 | 2417.0 | 0.815 | 0.819 | 0.179 | 0.987 |
| musique | 0.5 | ndcg@20 | 2417.0 | 0.847 | 0.948 | 0.046 | 0.999 |
| musique | 0.5 | mrr@10 | 2417.0 | 0.643 | 0.595 | 0.400 | 0.952 |
| musique | 0.5 | qrels-half | 2417.0 | -0.014 | 0.135 | 0.713 | 0.601 |
| 2wikimultihopqa | 0.6 | alpha-0.1 | 3000.0 | 0.857 | 0.905 | 0.095 | — |
| 2wikimultihopqa | 0.6 | alpha+0.1 | 3000.0 | 0.837 | 0.807 | 0.193 | — |
| 2wikimultihopqa | 0.6 | ndcg@5 | 3000.0 | 0.838 | 0.759 | 0.230 | 0.964 |
| 2wikimultihopqa | 0.6 | ndcg@20 | 3000.0 | 0.899 | 0.981 | 0.014 | 0.998 |
| 2wikimultihopqa | 0.6 | mrr@10 | 3000.0 | 0.262 | 0.165 | 0.829 | 0.742 |
| 2wikimultihopqa | 0.6 | qrels-half | 3000.0 | -0.084 | 0.021 | 0.925 | 0.819 |
| dbpedia | 0.9 | alpha-0.1 | 400.0 | 0.709 | 0.943 | 0.057 | — |
| dbpedia | 0.9 | alpha+0.1 | 400.0 | 0.426 | 0.907 | 0.093 | — |
| dbpedia | 0.9 | ndcg@5 | 400.0 | 0.624 | 0.703 | 0.264 | 0.977 |
| dbpedia | 0.9 | ndcg@20 | 400.0 | 0.558 | 0.764 | 0.187 | 0.968 |
| dbpedia | 0.9 | mrr@10 | 400.0 | 0.199 | 0.256 | 0.732 | 0.830 |
| dbpedia | 0.9 | qrels-half | 384.0 | -0.051 | 0.295 | 0.407 | 0.600 |
