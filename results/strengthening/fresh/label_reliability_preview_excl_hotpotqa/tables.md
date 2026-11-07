# Label reliability: is the per-query winner a stable property of the query?

Datasets: cqadupstack-stats, cqadupstack-wordpress, cqadupstack-tex, scidocs, nq. Base label: nDCG@10, e5-small-v2, official qrels. Share of queries with a single relevant document: 69.4%.

Pooled (query-weighted mean over datasets; qrels-half averaged over 5 random splits):

| Perturbation | Datasets | n | κ (3-way) | 3-way agreement | Binary retained | Binary flipped | Binary lost | Reference AUC′ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| alpha-0.1 | 5 | 8551 | 0.834 | 0.960 | 0.925 | 0.000 | 0.075 | — |
| alpha+0.1 | 5 | 8551 | 0.803 | 0.956 | 0.811 | 0.000 | 0.189 | — |
| ndcg@5 | 5 | 8551 | 0.857 | 0.957 | 0.803 | 0.003 | 0.193 | 0.978 |
| ndcg@20 | 5 | 8551 | 0.874 | 0.952 | 0.974 | 0.004 | 0.022 | 0.999 |
| mrr@10 | 5 | 8551 | 0.884 | 0.971 | 0.819 | 0.004 | 0.177 | 0.989 |
| qrels-half | 5 | 2618 | 0.033 | 0.659 | 0.172 | 0.101 | 0.727 | 0.551 |

Reference AUC′ pooled over queries (perturbed dense − BM25 margin → base binary label):

| Perturbation | queries | AUC′ [95% CI] |
| :--- | :---: | :---: |
| ndcg@5 | 1497 | 0.976 [0.971, 0.981] |
| ndcg@20 | 1497 | 0.999 [0.999, 1.000] |
| mrr@10 | 1497 | 0.989 [0.984, 0.993] |
| qrels-half | 1024 | 0.522 [0.501, 0.559] |

## Per dataset

| Dataset | α* | Perturbation | n | κ (3-way) | Binary retained | Binary lost | Reference AUC′ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-stats | 0.6 | alpha-0.1 | 652.0 | 0.921 | 0.965 | 0.035 | — |
| cqadupstack-stats | 0.6 | alpha+0.1 | 652.0 | 0.869 | 0.886 | 0.114 | — |
| cqadupstack-stats | 0.6 | ndcg@5 | 652.0 | 0.816 | 0.719 | 0.281 | 0.966 |
| cqadupstack-stats | 0.6 | ndcg@20 | 652.0 | 0.827 | 0.991 | 0.009 | 1.000 |
| cqadupstack-stats | 0.6 | mrr@10 | 652.0 | 0.958 | 0.930 | 0.070 | 1.000 |
| cqadupstack-stats | 0.6 | qrels-half | 118.0 | 0.137 | 0.240 | 0.670 | 0.593 |
| cqadupstack-wordpress | 0.6 | alpha-0.1 | 541.0 | 0.863 | 0.910 | 0.090 | — |
| cqadupstack-wordpress | 0.6 | alpha+0.1 | 541.0 | 0.910 | 0.921 | 0.079 | — |
| cqadupstack-wordpress | 0.6 | ndcg@5 | 541.0 | 0.852 | 0.775 | 0.225 | 0.972 |
| cqadupstack-wordpress | 0.6 | ndcg@20 | 541.0 | 0.821 | 0.989 | 0.011 | 1.000 |
| cqadupstack-wordpress | 0.6 | mrr@10 | 541.0 | 0.954 | 0.933 | 0.067 | 0.993 |
| cqadupstack-wordpress | 0.6 | qrels-half | 87.0 | 0.023 | 0.096 | 0.827 | — |
| cqadupstack-tex | 0.6 | alpha-0.1 | 2906.0 | 0.877 | 0.900 | 0.100 | — |
| cqadupstack-tex | 0.6 | alpha+0.1 | 2906.0 | 0.853 | 0.844 | 0.156 | — |
| cqadupstack-tex | 0.6 | ndcg@5 | 2906.0 | 0.827 | 0.744 | 0.254 | 0.970 |
| cqadupstack-tex | 0.6 | ndcg@20 | 2906.0 | 0.829 | 0.975 | 0.025 | 1.000 |
| cqadupstack-tex | 0.6 | mrr@10 | 2906.0 | 0.936 | 0.904 | 0.091 | 0.989 |
| cqadupstack-tex | 0.6 | qrels-half | 747.0 | 0.083 | 0.170 | 0.726 | 0.529 |
| scidocs | 0.6 | alpha-0.1 | 1000.0 | 0.879 | 0.934 | 0.066 | — |
| scidocs | 0.6 | alpha+0.1 | 1000.0 | 0.868 | 0.889 | 0.111 | — |
| scidocs | 0.6 | ndcg@5 | 1000.0 | 0.708 | 0.667 | 0.319 | 0.950 |
| scidocs | 0.6 | ndcg@20 | 1000.0 | 0.738 | 0.906 | 0.071 | 0.992 |
| scidocs | 0.6 | mrr@10 | 1000.0 | 0.777 | 0.735 | 0.254 | 0.965 |
| scidocs | 0.6 | qrels-half | 1000.0 | 0.059 | 0.195 | 0.667 | 0.528 |
| nq | 0.8 | alpha-0.1 | 3452.0 | 0.765 | 0.939 | 0.061 | — |
| nq | 0.8 | alpha+0.1 | 3452.0 | 0.712 | 0.730 | 0.270 | — |
| nq | 0.8 | ndcg@5 | 3452.0 | 0.934 | 0.913 | 0.084 | 0.996 |
| nq | 0.8 | ndcg@20 | 3452.0 | 0.968 | 0.987 | 0.010 | 1.000 |
| nq | 0.8 | mrr@10 | 3452.0 | 0.845 | 0.732 | 0.265 | 0.992 |
| nq | 0.8 | qrels-half | 666.0 | -0.077 | 0.136 | 0.816 | 0.603 |
