# Label reliability: is the per-query winner a stable property of the query?

Datasets: nfcorpus, scifact, fiqa, cqadupstack-unix, quora, wands, cosqa, cqadupstack-android, cqadupstack-webmasters, arguana, cqadupstack-mathematica, cqadupstack-english, cqadupstack-gaming, cqadupstack-physics, cqadupstack-programmers, cqadupstack-gis. Base label: nDCG@10, e5-small-v2, official qrels. Share of queries with a single relevant document: 72.6%.

Pooled (query-weighted mean over datasets; qrels-half averaged over 5 random splits):

| Perturbation | Datasets | n | κ (3-way) | 3-way agreement | Binary retained | Binary flipped | Binary lost | Reference AUC′ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| alpha-0.1 | 16 | 13697 | 0.854 | 0.947 | 0.907 | 0.000 | 0.093 | — |
| alpha+0.1 | 16 | 13697 | 0.857 | 0.949 | 0.871 | 0.000 | 0.129 | — |
| ndcg@5 | 16 | 13697 | 0.817 | 0.938 | 0.749 | 0.004 | 0.248 | 0.972 |
| ndcg@20 | 16 | 13697 | 0.840 | 0.934 | 0.959 | 0.007 | 0.034 | 0.998 |
| mrr@10 | 16 | 13697 | 0.864 | 0.954 | 0.819 | 0.004 | 0.178 | 0.976 |
| e5-base | 7 | 4533 | 0.500 | 0.812 | 0.582 | 0.027 | 0.392 | 0.941 |
| qrels-half | 14 | 3752 | 0.032 | 0.562 | 0.192 | 0.176 | 0.632 | 0.594 |

Reference AUC′ pooled over queries (perturbed dense − BM25 margin → base binary label):

| Perturbation | queries | AUC′ [95% CI] |
| :--- | :---: | :---: |
| ndcg@5 | 3089 | 0.974 [0.970, 0.977] |
| ndcg@20 | 3089 | 0.999 [0.998, 0.999] |
| mrr@10 | 3089 | 0.987 [0.983, 0.990] |
| e5-base | 1154 | 0.943 [0.931, 0.955] |
| qrels-half | 1890 | 0.569 [0.543, 0.594] |

## Per dataset

| Dataset | α* | Perturbation | n | κ (3-way) | Binary retained | Binary lost | Reference AUC′ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 0.5 | alpha-0.1 | 323.0 | 0.862 | 0.847 | 0.153 | — |
| nfcorpus | 0.5 | alpha+0.1 | 323.0 | 0.832 | 0.919 | 0.081 | — |
| nfcorpus | 0.5 | ndcg@5 | 323.0 | 0.670 | 0.613 | 0.355 | 0.942 |
| nfcorpus | 0.5 | ndcg@20 | 323.0 | 0.687 | 0.766 | 0.234 | 0.989 |
| nfcorpus | 0.5 | mrr@10 | 323.0 | 0.523 | 0.411 | 0.581 | 0.894 |
| nfcorpus | 0.5 | e5-base | 323.0 | 0.573 | 0.613 | 0.355 | 0.941 |
| nfcorpus | 0.5 | qrels-half | 300.0 | 0.101 | 0.239 | 0.532 | 0.542 |
| scifact | 0.5 | alpha-0.1 | 300.0 | 0.901 | 0.902 | 0.098 | — |
| scifact | 0.5 | alpha+0.1 | 300.0 | 0.875 | 0.918 | 0.082 | — |
| scifact | 0.5 | ndcg@5 | 300.0 | 0.897 | 0.803 | 0.197 | 0.990 |
| scifact | 0.5 | ndcg@20 | 300.0 | 0.918 | 1.000 | 0.000 | 1.000 |
| scifact | 0.5 | mrr@10 | 300.0 | 0.940 | 0.869 | 0.131 | 1.000 |
| scifact | 0.5 | e5-base | 300.0 | 0.485 | 0.492 | 0.443 | 0.929 |
| scifact | 0.5 | qrels-half | 23.0 | -0.157 | 0.102 | 0.876 | — |
| fiqa | 0.6 | alpha-0.1 | 648.0 | 0.814 | 0.932 | 0.068 | — |
| fiqa | 0.6 | alpha+0.1 | 648.0 | 0.807 | 0.811 | 0.189 | — |
| fiqa | 0.6 | ndcg@5 | 648.0 | 0.741 | 0.684 | 0.311 | 0.966 |
| fiqa | 0.6 | ndcg@20 | 648.0 | 0.793 | 0.916 | 0.084 | 1.000 |
| fiqa | 0.6 | mrr@10 | 648.0 | 0.782 | 0.705 | 0.289 | 0.991 |
| fiqa | 0.6 | qrels-half | 428.0 | 0.014 | 0.181 | 0.733 | 0.513 |
| cqadupstack-unix | 0.6 | alpha-0.1 | 1072.0 | 0.874 | 0.925 | 0.075 | — |
| cqadupstack-unix | 0.6 | alpha+0.1 | 1072.0 | 0.841 | 0.850 | 0.150 | — |
| cqadupstack-unix | 0.6 | ndcg@5 | 1072.0 | 0.834 | 0.755 | 0.245 | 0.973 |
| cqadupstack-unix | 0.6 | ndcg@20 | 1072.0 | 0.792 | 0.985 | 0.015 | 0.999 |
| cqadupstack-unix | 0.6 | mrr@10 | 1072.0 | 0.946 | 0.920 | 0.075 | 0.995 |
| cqadupstack-unix | 0.6 | qrels-half | 271.0 | 0.099 | 0.186 | 0.695 | 0.569 |
| quora | 0.6 | alpha-0.1 | 1000.0 | 0.855 | 0.913 | 0.087 | — |
| quora | 0.6 | alpha+0.1 | 1000.0 | 0.908 | 0.913 | 0.087 | — |
| quora | 0.6 | ndcg@5 | 1000.0 | 0.902 | 0.841 | 0.159 | 0.995 |
| quora | 0.6 | ndcg@20 | 1000.0 | 0.977 | 0.971 | 0.029 | 1.000 |
| quora | 0.6 | mrr@10 | 1000.0 | 0.657 | 0.493 | 0.507 | 0.935 |
| quora | 0.6 | qrels-half | 217.0 | -0.003 | 0.128 | 0.711 | 0.738 |
| wands | 0.5 | alpha-0.1 | 479.0 | 0.823 | 0.904 | 0.096 | — |
| wands | 0.5 | alpha+0.1 | 479.0 | 0.845 | 0.875 | 0.125 | — |
| wands | 0.5 | ndcg@5 | 479.0 | 0.610 | 0.611 | 0.370 | 0.969 |
| wands | 0.5 | ndcg@20 | 479.0 | 0.586 | 0.755 | 0.207 | 0.963 |
| wands | 0.5 | mrr@10 | 479.0 | 0.190 | 0.120 | 0.875 | 0.732 |
| wands | 0.5 | qrels-half | 477.0 | -0.253 | 0.146 | 0.359 | 0.763 |
| cosqa | 0.8 | alpha-0.1 | 500.0 | 0.813 | 0.964 | 0.036 | — |
| cosqa | 0.8 | alpha+0.1 | 500.0 | 0.865 | 0.783 | 0.217 | — |
| cosqa | 0.8 | ndcg@5 | 500.0 | 0.775 | 0.602 | 0.398 | 0.922 |
| cosqa | 0.8 | ndcg@20 | 500.0 | 0.784 | 1.000 | 0.000 | 1.000 |
| cosqa | 0.8 | mrr@10 | 500.0 | 1.000 | 1.000 | 0.000 | 1.000 |
| cosqa | 0.8 | e5-base | 500.0 | 0.483 | 0.614 | 0.386 | 0.952 |
| cqadupstack-android | 0.5 | alpha-0.1 | 699.0 | 0.893 | 0.895 | 0.105 | — |
| cqadupstack-android | 0.5 | alpha+0.1 | 699.0 | 0.874 | 0.860 | 0.140 | — |
| cqadupstack-android | 0.5 | ndcg@5 | 699.0 | 0.834 | 0.749 | 0.240 | 0.977 |
| cqadupstack-android | 0.5 | ndcg@20 | 699.0 | 0.823 | 0.930 | 0.029 | 0.995 |
| cqadupstack-android | 0.5 | mrr@10 | 699.0 | 0.884 | 0.842 | 0.146 | 0.988 |
| cqadupstack-android | 0.5 | e5-base | 699.0 | 0.553 | 0.602 | 0.351 | 0.934 |
| cqadupstack-android | 0.5 | qrels-half | 191.0 | 0.131 | 0.228 | 0.618 | 0.602 |
| cqadupstack-webmasters | 0.7 | alpha-0.1 | 506.0 | 0.872 | 0.931 | 0.069 | — |
| cqadupstack-webmasters | 0.7 | alpha+0.1 | 506.0 | 0.887 | 0.882 | 0.118 | — |
| cqadupstack-webmasters | 0.7 | ndcg@5 | 506.0 | 0.781 | 0.696 | 0.294 | 0.960 |
| cqadupstack-webmasters | 0.7 | ndcg@20 | 506.0 | 0.789 | 0.951 | 0.029 | 0.996 |
| cqadupstack-webmasters | 0.7 | mrr@10 | 506.0 | 0.886 | 0.843 | 0.157 | 0.995 |
| cqadupstack-webmasters | 0.7 | e5-base | 506.0 | 0.523 | 0.647 | 0.353 | 0.984 |
| cqadupstack-webmasters | 0.7 | qrels-half | 127.0 | 0.200 | 0.263 | 0.570 | 0.557 |
| arguana | 0.3 | alpha-0.1 | 1401.0 | 0.835 | 0.852 | 0.148 | — |
| arguana | 0.3 | alpha+0.1 | 1401.0 | 0.845 | 0.953 | 0.047 | — |
| arguana | 0.3 | ndcg@5 | 1401.0 | 0.873 | 0.823 | 0.177 | 0.983 |
| arguana | 0.3 | ndcg@20 | 1401.0 | 0.936 | 1.000 | 0.000 | 1.000 |
| arguana | 0.3 | mrr@10 | 1401.0 | 1.000 | 1.000 | 0.000 | 1.000 |
| arguana | 0.3 | e5-base | 1401.0 | 0.487 | 0.552 | 0.424 | 0.941 |
| cqadupstack-mathematica | 0.7 | alpha-0.1 | 804.0 | 0.863 | 0.898 | 0.102 | — |
| cqadupstack-mathematica | 0.7 | alpha+0.1 | 804.0 | 0.871 | 0.868 | 0.132 | — |
| cqadupstack-mathematica | 0.7 | ndcg@5 | 804.0 | 0.751 | 0.683 | 0.317 | 0.949 |
| cqadupstack-mathematica | 0.7 | ndcg@20 | 804.0 | 0.796 | 0.964 | 0.030 | 0.998 |
| cqadupstack-mathematica | 0.7 | mrr@10 | 804.0 | 0.933 | 0.922 | 0.066 | 0.974 |
| cqadupstack-mathematica | 0.7 | e5-base | 804.0 | 0.449 | 0.575 | 0.395 | 0.919 |
| cqadupstack-mathematica | 0.7 | qrels-half | 218.0 | 0.077 | 0.156 | 0.684 | 0.536 |
| cqadupstack-english | 0.6 | alpha-0.1 | 1570.0 | 0.839 | 0.897 | 0.103 | — |
| cqadupstack-english | 0.6 | alpha+0.1 | 1570.0 | 0.815 | 0.806 | 0.194 | — |
| cqadupstack-english | 0.6 | ndcg@5 | 1570.0 | 0.801 | 0.741 | 0.259 | 0.970 |
| cqadupstack-english | 0.6 | ndcg@20 | 1570.0 | 0.836 | 0.953 | 0.041 | 1.000 |
| cqadupstack-english | 0.6 | mrr@10 | 1570.0 | 0.890 | 0.844 | 0.156 | 0.983 |
| cqadupstack-english | 0.6 | qrels-half | 485.0 | 0.123 | 0.240 | 0.655 | 0.531 |
| cqadupstack-gaming | 0.6 | alpha-0.1 | 1595.0 | 0.852 | 0.908 | 0.092 | — |
| cqadupstack-gaming | 0.6 | alpha+0.1 | 1595.0 | 0.875 | 0.884 | 0.116 | — |
| cqadupstack-gaming | 0.6 | ndcg@5 | 1595.0 | 0.866 | 0.801 | 0.194 | 0.983 |
| cqadupstack-gaming | 0.6 | ndcg@20 | 1595.0 | 0.862 | 0.971 | 0.020 | 0.999 |
| cqadupstack-gaming | 0.6 | mrr@10 | 1595.0 | 0.892 | 0.835 | 0.159 | 0.991 |
| cqadupstack-gaming | 0.6 | qrels-half | 336.0 | -0.011 | 0.167 | 0.740 | 0.615 |
| cqadupstack-physics | 0.6 | alpha-0.1 | 1039.0 | 0.855 | 0.913 | 0.087 | — |
| cqadupstack-physics | 0.6 | alpha+0.1 | 1039.0 | 0.859 | 0.878 | 0.122 | — |
| cqadupstack-physics | 0.6 | ndcg@5 | 1039.0 | 0.813 | 0.739 | 0.257 | 0.970 |
| cqadupstack-physics | 0.6 | ndcg@20 | 1039.0 | 0.837 | 0.978 | 0.017 | 1.000 |
| cqadupstack-physics | 0.6 | mrr@10 | 1039.0 | 0.891 | 0.852 | 0.139 | 0.991 |
| cqadupstack-physics | 0.6 | qrels-half | 298.0 | 0.111 | 0.213 | 0.661 | 0.542 |
| cqadupstack-programmers | 0.7 | alpha-0.1 | 876.0 | 0.831 | 0.940 | 0.060 | — |
| cqadupstack-programmers | 0.7 | alpha+0.1 | 876.0 | 0.834 | 0.814 | 0.186 | — |
| cqadupstack-programmers | 0.7 | ndcg@5 | 876.0 | 0.821 | 0.764 | 0.236 | 0.983 |
| cqadupstack-programmers | 0.7 | ndcg@20 | 876.0 | 0.854 | 0.990 | 0.010 | 1.000 |
| cqadupstack-programmers | 0.7 | mrr@10 | 876.0 | 0.895 | 0.874 | 0.126 | 0.999 |
| cqadupstack-programmers | 0.7 | qrels-half | 248.0 | 0.032 | 0.188 | 0.741 | 0.612 |
| cqadupstack-gis | 0.6 | alpha-0.1 | 885.0 | 0.905 | 0.919 | 0.081 | — |
| cqadupstack-gis | 0.6 | alpha+0.1 | 885.0 | 0.905 | 0.913 | 0.087 | — |
| cqadupstack-gis | 0.6 | ndcg@5 | 885.0 | 0.843 | 0.763 | 0.237 | 0.972 |
| cqadupstack-gis | 0.6 | ndcg@20 | 885.0 | 0.866 | 0.983 | 0.017 | 1.000 |
| cqadupstack-gis | 0.6 | mrr@10 | 885.0 | 0.979 | 0.971 | 0.029 | 0.992 |
| cqadupstack-gis | 0.6 | qrels-half | 133.0 | 0.121 | 0.192 | 0.594 | 0.568 |
