# Real-world niche benchmark: feature comparison vs the 288-query synthetic set

### Track 1 (natural, n = 384) vs synthetic (n = 288)

**semantic** (sorted by KS distance; top rows shift most from synthetic)

| Feature | Real P10 / P50 / P90 | Synthetic P10 / P50 / P90 | KS |
| :--- | :---: | :---: | :---: |
| dense_top10_query_term_coverage | 0.143 / 0.262 / 0.533 | 0.0409 / 0.1 / 0.175 | 0.70 |
| mean_idf | 2.28 / 3.25 / 4.43 | 3.59 / 4.28 / 5.13 | 0.61 |
| entity_density | 0 / 0.132 / 0.416 | 0 / 0 / 0.0867 | 0.59 |
| bm25_top10_query_term_coverage | 0.189 / 0.431 / 0.667 | 0.2 / 0.267 / 0.333 | 0.57 |
| rare_term_density | 0 / 0.111 / 0.333 | 0.1 / 0.25 / 0.5 | 0.46 |
| bm25_top1_score | 11.5 / 21.3 / 145 | 13 / 15.5 / 20.9 | 0.44 |
| dense_top1_score | 0.848 / 0.87 / 0.89 | 0.832 / 0.856 / 0.875 | 0.42 |
| query_length | 5 / 12 / 211 | 9 / 12 / 15 | 0.33 |
| oov_fraction | 0 / 0 / 0.0377 | 0 / 0 / 0.143 | 0.32 |
| max_idf | 4.28 / 6.79 / 8.5 | 5.55 / 7.05 / 8.15 | 0.25 |
| jaccard_overlap_top20 | 0 / 0.0811 / 0.212 | 0 / 0.0526 / 0.166 | 0.21 |
| jaccard_overlap_top10 | 0 / 0.0526 / 0.111 | 0 / 0.0526 / 0.176 | 0.20 |
| dense_score_entropy | 0.977 / 0.991 / 0.998 | 0.968 / 0.989 / 0.996 | 0.19 |
| bm25_top1_top2_ratio | 1 / 1.06 / 1.23 | 1.01 / 1.07 / 1.3 | 0.16 |
| rank_correlation_spearman | -0.426 / 0 / 0.8 | -0.47 / 0 / 0.68 | 0.14 |
| dense_top1_top2_gap | 2.43e-06 / 0.104 / 0.319 | 0.0232 / 0.145 / 0.396 | 0.13 |
| bm25_top1_top2_gap | 0.000313 / 0.128 / 0.37 | 0.0256 / 0.109 / 0.365 | 0.11 |
| bm25_score_entropy | 0.959 / 0.988 / 0.996 | 0.968 / 0.987 / 0.995 | 0.07 |

**lexical** (sorted by KS distance; top rows shift most from synthetic)

| Feature | Real P10 / P50 / P90 | Synthetic P10 / P50 / P90 | KS |
| :--- | :---: | :---: | :---: |
| rare_term_density | 0 / 0.25 / 0.564 | 0.758 / 1 / 1 | 0.88 |
| mean_idf | 2.6 / 3.67 / 5.8 | 5.8 / 7.06 / 7.8 | 0.83 |
| dense_top10_query_term_coverage | 0.263 / 0.465 / 0.7 | 0 / 0.112 / 0.238 | 0.83 |
| bm25_top10_query_term_coverage | 0.438 / 0.6 / 0.8 | 0.267 / 0.375 / 0.933 | 0.62 |
| bm25_score_entropy | 0.742 / 0.931 / 0.985 | 0 / 0.67 / 0.913 | 0.60 |
| bm25_top1_top2_ratio | 1.18 / 1.48 / 2.2 | 1.02 / 2.74 / 4.9 | 0.59 |
| dense_top1_score | 0.858 / 0.891 / 0.925 | 0.815 / 0.853 / 0.89 | 0.51 |
| bm25_top1_score | 15.5 / 26.9 / 48.3 | 28.2 / 41.7 / 61.6 | 0.48 |
| query_length | 2 / 8 / 15 | 3 / 5 / 7 | 0.47 |
| max_idf | 5.22 / 7.27 / 9.63 | 7.28 / 7.79 / 8.15 | 0.42 |
| entity_density | 0 / 0.143 / 0.6 | 0 / 0.5 / 1 | 0.42 |
| bm25_top1_top2_gap | 0.26 / 0.524 / 0.77 | 0.0178 / 0.73 / 0.904 | 0.39 |
| rank_correlation_spearman | 0 / 0.4 / 1 | 0 / 0 / 0.889 | 0.37 |
| jaccard_overlap_top20 | 0.0526 / 0.111 / 0.25 | 0 / 0.0526 / 0.217 | 0.36 |
| dense_top1_top2_gap | 0.0692 / 0.383 / 0.621 | 0.0394 / 0.18 / 0.562 | 0.30 |
| jaccard_overlap_top10 | 0.0526 / 0.111 / 0.176 | 0 / 0.1 / 0.323 | 0.28 |
| dense_score_entropy | 0.921 / 0.969 / 0.994 | 0.931 / 0.981 / 0.994 | 0.22 |
| oov_fraction | 0 / 0 / 0 | 0 / 0 / 0 | 0.01 |

### Track 2 (contrastive, n = 628) vs synthetic (n = 288)

**semantic** (sorted by KS distance; top rows shift most from synthetic)

| Feature | Real P10 / P50 / P90 | Synthetic P10 / P50 / P90 | KS |
| :--- | :---: | :---: | :---: |
| dense_top10_query_term_coverage | 0.144 / 0.254 / 0.585 | 0.0409 / 0.1 / 0.175 | 0.71 |
| mean_idf | 2.4 / 3.23 / 4.12 | 3.59 / 4.28 / 5.13 | 0.67 |
| entity_density | 0 / 0.189 / 0.404 | 0 / 0 / 0.0867 | 0.65 |
| bm25_top1_score | 13.7 / 33.6 / 185 | 13 / 15.5 / 20.9 | 0.63 |
| jaccard_overlap_top20 | 0.0526 / 0.176 / 0.429 | 0 / 0.0526 / 0.166 | 0.57 |
| rare_term_density | 0 / 0.108 / 0.25 | 0.1 / 0.25 / 0.5 | 0.50 |
| jaccard_overlap_top10 | 0.0526 / 0.176 / 0.429 | 0 / 0.0526 / 0.176 | 0.49 |
| bm25_top10_query_term_coverage | 0.172 / 0.393 / 0.697 | 0.2 / 0.267 / 0.333 | 0.49 |
| dense_top1_score | 0.85 / 0.874 / 0.902 | 0.832 / 0.856 / 0.875 | 0.47 |
| query_length | 5 / 15 / 220 | 9 / 12 / 15 | 0.45 |
| rank_correlation_spearman | -0.4 / 0.3 / 0.8 | -0.47 / 0 / 0.68 | 0.41 |
| oov_fraction | 0 / 0 / 0.0355 | 0 / 0 / 0.143 | 0.36 |
| max_idf | 4.18 / 7.13 / 8.5 | 5.55 / 7.05 / 8.15 | 0.22 |
| bm25_top1_top2_ratio | 1 / 1.05 / 1.25 | 1.01 / 1.07 / 1.3 | 0.18 |
| dense_score_entropy | 0.968 / 0.986 / 0.995 | 0.968 / 0.989 / 0.996 | 0.18 |
| bm25_top1_top2_gap | 0.00644 / 0.101 / 0.329 | 0.0256 / 0.109 / 0.365 | 0.15 |
| bm25_score_entropy | 0.953 / 0.984 / 0.996 | 0.968 / 0.987 / 0.995 | 0.15 |
| dense_top1_top2_gap | 0.00853 / 0.129 / 0.334 | 0.0232 / 0.145 / 0.396 | 0.08 |

**lexical** (sorted by KS distance; top rows shift most from synthetic)

| Feature | Real P10 / P50 / P90 | Synthetic P10 / P50 / P90 | KS |
| :--- | :---: | :---: | :---: |
| mean_idf | 2.53 / 3.22 / 4.44 | 5.8 / 7.06 / 7.8 | 0.96 |
| rare_term_density | 0 / 0.111 / 0.333 | 0.758 / 1 / 1 | 0.96 |
| bm25_score_entropy | 0.944 / 0.981 / 0.995 | 0 / 0.67 / 0.913 | 0.89 |
| bm25_top1_top2_ratio | 1 / 1.11 / 1.35 | 1.02 / 2.74 / 4.9 | 0.81 |
| bm25_top1_top2_gap | 0.00751 / 0.165 / 0.417 | 0.0178 / 0.73 / 0.904 | 0.76 |
| query_length | 5 / 14 / 260 | 3 / 5 / 7 | 0.72 |
| dense_top10_query_term_coverage | 0.148 / 0.254 / 0.611 | 0 / 0.112 / 0.238 | 0.59 |
| entity_density | 0 / 0.155 / 0.416 | 0 / 0.5 / 1 | 0.50 |
| jaccard_overlap_top20 | 0.0526 / 0.212 / 0.379 | 0 / 0.0526 / 0.217 | 0.47 |
| max_idf | 4.27 / 7.2 / 8.5 | 7.28 / 7.79 / 8.15 | 0.44 |
| bm25_top1_score | 15.5 / 32.5 / 201 | 28.2 / 41.7 / 61.6 | 0.41 |
| oov_fraction | 0 / 0 / 0.0349 | 0 / 0 / 0 | 0.37 |
| bm25_top10_query_term_coverage | 0.176 / 0.38 / 0.75 | 0.267 / 0.375 / 0.933 | 0.35 |
| dense_top1_score | 0.844 / 0.872 / 0.895 | 0.815 / 0.853 / 0.89 | 0.35 |
| dense_score_entropy | 0.975 / 0.99 / 0.996 | 0.931 / 0.981 / 0.994 | 0.33 |
| rank_correlation_spearman | -0.196 / 0.215 / 0.781 | 0 / 0 / 0.889 | 0.31 |
| jaccard_overlap_top10 | 0.0526 / 0.176 / 0.429 | 0 / 0.1 / 0.323 | 0.30 |
| dense_top1_top2_gap | 0.0134 / 0.114 / 0.28 | 0.0394 / 0.18 / 0.562 | 0.27 |

