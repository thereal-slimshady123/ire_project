# Niche discovery (exploration data)

Datasets: nfcorpus, scifact, fiqa, cqadupstack-unix, quora, wands, cosqa, cqadupstack-android, cqadupstack-webmasters, arguana, bright-biology, bright-economics, bright-psychology, cqadupstack-mathematica, cqadupstack-english, cqadupstack-gaming, cqadupstack-physics, cqadupstack-programmers, cqadupstack-gis (n = 14004). 5616 niches × 13 actions = 73008 candidates; 682 eligible (n ≥ 200, ≥ 5 datasets, positive in both halves).

| Rank | Niche | Action | n | Datasets | Mean − hybrid* | 95% LB | Half A | Half B |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | bm25_top1_score in bottom 10% AND bm25_top1_top2_ratio in bottom 10% | hybrid@0.7 | 260 | 5 | +0.0122 | +0.0040 | +0.0075 | +0.0217 |
| 2 | bm25_top1_score in bottom 10% AND bm25_top1_top2_ratio in bottom 10% | hybrid@0.8 | 260 | 5 | +0.0167 | +0.0031 | +0.0124 | +0.0257 |
| 3 | bm25_top1_top2_gap in top 10% AND dense_top1_score in bottom 33% | hybrid@0.5 | 305 | 8 | +0.0128 | +0.0021 | +0.0098 | +0.0194 |
| 4 | mean_idf in bottom 33% AND entity_density in top 10% | hybrid@0.7 | 276 | 5 | +0.0115 | +0.0019 | +0.0159 | +0.0048 |
| 5 | max_idf in top 10% AND jaccard_overlap_top20 in top 33% | hybrid@0.7 | 256 | 6 | +0.0103 | +0.0017 | +0.0044 | +0.0185 |
| 6 | max_idf in top 10% AND dense_top1_score in top 33% | hybrid@0.7 | 325 | 8 | +0.0061 | +0.0014 | +0.0032 | +0.0121 |
| 7 | max_idf in top 20% AND dense_top10_query_term_coverage in top 33% | hybrid@0.6 | 340 | 6 | +0.0075 | +0.0013 | +0.0045 | +0.0113 |
| 8 | max_idf in top 10% AND bm25_top1_top2_gap in top 33% | hybrid@0.6 | 614 | 13 | +0.0038 | +0.0010 | +0.0039 | +0.0035 |
| 9 | bm25_top1_score in top 10% AND dense_top10_query_term_coverage in bottom 33% | hybrid@0.5 | 490 | 10 | +0.0076 | +0.0009 | +0.0063 | +0.0107 |
| 10 | bm25_top1_top2_gap in top 33% AND bm25_score_entropy in bottom 20% | hybrid@0.5 | 1898 | 16 | +0.0040 | +0.0009 | +0.0041 | +0.0037 |
| 11 | bm25_top1_score in top 10% AND bm25_top10_query_term_coverage in bottom 20% | hybrid@0.5 | 464 | 10 | +0.0070 | +0.0009 | +0.0041 | +0.0124 |
| 12 | mean_idf in top 33% AND bm25_top1_score in top 10% | hybrid@0.5 | 709 | 14 | +0.0068 | +0.0008 | +0.0081 | +0.0041 |
| 13 | bm25_top1_score in top 10% AND dense_top10_query_term_coverage in bottom 20% | hybrid@0.5 | 286 | 6 | +0.0103 | +0.0008 | +0.0096 | +0.0120 |
| 14 | rare_term_density in top 20% AND bm25_top1_top2_gap in top 20% | hybrid@0.5 | 693 | 14 | +0.0062 | +0.0008 | +0.0065 | +0.0056 |
| 15 | bm25_top1_score in bottom 10% AND bm25_top1_top2_ratio in bottom 20% | hybrid@0.7 | 500 | 13 | +0.0089 | +0.0008 | +0.0076 | +0.0115 |
| 16 | mean_idf in top 33% AND bm25_top1_top2_gap in top 10% | hybrid@0.5 | 651 | 15 | +0.0056 | +0.0007 | +0.0040 | +0.0086 |
| 17 | rank_correlation_spearman in bottom 33% AND bm25_score_entropy in bottom 20% | hybrid@0.5 | 669 | 14 | +0.0086 | +0.0007 | +0.0091 | +0.0072 |
| 18 | bm25_top1_score in top 10% AND bm25_top1_top2_ratio in top 10% | hybrid@0.5 | 585 | 11 | +0.0046 | +0.0007 | +0.0032 | +0.0069 |
| 19 | bm25_top1_top2_ratio in top 20% AND bm25_score_entropy in bottom 10% | hybrid@0.5 | 955 | 15 | +0.0042 | +0.0006 | +0.0036 | +0.0053 |
| 20 | bm25_top1_top2_ratio in top 20% AND bm25_score_entropy in bottom 20% | hybrid@0.5 | 1683 | 15 | +0.0038 | +0.0005 | +0.0035 | +0.0041 |

Frozen for the confirmatory test: (1) bm25_top1_score in bottom 10% AND bm25_top1_top2_ratio in bottom 10% → hybrid@0.7; (2) bm25_top1_top2_gap in top 10% AND dense_top1_score in bottom 33% → hybrid@0.5; (3) mean_idf in bottom 33% AND entity_density in top 10% → hybrid@0.7; (4) bm25_top1_top2_gap in top 33% AND bm25_score_entropy in bottom 20% → hybrid@0.5
