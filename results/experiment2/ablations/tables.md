# Feature ablations: held-out routed nDCG@10 minus always-hybrid

5-fold CV within each dataset (Experiment 2 folds); alpha*, targets, theta and models fitted on training folds only. Cell = mean paired difference [95% bootstrap CI]; negative = worse than always-hybrid. 'Hybrid' is the always-hybrid mean nDCG@10 the differences are measured against.

## Router: `advantage`

| Features (n) | nfcorpus | scifact | fiqa | cqadupstack-unix | quora | **pooled** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| query_only (6) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| retrieval_signal (12) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0008 [-0.0026, +0.0006] | +0.0000 [+0.0000, +0.0000] | -0.0003 [-0.0008, +0.0002] |
| pre_dense (11) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0007 [+0.0000, +0.0018] | +0.0000 [+0.0000, +0.0000] | +0.0002 [+0.0000, +0.0006] |
| all (18) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0003, +0.0000] | -0.0003 [-0.0020, +0.0010] | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0007, +0.0003] |

Route shares (BM25 / dense / hybrid), pooled: query_only: 0%/0%/100%; retrieval_signal: 0%/1%/99%; pre_dense: 0%/1%/99%; all: 0%/1%/99%

## Router: `advantage_fixed`

| Features (n) | nfcorpus | scifact | fiqa | cqadupstack-unix | quora | **pooled** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| query_only (6) | -0.0003 [-0.0018, +0.0012] | -0.0041 [-0.0178, +0.0096] | -0.0027 [-0.0059, -0.0000] | -0.0029 [-0.0060, -0.0001] | -0.0006 [-0.0016, +0.0000] | -0.0020 [-0.0038, -0.0004] |
| retrieval_signal (12) | -0.0033 [-0.0079, +0.0005] | +0.0034 [-0.0026, +0.0101] | -0.0014 [-0.0054, +0.0024] | -0.0033 [-0.0071, +0.0004] | -0.0015 [-0.0045, +0.0016] | -0.0018 [-0.0036, -0.0000] |
| pre_dense (11) | +0.0001 [-0.0030, +0.0030] | -0.0032 [-0.0164, +0.0108] | -0.0050 [-0.0098, -0.0005] | -0.0007 [-0.0042, +0.0027] | +0.0005 [-0.0008, +0.0019] | -0.0013 [-0.0033, +0.0006] |
| all (18) | -0.0034 [-0.0078, +0.0003] | +0.0002 [-0.0096, +0.0102] | -0.0000 [-0.0040, +0.0040] | -0.0026 [-0.0060, +0.0008] | +0.0008 [-0.0016, +0.0036] | -0.0009 [-0.0028, +0.0009] |

Route shares (BM25 / dense / hybrid), pooled: query_only: 1%/7%/92%; retrieval_signal: 3%/11%/86%; pre_dense: 2%/10%/88%; all: 3%/11%/87%

## Router: `gbm_cls`

| Features (n) | nfcorpus | scifact | fiqa | cqadupstack-unix | quora | **pooled** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| query_only (6) | -0.0170 [-0.0268, -0.0078] | -0.0163 [-0.0353, +0.0025] | -0.0363 [-0.0479, -0.0256] | -0.0201 [-0.0282, -0.0122] | -0.0047 [-0.0077, -0.0018] | -0.0180 [-0.0220, -0.0140] |
| retrieval_signal (12) | -0.0119 [-0.0199, -0.0044] | +0.0014 [-0.0122, +0.0152] | -0.0206 [-0.0297, -0.0118] | -0.0118 [-0.0190, -0.0048] | -0.0035 [-0.0067, -0.0006] | -0.0099 [-0.0132, -0.0065] |
| pre_dense (11) | -0.0218 [-0.0313, -0.0129] | -0.0121 [-0.0287, +0.0049] | -0.0323 [-0.0431, -0.0225] | -0.0098 [-0.0161, -0.0038] | -0.0054 [-0.0095, -0.0017] | -0.0142 [-0.0178, -0.0108] |
| all (18) | -0.0156 [-0.0241, -0.0077] | -0.0021 [-0.0135, +0.0091] | -0.0259 [-0.0356, -0.0170] | -0.0090 [-0.0159, -0.0023] | -0.0007 [-0.0038, +0.0027] | -0.0098 [-0.0132, -0.0065] |

Route shares (BM25 / dense / hybrid), pooled: query_only: 15%/29%/56%; retrieval_signal: 10%/22%/69%; pre_dense: 9%/23%/68%; all: 8%/20%/71%

## Always-hybrid mean nDCG@10

| nfcorpus | scifact | fiqa | cqadupstack-unix | quora | pooled |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.3589 | 0.7264 | 0.4029 | 0.3764 | 0.9674 | 0.5880 |

## Gain importance of the advantage regressors (all features, mean over datasets)

| Feature | Delta_bm25 regressor | Delta_dense regressor |
| :--- | :---: | :---: |
| dense_score_entropy | 0.101 | 0.138 |
| bm25_score_entropy | 0.123 | 0.081 |
| dense_top1_top2_gap | 0.086 | 0.111 |
| dense_top1_score | 0.063 | 0.122 |
| bm25_top1_top2_ratio | 0.099 | 0.058 |
| bm25_top1_score | 0.071 | 0.080 |
| rank_correlation_spearman | 0.085 | 0.038 |
| bm25_top1_top2_gap | 0.066 | 0.054 |
| mean_idf | 0.059 | 0.051 |
| dense_top10_query_term_coverage | 0.054 | 0.043 |
| bm25_top10_query_term_coverage | 0.042 | 0.055 |
| max_idf | 0.038 | 0.048 |
| jaccard_overlap_top20 | 0.023 | 0.030 |
| query_length | 0.020 | 0.029 |
| jaccard_overlap_top10 | 0.030 | 0.016 |
| rare_term_density | 0.019 | 0.026 |
| entity_density | 0.021 | 0.016 |
| oov_fraction | 0.000 | 0.003 |
