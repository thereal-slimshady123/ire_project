# E2 replication: fresh forums, e5-small-v2

Datasets: cqadupstack-stats, cqadupstack-wordpress, cqadupstack-tex (n = 4099). Definitions as in `results/preregistration/phase2_explanatory_validation_20261107.md`.

**E2: ACCEPTED**

- (a) Niche found: **no** (18 (group, retriever) pairs with n ≥ 300, Holm-corrected).
- (b) Cost-aware router − hybrid: **-0.0011 [-0.0019, -0.0002]** → CI upper < +0.005: **yes**.
- (c) Best AUC′ (dense-winner n = 381 vs BM25-winner n = 373): **0.583** [0.542, 0.624] (`query_length`; within-dataset percentile features). Other definition: raw 0.592 [0.551, 0.632] (`mean_idf`), percentile 0.583 [0.542, 0.624] (`query_length`).

Secondary: TOST 90% CI [-0.0018, -0.0004]; oracle − hybrid headroom +0.0449; queries with a single retriever ≥ 0.10 above hybrid 13.5%; fallback gate 20/30 folds; router hybrid share 97.7%.

| Grouping | Group | n | Retriever | − hybrid [95% CI] | Holm p | Niche |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| dataset | cqadupstack-stats | 652 | bm25 | -0.0508 [-0.0657, -0.0360] | 0.0018 | no |
| dataset | cqadupstack-stats | 652 | dense | -0.0195 [-0.0308, -0.0086] | 0.0018 | no |
| dataset | cqadupstack-tex | 2906 | bm25 | -0.0405 [-0.0478, -0.0333] | 0.0018 | no |
| dataset | cqadupstack-tex | 2906 | dense | -0.0267 [-0.0327, -0.0208] | 0.0018 | no |
| dataset | cqadupstack-wordpress | 541 | bm25 | -0.0630 [-0.0815, -0.0446] | 0.0018 | no |
| dataset | cqadupstack-wordpress | 541 | dense | -0.0288 [-0.0416, -0.0162] | 0.0018 | no |
| g_length | 4-7 | 1823 | bm25 | -0.0403 [-0.0499, -0.0309] | 0.0018 | no |
| g_length | 4-7 | 1823 | dense | -0.0305 [-0.0378, -0.0234] | 0.0018 | no |
| g_length | 8-15 | 1907 | bm25 | -0.0488 [-0.0576, -0.0398] | 0.0018 | no |
| g_length | 8-15 | 1907 | dense | -0.0178 [-0.0249, -0.0106] | 0.0018 | no |
| g_form | question | 1612 | bm25 | -0.0596 [-0.0697, -0.0497] | 0.0018 | no |
| g_form | question | 1612 | dense | -0.0153 [-0.0232, -0.0078] | 0.0018 | no |
| g_form | statement/keywords | 2487 | bm25 | -0.0356 [-0.0438, -0.0275] | 0.0018 | no |
| g_form | statement/keywords | 2487 | dense | -0.0327 [-0.0390, -0.0264] | 0.0018 | no |
| g_ident | has identifier-like token | 678 | bm25 | -0.0522 [-0.0690, -0.0361] | 0.0018 | no |
| g_ident | has identifier-like token | 678 | dense | -0.0306 [-0.0429, -0.0185] | 0.0018 | no |
| g_ident | no identifier | 3421 | bm25 | -0.0437 [-0.0504, -0.0370] | 0.0018 | no |
| g_ident | no identifier | 3421 | dense | -0.0249 [-0.0301, -0.0198] | 0.0018 | no |
