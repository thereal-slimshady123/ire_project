# E2 replication: fresh non-forum, e5-small-v2

Datasets: scidocs, nq, hotpotqa (n = 11857). Definitions as in `results/preregistration/phase2_explanatory_validation_20261107.md`.

**E2: ACCEPTED**

- (a) Niche found: **no** (20 (group, retriever) pairs with n ≥ 300, Holm-corrected).
- (b) Cost-aware router − hybrid: **-0.0004 [-0.0008, -0.0001]** → CI upper < +0.005: **yes**.
- (c) Best AUC′ (dense-winner n = 972 vs BM25-winner n = 735): **0.588** [0.562, 0.615] (`dense_top1_score`; within-dataset percentile features). Other definition: raw 0.633 [0.606, 0.658] (`query_length`), percentile 0.588 [0.562, 0.615] (`dense_top1_score`).

Secondary: TOST 90% CI [-0.0007, -0.0002]; oracle − hybrid headroom +0.0187; queries with a single retriever ≥ 0.10 above hybrid 6.4%; fallback gate 20/30 folds; router hybrid share 99.0%.

| Grouping | Group | n | Retriever | − hybrid [95% CI] | Holm p | Niche |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| dataset | hotpotqa | 7405 | bm25 | -0.0597 [-0.0627, -0.0566] | 0.002 | no |
| dataset | hotpotqa | 7405 | dense | -0.0276 [-0.0301, -0.0250] | 0.002 | no |
| dataset | nq | 3452 | bm25 | -0.1928 [-0.2034, -0.1823] | 0.002 | no |
| dataset | nq | 3452 | dense | -0.0066 [-0.0093, -0.0039] | 0.002 | no |
| dataset | scidocs | 1000 | bm25 | -0.0312 [-0.0381, -0.0246] | 0.002 | no |
| dataset | scidocs | 1000 | dense | -0.0098 [-0.0147, -0.0047] | 0.002 | no |
| g_length | 16-50 | 3402 | bm25 | -0.0572 [-0.0617, -0.0527] | 0.002 | no |
| g_length | 16-50 | 3402 | dense | -0.0374 [-0.0415, -0.0335] | 0.002 | no |
| g_length | 4-7 | 509 | bm25 | -0.0677 [-0.0855, -0.0506] | 0.002 | no |
| g_length | 4-7 | 509 | dense | -0.0112 [-0.0182, -0.0044] | 0.002 | no |
| g_length | 8-15 | 7934 | bm25 | -0.1147 [-0.1199, -0.1095] | 0.002 | no |
| g_length | 8-15 | 7934 | dense | -0.0131 [-0.0151, -0.0111] | 0.002 | no |
| g_form | question | 10273 | bm25 | -0.1002 [-0.1044, -0.0960] | 0.002 | no |
| g_form | question | 10273 | dense | -0.0215 [-0.0236, -0.0196] | 0.002 | no |
| g_form | statement/keywords | 1584 | bm25 | -0.0690 [-0.0790, -0.0589] | 0.002 | no |
| g_form | statement/keywords | 1584 | dense | -0.0098 [-0.0139, -0.0058] | 0.002 | no |
| g_ident | has identifier-like token | 1940 | bm25 | -0.0691 [-0.0759, -0.0622] | 0.002 | no |
| g_ident | has identifier-like token | 1940 | dense | -0.0354 [-0.0409, -0.0300] | 0.002 | no |
| g_ident | no identifier | 9917 | bm25 | -0.1013 [-0.1057, -0.0969] | 0.002 | no |
| g_ident | no identifier | 9917 | dense | -0.0169 [-0.0188, -0.0151] | 0.002 | no |
