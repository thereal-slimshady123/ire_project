# E2 replication: fresh, all six, e5-base-v2

Datasets: cqadupstack-stats, cqadupstack-wordpress, cqadupstack-tex, scidocs, nq, hotpotqa (n = 15956). Definitions as in `results/preregistration/phase2_explanatory_validation_20261107.md`.

**E2: ACCEPTED**

- (a) Niche found: **no** (26 (group, retriever) pairs with n ≥ 300, Holm-corrected).
- (b) Cost-aware router − hybrid: **-0.0003 [-0.0006, +0.0000]** → CI upper < +0.005: **yes**.
- (c) Best AUC′ (dense-winner n = 1498 vs BM25-winner n = 975): **0.571** [0.548, 0.595] (`bm25_score_entropy`; within-dataset percentile features). Other definition: raw 0.553 [0.531, 0.576] (`bm25_top1_score`), percentile 0.571 [0.548, 0.595] (`bm25_score_entropy`).

Secondary: TOST 90% CI [-0.0006, -0.0000]; oracle − hybrid headroom +0.0244; queries with a single retriever ≥ 0.10 above hybrid 7.8%; fallback gate 22/30 folds; router hybrid share 99.3%.

| Grouping | Group | n | Retriever | − hybrid [95% CI] | Holm p | Niche |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| dataset | cqadupstack-stats | 652 | bm25 | -0.0525 [-0.0700, -0.0357] | 0.0026 | no |
| dataset | cqadupstack-stats | 652 | dense | -0.0204 [-0.0321, -0.0090] | 0.0026 | no |
| dataset | cqadupstack-tex | 2906 | bm25 | -0.0509 [-0.0590, -0.0431] | 0.0026 | no |
| dataset | cqadupstack-tex | 2906 | dense | -0.0310 [-0.0370, -0.0252] | 0.0026 | no |
| dataset | cqadupstack-wordpress | 541 | bm25 | -0.0643 [-0.0848, -0.0446] | 0.0026 | no |
| dataset | cqadupstack-wordpress | 541 | dense | -0.0338 [-0.0478, -0.0199] | 0.0026 | no |
| dataset | hotpotqa | 7405 | bm25 | -0.0722 [-0.0756, -0.0687] | 0.0026 | no |
| dataset | hotpotqa | 7405 | dense | -0.0226 [-0.0246, -0.0205] | 0.0026 | no |
| dataset | nq | 3452 | bm25 | -0.1980 [-0.2089, -0.1873] | 0.0026 | no |
| dataset | nq | 3452 | dense | -0.0055 [-0.0083, -0.0028] | 0.0026 | no |
| dataset | scidocs | 1000 | bm25 | -0.0328 [-0.0403, -0.0254] | 0.0026 | no |
| dataset | scidocs | 1000 | dense | -0.0017 [-0.0051, +0.0020] | 0.365 | no |
| g_length | 16-50 | 3533 | bm25 | -0.0686 [-0.0736, -0.0638] | 0.0026 | no |
| g_length | 16-50 | 3533 | dense | -0.0287 [-0.0318, -0.0255] | 0.0026 | no |
| g_length | 4-7 | 2332 | bm25 | -0.0538 [-0.0627, -0.0449] | 0.0026 | no |
| g_length | 4-7 | 2332 | dense | -0.0273 [-0.0337, -0.0211] | 0.0026 | no |
| g_length | 8-15 | 9841 | bm25 | -0.1101 [-0.1150, -0.1053] | 0.0026 | no |
| g_length | 8-15 | 9841 | dense | -0.0138 [-0.0158, -0.0118] | 0.0026 | no |
| g_form | question | 11885 | bm25 | -0.1038 [-0.1078, -0.0998] | 0.0026 | no |
| g_form | question | 11885 | dense | -0.0182 [-0.0200, -0.0164] | 0.0026 | no |
| g_form | statement/keywords | 4071 | bm25 | -0.0575 [-0.0642, -0.0510] | 0.0026 | no |
| g_form | statement/keywords | 4071 | dense | -0.0230 [-0.0271, -0.0188] | 0.0026 | no |
| g_ident | has identifier-like token | 2618 | bm25 | -0.0779 [-0.0851, -0.0705] | 0.0026 | no |
| g_ident | has identifier-like token | 2618 | dense | -0.0275 [-0.0323, -0.0229] | 0.0026 | no |
| g_ident | no identifier | 13338 | bm25 | -0.0948 [-0.0987, -0.0909] | 0.0026 | no |
| g_ident | no identifier | 13338 | dense | -0.0178 [-0.0197, -0.0160] | 0.0026 | no |
