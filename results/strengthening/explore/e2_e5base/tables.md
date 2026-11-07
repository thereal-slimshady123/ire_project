# E2 replication: seen datasets, e5-base-v2 (exploration)

Datasets: nfcorpus, scifact, arguana, cqadupstack-webmasters, cqadupstack-android, cosqa, cqadupstack-mathematica (n = 4533). Definitions as in `results/preregistration/phase2_explanatory_validation_20261107.md`.

**E2: REJECTED**

- (a) Niche found: **no** (28 (group, retriever) pairs with n ≥ 300, Holm-corrected).
- (b) Cost-aware router − hybrid: **-0.0000 [-0.0009, +0.0009]** → CI upper < +0.005: **yes**.
- (c) Best AUC′ (dense-winner n = 548 vs BM25-winner n = 613): **0.702** [0.672, 0.732] (`bm25_top1_score`).

Secondary: TOST 90% CI [-0.0008, +0.0008]; oracle − hybrid headroom +0.0506; queries with a single retriever ≥ 0.10 above hybrid 15.0%; fallback gate 28/35 folds; router hybrid share 97.5%.

| Grouping | Group | n | Retriever | − hybrid [95% CI] | Holm p | Niche |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| dataset | arguana | 1401 | bm25 | -0.0368 [-0.0464, -0.0271] | 0.0028 | no |
| dataset | arguana | 1401 | dense | -0.1389 [-0.1538, -0.1242] | 0.0028 | no |
| dataset | cosqa | 500 | bm25 | -0.1037 [-0.1290, -0.0799] | 0.0028 | no |
| dataset | cosqa | 500 | dense | +0.0064 [-0.0005, +0.0135] | 0.0768 | no |
| dataset | cqadupstack-android | 699 | bm25 | -0.0707 [-0.0872, -0.0549] | 0.0028 | no |
| dataset | cqadupstack-android | 699 | dense | -0.0250 [-0.0367, -0.0134] | 0.0028 | no |
| dataset | cqadupstack-mathematica | 804 | bm25 | -0.0680 [-0.0854, -0.0505] | 0.0028 | no |
| dataset | cqadupstack-mathematica | 804 | dense | -0.0087 [-0.0153, -0.0021] | 0.043 | no |
| dataset | cqadupstack-webmasters | 506 | bm25 | -0.0791 [-0.1022, -0.0570] | 0.0028 | no |
| dataset | cqadupstack-webmasters | 506 | dense | -0.0140 [-0.0252, -0.0030] | 0.0456 | no |
| dataset | nfcorpus | 323 | bm25 | -0.0522 [-0.0653, -0.0401] | 0.0028 | no |
| dataset | nfcorpus | 323 | dense | -0.0194 [-0.0304, -0.0087] | 0.0054 | no |
| dataset | scifact | 300 | bm25 | -0.0580 [-0.0814, -0.0356] | 0.0028 | no |
| dataset | scifact | 300 | dense | -0.0213 [-0.0410, -0.0017] | 0.0682 | no |
| g_length | 4-7 | 1276 | bm25 | -0.0787 [-0.0922, -0.0652] | 0.0028 | no |
| g_length | 4-7 | 1276 | dense | -0.0083 [-0.0145, -0.0019] | 0.043 | no |
| g_length | 8-15 | 1404 | bm25 | -0.0708 [-0.0838, -0.0585] | 0.0028 | no |
| g_length | 8-15 | 1404 | dense | -0.0144 [-0.0217, -0.0074] | 0.0028 | no |
| g_length | >50 | 1399 | bm25 | -0.0366 [-0.0464, -0.0267] | 0.0028 | no |
| g_length | >50 | 1399 | dense | -0.1387 [-0.1541, -0.1234] | 0.0028 | no |
| g_form | question | 1467 | bm25 | -0.0718 [-0.0837, -0.0602] | 0.0028 | no |
| g_form | question | 1467 | dense | -0.0382 [-0.0475, -0.0291] | 0.0028 | no |
| g_form | statement/keywords | 3066 | bm25 | -0.0576 [-0.0653, -0.0499] | 0.0028 | no |
| g_form | statement/keywords | 3066 | dense | -0.0586 [-0.0659, -0.0513] | 0.0028 | no |
| g_ident | has identifier-like token | 1707 | bm25 | -0.0509 [-0.0609, -0.0408] | 0.0028 | no |
| g_ident | has identifier-like token | 1707 | dense | -0.0836 [-0.0952, -0.0722] | 0.0028 | no |
| g_ident | no identifier | 2826 | bm25 | -0.0690 [-0.0773, -0.0604] | 0.0028 | no |
| g_ident | no identifier | 2826 | dense | -0.0329 [-0.0391, -0.0267] | 0.0028 | no |
