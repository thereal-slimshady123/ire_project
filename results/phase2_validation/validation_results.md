# Phase 2 validation: explanatory hypotheses E1–E2

Pre-registered in `results/preregistration/phase2_explanatory_validation_20261107.md` (hash in `SHA256SUMS.txt`). Validation data: CQADupStack mathematica, english, gaming (n = 3969). nDCG@10, paired bootstrap (10000 resamples), δ = 0.005.

## E1 — Distribution shift: **ACCEPTED**

Synthetic-trained Exp 2d router (gate on synthetic data: eps=0) − hybrid (α = 0.6), pooled: **-0.0209 [-0.0256, -0.0162]**, p = 1.0e-04. Rule: accept if CI upper < −0.005.

| Dataset | n | Synthetic-trained router − hybrid | Route share BM25 / dense / hybrid |
| :--- | :---: | :---: | :---: |
| cqadupstack-mathematica | 804 | -0.0199 [-0.0289, -0.0107] | 5.2% / 75.2% / 19.5% |
| cqadupstack-english | 1570 | -0.0281 [-0.0360, -0.0200] | 4.3% / 62.1% / 33.6% |
| cqadupstack-gaming | 1595 | -0.0142 [-0.0213, -0.0074] | 1.9% / 60.9% / 37.2% |

Secondary: winner-prediction AUC' — synthetic **1.000** [1.000, 1.000] (`query_length`) vs validation **0.597** [0.557, 0.636] (`jaccard_overlap_top10`).

## E2 — No pre-retrieval-identifiable niche: **ACCEPTED**

- (a) Niche found: **no** (18 (group, retriever) pairs with n ≥ 300, Holm-corrected; table below).
- (b) Cost-aware router − hybrid: **+0.0001 [-0.0004, +0.0005]** → CI upper < +0.005: **yes**.
- (c) Best AUC' (dense-winner n = 505 vs BM25-winner n = 328): **0.597** [0.557, 0.636] (`jaccard_overlap_top10`) → < 0.65 and CI upper < 0.70: **yes**.

Secondary: TOST 90% CI [-0.0003, +0.0004] (equivalent within ±0.005: yes); oracle − hybrid headroom +0.0447; queries with a single retriever ≥ 0.10 above hybrid 14.0%; fallback gate 13/15 folds; router hybrid share 99.3%.

| Grouping | Group | n | Retriever | − hybrid [95% CI] | Holm p | Niche |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| dataset | cqadupstack-english | 1570 | bm25 | -0.1014 [-0.1142, -0.0891] | 0.002 | no |
| dataset | cqadupstack-english | 1570 | dense | -0.0361 [-0.0450, -0.0273] | 0.002 | no |
| dataset | cqadupstack-gaming | 1595 | bm25 | -0.0867 [-0.0988, -0.0748] | 0.002 | no |
| dataset | cqadupstack-gaming | 1595 | dense | -0.0199 [-0.0275, -0.0124] | 0.002 | no |
| dataset | cqadupstack-mathematica | 804 | bm25 | -0.0542 [-0.0700, -0.0389] | 0.002 | no |
| dataset | cqadupstack-mathematica | 804 | dense | -0.0231 [-0.0308, -0.0155] | 0.002 | no |
| length | 4-7 | 1522 | bm25 | -0.0863 [-0.0986, -0.0742] | 0.002 | no |
| length | 4-7 | 1522 | dense | -0.0292 [-0.0371, -0.0215] | 0.002 | no |
| length | 8-15 | 2074 | bm25 | -0.0884 [-0.0989, -0.0778] | 0.002 | no |
| length | 8-15 | 2074 | dense | -0.0229 [-0.0297, -0.0161] | 0.002 | no |
| form | question | 2634 | bm25 | -0.0949 [-0.1046, -0.0855] | 0.002 | no |
| form | question | 2634 | dense | -0.0254 [-0.0315, -0.0193] | 0.002 | no |
| form | statement/keywords | 1335 | bm25 | -0.0682 [-0.0809, -0.0560] | 0.002 | no |
| form | statement/keywords | 1335 | dense | -0.0301 [-0.0383, -0.0218] | 0.002 | no |
| ident | has identifier-like token | 1187 | bm25 | -0.1146 [-0.1288, -0.1008] | 0.002 | no |
| ident | has identifier-like token | 1187 | dense | -0.0315 [-0.0415, -0.0218] | 0.002 | no |
| ident | no identifier | 2782 | bm25 | -0.0738 [-0.0828, -0.0650] | 0.002 | no |
| ident | no identifier | 2782 | dense | -0.0250 [-0.0306, -0.0195] | 0.002 | no |

## E3 — Skipping dense retrieval (secondary, descriptive)

See `results/phase2_validation/experiment3/tables.md`. Pre-recorded prediction: skip rate < 20% on forum queries.
