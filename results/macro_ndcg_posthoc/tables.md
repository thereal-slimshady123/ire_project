# Category-balanced (macro) nDCG@10 over the E2 groupings — POST HOC

Not pre-registered. Built by `scripts/posthoc_macro_ndcg.py` from saved held-out predictions; no retrieval was re-run. **Micro** = every query weighs the same (the usual average over the queries the grouping covers). **Macro** = every group weighs the same (mean of the group means). Groupings and their definitions are E2's (`scripts/analyze_niche_existence.py`); groups with fewer than 30 queries are left out of that grouping. Hybrid = per-fold α* (Experiment 2d). Macro differences: paired bootstrap resampling queries within each group, 10,000 resamples, seed 42. The oracle is an upper bound, not a policy.

## Validation (3,969 queries, 3 datasets)

| Grouping | Groups (n) | Hybrid micro / macro | Router micro / macro | Oracle micro / macro | Router − hybrid, macro [95% CI] | BM25 − hybrid, macro | Dense − hybrid, macro |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dataset | cqadupstack-english (1570), cqadupstack-gaming (1595), cqadupstack-mathematica (804) | 0.4682 / 0.4362 | 0.4683 / 0.4363 | 0.5129 / 0.4802 | +0.0001 [-0.0004, +0.0006] | -0.0808 [-0.0888, -0.0731] | -0.0264 [-0.0311, -0.0217] |
| Length (words) | 1-3 words (184), 16-50 (189), 4-7 (1522), 8-15 (2074) | 0.4682 / 0.4537 | 0.4683 / 0.4535 | 0.5129 / 0.5000 | -0.0002 [-0.0020, +0.0012] | -0.0790 [-0.0924, -0.0661] | -0.0332 [-0.0419, -0.0251] |
| Question form | question (2634), statement/keywords (1335) | 0.4682 / 0.4470 | 0.4683 / 0.4471 | 0.5129 / 0.4924 | +0.0001 [-0.0004, +0.0006] | -0.0816 [-0.0895, -0.0737] | -0.0277 [-0.0329, -0.0226] |
| Identifier-like token | has identifier-like token (1187), no identifier (2782) | 0.4682 / 0.4829 | 0.4683 / 0.4828 | 0.5129 / 0.5255 | -0.0001 [-0.0006, +0.0004] | -0.0942 [-0.1026, -0.0858] | -0.0282 [-0.0340, -0.0227] |

## Exploration (7,235 queries, 13 datasets; 307 BRIGHT queries have no text and count only by dataset)

| Grouping | Groups (n) | Hybrid micro / macro | Router micro / macro | Oracle micro / macro | Router − hybrid, macro [95% CI] | BM25 − hybrid, macro | Dense − hybrid, macro |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dataset | arguana (1401), bright-biology (103), bright-economics (103), bright-psychology (101), cosqa (500), cqadupstack-android (699), cqadupstack-unix (1072), cqadupstack-webmasters (506), fiqa (648), nfcorpus (323), quora (1000), scifact (300), wands (479) | 0.5423 / 0.4451 | 0.5415 / 0.4447 | 0.5855 / 0.4866 | -0.0003 [-0.0013, +0.0007] | -0.0518 [-0.0569, -0.0467] | -0.0381 [-0.0442, -0.0321] |
| Query family | finance QA (648), forum titles (3277), long paragraphs (1708), scientific claims / topics (623), short keyword search (979) | 0.5423 / 0.5149 | 0.5415 / 0.5144 | 0.5855 / 0.5597 | -0.0005 [-0.0015, +0.0005] | -0.0684 [-0.0741, -0.0627] | -0.0451 [-0.0501, -0.0402] |
| Length (words) | 1-3 words (592), 16-50 (358), 4-7 (1960), 8-15 (2619), >50 (1399) | 0.5599 / 0.5707 | 0.5590 / 0.5711 | 0.6034 / 0.6131 | +0.0004 [-0.0008, +0.0020] | -0.0575 [-0.0628, -0.0522] | -0.0453 [-0.0506, -0.0400] |
| Question form | question (3220), statement/keywords (3708) | 0.5599 / 0.5630 | 0.5590 / 0.5620 | 0.6034 / 0.6059 | -0.0010 [-0.0018, -0.0001] | -0.0628 [-0.0675, -0.0582] | -0.0438 [-0.0483, -0.0393] |
| Identifier-like token | has identifier-like token (2190), no identifier (4738) | 0.5599 / 0.5672 | 0.5590 / 0.5661 | 0.6034 / 0.6136 | -0.0011 [-0.0020, -0.0001] | -0.0586 [-0.0637, -0.0536] | -0.0547 [-0.0604, -0.0490] |

Per-group means of every system: `summary.json`.
