# Experiment 1b: query types where always-hybrid loses

Dense model: `intfloat/e5-small-v2`. Hybrid = one global α chosen by 5-fold CV on the pooled mixed workload (α* ∈ {0.6, 0.7}). nDCG@10; Δ = single − hybrid, paired bootstrap 95% CI and two-sided p; Wilcoxon one-sided (single > hybrid).

| Query type | n | BM25 | Dense | Hybrid (global α) | BM25 − Hybrid [95% CI] (p) | Dense − Hybrid [95% CI] (p) | Best α for this type → nDCG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| quora_conversational | 1000 | 0.9333 | 0.9587 | 0.9677 | -0.0344 [-0.0434, -0.0258] (p=1.0e-04; W p=1.0e+00) | -0.0090 [-0.0140, -0.0042] (p=7.0e-04; W p=1.0e+00) | α=0.6 → 0.9683 |
| unix_plain | 884 | 0.2826 | 0.3453 | 0.3562 | -0.0736 [-0.0885, -0.0594] (p=1.0e-04; W p=1.0e+00) | -0.0109 [-0.0202, -0.0014] (p=2.2e-02; W p=9.8e-01) | α=0.6 → 0.3594 |
| unix_technical | 188 | 0.3458 | 0.4274 | 0.4476 | -0.1018 [-0.1372, -0.0672] (p=1.0e-04; W p=1.0e+00) | -0.0202 [-0.0426, +0.0013] (p=6.7e-02; W p=9.0e-01) | α=0.6 → 0.4562 |
