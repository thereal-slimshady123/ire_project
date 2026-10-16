# Experiment 1 results: fixed-α hybrid degradation

ε = 0.02; hybrid uses α* selected on training folds (5-fold CV, out-of-fold scores). Metric: nDCG@10.

## Query win/loss table

| Dataset | Total Queries | Hybrid Wins (%) | BM25 Wins (%) | Dense Wins (%) | Near-tie, hybrid degraded (%) | Near-tie, hybrid safe (%) | Ties (%) | Total Degraded by Hybrid (%) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 11.1% | 13.0% | 12.7% | 0.9% | 26.9% | 35.3% | **26.6%** [22.1%, 31.7%] |
| scifact | 300 | 5.7% | 7.7% | 9.7% | 0.0% | 31.7% | 45.3% | **17.3%** [13.5%, 22.0%] |
| fiqa | 648 | 13.0% | 12.5% | 12.8% | 0.2% | 17.7% | 43.8% | **25.5%** [22.3%, 29.0%] |

## Oracle headroom table

| Dataset | BM25 nDCG | Dense nDCG | Best Hybrid (α*) | Oracle Router | Headroom (Δ) [95% CI] | p (bootstrap) | p (Wilcoxon) | Oracle{BM25,Dense} − Hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 0.3223 | 0.2675 | 0.3432 (α*∈{0.4}) | 0.3694 | **0.0262** [0.0202, 0.0327] | 1.0e-04 | 4.4e-20 | +0.0185 |
| scifact | 0.6827 | 0.5293 | 0.7032 (α*∈{0.2, 0.4}) | 0.7500 | **0.0468** [0.0318, 0.0632] | 1.0e-04 | 7.5e-11 | +0.0357 |
| fiqa | 0.2528 | 0.2452 | 0.3049 (α*∈{0.4, 0.5}) | 0.3530 | **0.0481** [0.0402, 0.0565] | 1.0e-04 | 4.2e-31 | +0.0316 |

## Hypothesis verdicts

| Dataset | H1: degraded ≥ 15% (CI lower bound) | H2: headroom ≥ 0.02, p < 0.01 | H1 w/ in-sample α | H2 w/ in-sample α |
| :--- | :---: | :---: | :---: | :---: |
| nfcorpus | ✅ (22.1%) | ✅ | ✅ | ✅ |
| scifact | ❌ (13.5%) | ✅ | ❌ | ✅ |
| fiqa | ✅ (22.3%) | ✅ | ✅ | ✅ |

## H3 diagnostics (one-sided Mann-Whitney U, p-values)

| Dataset | Max-IDF top quartile → larger Δ_BM25 | Low dense keyword coverage → larger Δ_Dense | Low BM25 top-gap → larger Δ_Dense | Jaccard@10 = 0 → larger Δ_MaxSingle |
| :--- | :---: | :---: | :---: | :---: |
| nfcorpus | -0.017 vs -0.022 (p=2.0e-01) | -0.098 vs -0.068 (p=6.5e-01) | -0.071 vs -0.077 (p=6.1e-01) | +0.004 vs +0.022 (p=1.0e+00) |
| scifact | -0.009 vs -0.024 (p=3.5e-01) | -0.289 vs -0.133 (p=1.0e+00) | -0.078 vs -0.206 (p=7.0e-02) | +0.091 vs +0.030 (p=2.1e-01) |
| fiqa | -0.040 vs -0.056 (p=1.7e-02) | -0.085 vs -0.051 (p=8.7e-01) | -0.062 vs -0.059 (p=5.5e-01) | +0.035 vs +0.031 (p=1.8e-01) |
