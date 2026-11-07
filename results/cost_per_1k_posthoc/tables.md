# Cost per 1,000 queries and cost–quality frontier — POST HOC

Not pre-registered. Built by `scripts/posthoc_cost_per_1k.py` from saved results only (`results/policy_latency/`, `results/experiment2/cache/`); no retrieval was re-run. Held-out nDCG@10; latency measured on an RTX 2050 (dense encoder, cross-encoder) and CPU (BM25, FAISS, fusion, features), one query at a time (report §3.5). **Cost = compute time: X ms per query = X seconds per 1,000 queries.** To convert to money, multiply seconds per 1,000 queries / 3,600 by the hourly price of the hardware. Oracle selection is excluded (it needs relevance labels).

## nfcorpus

| Policy | nDCG@10 | − hybrid | Seconds per 1,000 queries (mean) | p95 ms per query | × hybrid cost | GPU share | On frontier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.3223 | -0.0366 | 0.4 | 0.5 | 0.05 | 0% | yes |
| Dense | 0.3255 | -0.0334 | 6.2 | 7.9 | 0.89 | 93% | yes |
| Hybrid | 0.3589 | +0.0000 | 7.0 | 8.9 | 1.00 | 83% | yes |
| GBM router | 0.3550 | -0.0039 | 12.0 | 15.8 | 1.72 | — |  |
| Skip-dense router | 0.3537 | -0.0052 | 7.0 | 11.1 | 1.00 | — | yes |
| Predicted-gain escalation 10% | 0.3599 | +0.0010 | 50.4 | 414.2 | 7.20 | — |  |
| Router + τ escalation 10% | 0.3556 | -0.0033 | 44.0 | 404.5 | 6.29 | — |  |
| Random escalation 10% | 0.3610 | +0.0021 | 44.4 | 409.5 | 6.35 | — | yes |
| Always rerank (MiniLM-L6) | 0.3546 | -0.0043 | 405.6 | 439.8 | 57.95 | 100% |  |

Frontier (cheapest first): BM25 (0.4 s, 0.3223) → Dense (6.2 s, 0.3255) → Skip-dense router (7.0 s, 0.3537) → Hybrid (7.0 s, 0.3589) → Random escalation 5% (25.4 s, 0.3592) → Predicted-gain escalation 5% (31.7 s, 0.3594) → Random escalation 10% (44.4 s, 0.3610)

## scifact

| Policy | nDCG@10 | − hybrid | Seconds per 1,000 queries (mean) | p95 ms per query | × hybrid cost | GPU share | On frontier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.6827 | -0.0437 | 0.5 | 0.6 | 0.06 | 0% | yes |
| Dense | 0.6875 | -0.0389 | 6.4 | 7.4 | 0.88 | 92% | yes |
| Hybrid | 0.7264 | +0.0000 | 7.3 | 8.4 | 1.00 | 81% | yes |
| GBM router | 0.7239 | -0.0025 | 12.6 | 14.6 | 1.72 | — |  |
| Skip-dense router | 0.7237 | -0.0027 | 6.7 | 10.2 | 0.91 | — | yes |
| Predicted-gain escalation 10% | 0.7272 | +0.0008 | 53.3 | 400.6 | 7.27 | — |  |
| Router + τ escalation 10% | 0.7247 | -0.0017 | 44.3 | 391.6 | 6.04 | — |  |
| Random escalation 10% | 0.7251 | -0.0013 | 47.0 | 399.5 | 6.40 | — |  |
| Always rerank (MiniLM-L6) | 0.6986 | -0.0278 | 403.0 | 454.9 | 54.96 | 100% |  |

Frontier (cheapest first): BM25 (0.5 s, 0.6827) → Dense (6.4 s, 0.6875) → Skip-dense router (6.7 s, 0.7237) → Hybrid (7.3 s, 0.7264) → Router-confidence escalation 5% (32.2 s, 0.7282) → Predicted-gain escalation 15% (73.3 s, 0.7299)

## fiqa

| Policy | nDCG@10 | − hybrid | Seconds per 1,000 queries (mean) | p95 ms per query | × hybrid cost | GPU share | On frontier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.2528 | -0.1501 | 1.1 | 3.1 | 0.04 | 0% | yes |
| Dense | 0.3747 | -0.0282 | 25.0 | 98.4 | 0.93 | 42% | yes |
| Hybrid | 0.4029 | +0.0000 | 27.1 | 103.9 | 1.00 | 39% | yes |
| GBM router | 0.3909 | -0.0120 | 39.8 | 153.5 | 1.47 | — |  |
| Skip-dense router | 0.3974 | -0.0055 | 32.5 | 130.5 | 1.20 | — |  |
| Predicted-gain escalation 10% | 0.4010 | -0.0019 | 76.5 | 352.2 | 2.83 | — |  |
| Router + τ escalation 10% | 0.3940 | -0.0089 | 67.7 | 351.4 | 2.50 | — |  |
| Random escalation 10% | 0.3987 | -0.0042 | 61.3 | 346.9 | 2.27 | — |  |
| Always rerank (MiniLM-L6) | 0.3759 | -0.0270 | 367.8 | 505.6 | 13.60 | 95% |  |

Frontier (cheapest first): BM25 (1.1 s, 0.2528) → Dense (25.0 s, 0.3747) → Hybrid (27.1 s, 0.4029) → Random escalation 5% (42.9 s, 0.4040) → Router-confidence escalation 10% (74.1 s, 0.4042) → Router-confidence escalation 15% (91.3 s, 0.4048)

## cqadupstack-unix

| Policy | nDCG@10 | − hybrid | Seconds per 1,000 queries (mean) | p95 ms per query | × hybrid cost | GPU share | On frontier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.2937 | -0.0827 | 0.7 | 1.0 | 0.05 | 0% | yes |
| Dense | 0.3597 | -0.0166 | 13.5 | 18.2 | 0.90 | 44% | yes |
| Hybrid | 0.3764 | +0.0000 | 14.9 | 19.8 | 1.00 | 40% | yes |
| GBM router | 0.3658 | -0.0106 | 21.0 | 27.2 | 1.41 | — |  |
| Skip-dense router | 0.3697 | -0.0067 | 15.5 | 22.8 | 1.04 | — |  |
| Predicted-gain escalation 10% | 0.3786 | +0.0022 | 55.8 | 355.5 | 3.74 | — |  |
| Router + τ escalation 10% | 0.3715 | -0.0048 | 56.1 | 353.5 | 3.77 | — |  |
| Random escalation 10% | 0.3764 | +0.0000 | 48.5 | 351.4 | 3.26 | — |  |
| Always rerank (MiniLM-L6) | 0.3683 | -0.0081 | 348.0 | 381.8 | 23.37 | 97% |  |

Frontier (cheapest first): BM25 (0.7 s, 0.2937) → Dense (13.5 s, 0.3597) → Hybrid (14.9 s, 0.3764) → Router-confidence escalation 5% (38.2 s, 0.3781) → Router-confidence escalation 10% (54.4 s, 0.3798)

## quora

| Policy | nDCG@10 | − hybrid | Seconds per 1,000 queries (mean) | p95 ms per query | × hybrid cost | GPU share | On frontier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.9333 | -0.0341 | 0.6 | 0.8 | 0.04 | 0% | yes |
| Dense | 0.9587 | -0.0087 | 14.2 | 20.1 | 0.91 | 43% |  |
| Hybrid | 0.9674 | +0.0000 | 15.7 | 21.5 | 1.00 | 39% | yes |
| GBM router | 0.9643 | -0.0032 | 21.5 | 28.0 | 1.37 | — |  |
| Skip-dense router | 0.9630 | -0.0044 | 9.4 | 21.8 | 0.60 | — | yes |
| Predicted-gain escalation 10% | 0.9630 | -0.0044 | 25.3 | 49.5 | 1.62 | — |  |
| Router + τ escalation 10% | 0.9623 | -0.0052 | 24.4 | 48.9 | 1.56 | — |  |
| Random escalation 10% | 0.9662 | -0.0012 | 18.6 | 43.6 | 1.19 | — |  |
| Always rerank (MiniLM-L6) | 0.9491 | -0.0183 | 44.8 | 58.2 | 2.86 | 79% |  |

Frontier (cheapest first): BM25 (0.6 s, 0.9333) → Skip-dense router (9.4 s, 0.9630) → Hybrid (15.7 s, 0.9674) → Router-confidence escalation 5% (22.9 s, 0.9675)

**Reading the frontier.** Points above hybrid gain at most +0.0035 nDCG@10 and are within noise: Experiment 4 found no selector significantly better than random escalation (`results/experiment4/tables.md`), and the random policy here is one seeded sample. The robust conclusions are the large gaps: hybrid costs about the same as dense alone and is far better; every rerank policy costs 1.1–58× hybrid for no reliable gain.

All budgets (5–50%) and policies: `cost_per_1k.csv`. Figure: `report/figures/cost_quality_frontier.png`.
