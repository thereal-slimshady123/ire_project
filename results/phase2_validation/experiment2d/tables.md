# Experiment 2d results: cost-aware router vs tuned always-hybrid (exploratory)

Design fixed before running in `results/preregistration/experiment2d_cost_aware_router_20261031.md` (not blind: a diagnostic on the same folds motivated it). Held-out queries, 5-fold CV within each dataset (Experiment 2 folds); α*, ε, PCA and routers fitted on training folds only.

**Primary (pooled, n = 3969): cost-aware GBM − always-hybrid nDCG@10 = +0.0001 [-0.0004, +0.0005], p = 7.2e-01 → superiority NOT SUPPORTED; non-inferiority (δ = 0.005) MET.**

## ndcg_cut_10: means

| Dataset | n | Always-hybrid | RRF | Cost-aware GBM | Blended target | + embedding | No fallback | Exp 2 refit (balanced) | Random (same mix) | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | 804 | 0.2748 | 0.2648 | 0.2752 | 0.2748 | 0.2735 | 0.2729 | 0.2671 | 0.2745 | 0.3156 |
| cqadupstack-english | 1570 | 0.4567 | 0.4336 | 0.4567 | 0.4557 | 0.4567 | 0.4556 | 0.4460 | 0.4561 | 0.5003 |
| cqadupstack-gaming | 1595 | 0.5770 | 0.5600 | 0.5770 | 0.5770 | 0.5770 | 0.5733 | 0.5670 | 0.5770 | 0.6248 |
| **pooled** | 3969 | 0.4682 | 0.4502 | 0.4683 | 0.4678 | 0.4679 | 0.4659 | 0.4584 | 0.4679 | 0.5129 |

## ndcg_cut_10: router − always-hybrid [95% bootstrap CI]

| Dataset | Cost-aware GBM | Blended target | + embedding | No fallback | Exp 2 refit (balanced) | Cost-aware − random | Cost-aware − RRF (added) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | +0.0004 [-0.0006, +0.0015] | +0.0000 [+0.0000, +0.0000] | -0.0014 [-0.0041, +0.0000] | -0.0019 [-0.0057, +0.0011] | -0.0077 [-0.0134, -0.0027] | +0.0007 [-0.0003, +0.0019] | +0.0104 [+0.0008, +0.0202] |
| cqadupstack-english | +0.0000 [-0.0010, +0.0009] | -0.0010 [-0.0031, +0.0007] | +0.0000 [+0.0000, +0.0000] | -0.0011 [-0.0035, +0.0012] | -0.0107 [-0.0164, -0.0054] | +0.0006 [-0.0003, +0.0015] | +0.0231 [+0.0163, +0.0302] |
| cqadupstack-gaming | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0006, +0.0007] | -0.0037 [-0.0067, -0.0010] | -0.0100 [-0.0153, -0.0048] | +0.0000 [+0.0000, +0.0000] | +0.0170 [+0.0103, +0.0238] |
| **pooled** | +0.0001 [-0.0004, +0.0005] | -0.0004 [-0.0012, +0.0003] | -0.0003 [-0.0009, +0.0002] | -0.0023 [-0.0039, -0.0007] | -0.0098 [-0.0130, -0.0066] | +0.0004 [-0.0000, +0.0008] | +0.0181 [+0.0138, +0.0224] |

## mrr_10: means

| Dataset | n | Always-hybrid | RRF | Cost-aware GBM | Blended target | + embedding | No fallback | Exp 2 refit (balanced) | Random (same mix) | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | 804 | 0.2686 | 0.2596 | 0.2692 | 0.2686 | 0.2670 | 0.2680 | 0.2618 | 0.2684 | 0.3165 |
| cqadupstack-english | 1570 | 0.4679 | 0.4423 | 0.4672 | 0.4660 | 0.4679 | 0.4662 | 0.4553 | 0.4672 | 0.5205 |
| cqadupstack-gaming | 1595 | 0.5606 | 0.5422 | 0.5606 | 0.5606 | 0.5608 | 0.5573 | 0.5523 | 0.5606 | 0.6137 |
| **pooled** | 3969 | 0.4648 | 0.4454 | 0.4646 | 0.4640 | 0.4645 | 0.4627 | 0.4551 | 0.4645 | 0.5166 |

## mrr_10: router − always-hybrid [95% bootstrap CI]

| Dataset | Cost-aware GBM | Blended target | + embedding | No fallback | Exp 2 refit (balanced) | Cost-aware − random | Cost-aware − RRF (added) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | +0.0006 [-0.0007, +0.0021] | +0.0000 [+0.0000, +0.0000] | -0.0016 [-0.0047, +0.0000] | -0.0006 [-0.0042, +0.0023] | -0.0068 [-0.0132, -0.0010] | +0.0009 [-0.0004, +0.0023] | +0.0097 [-0.0020, +0.0214] |
| cqadupstack-english | -0.0007 [-0.0024, +0.0007] | -0.0019 [-0.0044, +0.0003] | +0.0000 [+0.0000, +0.0000] | -0.0016 [-0.0050, +0.0016] | -0.0125 [-0.0192, -0.0062] | +0.0000 [-0.0016, +0.0014] | +0.0249 [+0.0167, +0.0335] |
| cqadupstack-gaming | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0002, +0.0009] | -0.0032 [-0.0068, +0.0002] | -0.0083 [-0.0142, -0.0025] | +0.0000 [+0.0000, +0.0000] | +0.0184 [+0.0105, +0.0263] |
| **pooled** | -0.0001 [-0.0009, +0.0005] | -0.0007 [-0.0018, +0.0001] | -0.0002 [-0.0009, +0.0003] | -0.0021 [-0.0041, -0.0001] | -0.0097 [-0.0133, -0.0060] | +0.0002 [-0.0005, +0.0008] | +0.0192 [+0.0140, +0.0245] |

## map: means

| Dataset | n | Always-hybrid | RRF | Cost-aware GBM | Blended target | + embedding | No fallback | Exp 2 refit (balanced) | Random (same mix) | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | 804 | 0.2386 | 0.2280 | 0.2390 | 0.2386 | 0.2373 | 0.2370 | 0.2296 | 0.2384 | 0.2747 |
| cqadupstack-english | 1570 | 0.4164 | 0.3943 | 0.4162 | 0.4150 | 0.4164 | 0.4153 | 0.4052 | 0.4159 | 0.4605 |
| cqadupstack-gaming | 1595 | 0.5353 | 0.5171 | 0.5353 | 0.5353 | 0.5355 | 0.5313 | 0.5262 | 0.5353 | 0.5837 |
| **pooled** | 3969 | 0.4282 | 0.4100 | 0.4282 | 0.4276 | 0.4280 | 0.4258 | 0.4182 | 0.4279 | 0.4724 |

## map: router − always-hybrid [95% bootstrap CI]

| Dataset | Cost-aware GBM | Blended target | + embedding | No fallback | Exp 2 refit (balanced) | Cost-aware − random | Cost-aware − RRF (added) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-mathematica | +0.0004 [-0.0008, +0.0017] | +0.0000 [+0.0000, +0.0000] | -0.0013 [-0.0038, +0.0000] | -0.0016 [-0.0048, +0.0009] | -0.0090 [-0.0149, -0.0039] | +0.0006 [-0.0006, +0.0019] | +0.0109 [+0.0018, +0.0203] |
| cqadupstack-english | -0.0002 [-0.0014, +0.0008] | -0.0014 [-0.0037, +0.0004] | +0.0000 [+0.0000, +0.0000] | -0.0011 [-0.0038, +0.0015] | -0.0112 [-0.0173, -0.0056] | +0.0003 [-0.0008, +0.0014] | +0.0218 [+0.0147, +0.0292] |
| cqadupstack-gaming | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0003, +0.0009] | -0.0040 [-0.0073, -0.0008] | -0.0091 [-0.0144, -0.0038] | +0.0000 [+0.0000, +0.0000] | +0.0182 [+0.0111, +0.0255] |
| **pooled** | -0.0000 [-0.0005, +0.0005] | -0.0006 [-0.0015, +0.0002] | -0.0002 [-0.0008, +0.0003] | -0.0024 [-0.0041, -0.0006] | -0.0099 [-0.0133, -0.0066] | +0.0002 [-0.0002, +0.0007] | +0.0182 [+0.0137, +0.0227] |

## Route shares (held-out) and per-fold choices

| Dataset | System | BM25 | Dense | Hybrid | ε per fold (`H` = always-hybrid fallback) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| cqadupstack-mathematica | Cost-aware GBM | 0.7% | 0.4% | 98.9% | H, 0, H, H, H |
| cqadupstack-mathematica | Blended target | 0.0% | 0.0% | 100.0% | H, H, H, H, H |
| cqadupstack-mathematica | + embedding | 0.9% | 0.0% | 99.1% | 0.05, 0.1, H, H, 0.1 |
| cqadupstack-mathematica | No fallback | 4.7% | 1.4% | 93.9% | 0.1, 0, 0.1, 0.1, 0.05 |
| cqadupstack-mathematica | Exp 2 refit (balanced) | 13.3% | 6.5% | 80.2% | 0.05, 0.1, 0.1, 0.1, 0.1 |
| cqadupstack-english | Cost-aware GBM | 0.5% | 0.8% | 98.7% | 0.1, H, H, H, H |
| cqadupstack-english | Blended target | 0.4% | 2.9% | 96.7% | 0, H, H, H, H |
| cqadupstack-english | + embedding | 0.0% | 0.0% | 100.0% | H, H, H, H, H |
| cqadupstack-english | No fallback | 2.9% | 3.3% | 93.8% | 0.1, 0.1, 0.1, 0.1, 0.02 |
| cqadupstack-english | Exp 2 refit (balanced) | 8.2% | 15.7% | 76.1% | 0.1, 0.1, 0.05, 0.1, 0.1 |
| cqadupstack-gaming | Cost-aware GBM | 0.0% | 0.0% | 100.0% | H, H, H, H, H |
| cqadupstack-gaming | Blended target | 0.0% | 0.0% | 100.0% | H, H, H, H, H |
| cqadupstack-gaming | + embedding | 0.2% | 0.2% | 99.6% | H, H, 0.1, H, H |
| cqadupstack-gaming | No fallback | 3.1% | 5.0% | 91.9% | 0.05, 0.1, 0.1, 0.1, 0.02 |
| cqadupstack-gaming | Exp 2 refit (balanced) | 10.9% | 16.4% | 72.7% | 0.05, 0.1, 0.1, 0.05, 0.1 |
