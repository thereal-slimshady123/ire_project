# Experiment 4 results: selective cross-encoder escalation

Pre-registered in improvements.md §4.2. Held-out nDCG@10; base = always-hybrid (training-fold α*); escalation = MiniLM cross-encoder rerank of the hybrid top-50.

**Primary (pooled, B = 10%): predicted-gain − random = +0.0008 [-0.0012, +0.0028] → NEGATIVE.** Router-confidence − random = +0.0013 [-0.0009, +0.0035].

| Dataset | n | Always-hybrid | Always-rerank | Pred-gain @10% | Router-conf @10% | Random @10% | Oracle @10% | Pred-gain − random @10% | Retained @10% (pred-gain) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 0.3589 | 0.3546 | 0.3599 | 0.3558 | 0.3585 | 0.3751 | +0.0014 [-0.0030, +0.0064] | n/a |
| scifact | 300 | 0.7264 | 0.6986 | 0.7272 | 0.7229 | 0.7236 | 0.7680 | +0.0036 [-0.0035, +0.0111] | n/a |
| fiqa | 648 | 0.4029 | 0.3759 | 0.4010 | 0.4042 | 0.4002 | 0.4399 | +0.0008 [-0.0033, +0.0051] | n/a |
| cqadupstack-unix | 1072 | 0.3764 | 0.3683 | 0.3786 | 0.3798 | 0.3755 | 0.4176 | +0.0030 [-0.0010, +0.0072] | n/a |
| quora | 1000 | 0.9674 | 0.9491 | 0.9630 | 0.9639 | 0.9656 | 0.9781 | -0.0026 [-0.0059, +0.0006] | n/a |
| **pooled** | 3343 | 0.5880 | 0.5718 | 0.5872 | 0.5877 | 0.5864 | 0.6169 | +0.0008 [-0.0012, +0.0028] | n/a |

## Pooled: quality retained = (policy − hybrid) / (always-rerank − hybrid)

| Budget | Predicted gain | Router confidence | Random | Oracle |
| :---: | :---: | :---: | :---: | :---: |
| 5% | 0.5877 (n/a) | 0.5884 (n/a) | 0.5872 (n/a) | 0.6078 (n/a) |
| 10% | 0.5872 (n/a) | 0.5877 (n/a) | 0.5864 (n/a) | 0.6169 (n/a) |
| 15% | 0.5875 (n/a) | 0.5866 (n/a) | 0.5856 (n/a) | 0.6206 (n/a) |
| 20% | 0.5865 (n/a) | 0.5849 (n/a) | 0.5848 (n/a) | 0.6216 (n/a) |
| 30% | 0.5861 (n/a) | 0.5841 (n/a) | 0.5832 (n/a) | 0.6221 (n/a) |
| 50% | 0.5825 (n/a) | 0.5803 (n/a) | 0.5799 (n/a) | 0.6221 (n/a) |
| 100% | 0.5718 (n/a) | 0.5718 (n/a) | 0.5718 (n/a) | 0.5718 (n/a) |
