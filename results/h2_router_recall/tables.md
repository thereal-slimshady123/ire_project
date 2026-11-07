# H2: held-out Recall@100 of the routing system vs always-hybrid

Saved held-out route decisions (no refitting); hybrid at each training fold's α*. Escalated queries keep hybrid's candidate set. Decision rule fixed before computing: supported iff CI within ±0.01 or entirely > 0; not supported iff CI < 0 and below -0.01; else inconclusive.

## Pooled (n = 3343, always-hybrid Recall@100 = 0.7745)

| System | Recall@100 | − hybrid [95% CI] | Verdict |
| :--- | :---: | :---: | :---: |
| GBM router (Exp 2, pre-registered) | 0.7709 | -0.0036 [-0.0074, -0.0000] | supported |
| Logistic router (Exp 2, pre-registered) | 0.7522 | -0.0224 [-0.0279, -0.0169] | not supported |
| Embedding GBM router (Exp 2c, exploratory) | 0.7717 | -0.0028 [-0.0059, +0.0002] | supported |
| Cost-aware GBM router (Exp 2d, exploratory) | 0.7742 | -0.0003 [-0.0018, +0.0010] | supported |
| GBM router + τ escalation, target 5% (exploratory) | 0.7710 | -0.0035 [-0.0072, -0.0001] | supported |
| GBM router + τ escalation, target 10% (exploratory) | 0.7715 | -0.0030 [-0.0064, +0.0003] | supported |
| GBM router + τ escalation, target 15% (exploratory) | 0.7718 | -0.0028 [-0.0061, +0.0004] | supported |

## Per dataset: system − hybrid [95% CI]

| Dataset | n | Hybrid | GBM (Exp 2) | Embedding GBM (Exp 2c) | Cost-aware GBM (Exp 2d) | GBM + τ, 10% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 0.3090 | -0.0003 [-0.0042, +0.0045] | +0.0008 [-0.0014, +0.0036] | -0.0008 [-0.0025, +0.0002] | -0.0014 [-0.0041, +0.0013] |
| scifact | 300 | 0.9583 | -0.0033 [-0.0167, +0.0067] | -0.0051 [-0.0173, +0.0060] | +0.0000 [+0.0000, +0.0000] | -0.0033 [-0.0167, +0.0067] |
| fiqa | 648 | 0.6995 | -0.0196 [-0.0302, -0.0099] | -0.0076 [-0.0171, +0.0012] | +0.0000 [+0.0000, +0.0000] | -0.0152 [-0.0248, -0.0066] |
| cqadupstack-unix | 1072 | 0.6991 | +0.0020 [-0.0069, +0.0108] | -0.0029 [-0.0098, +0.0039] | -0.0008 [-0.0052, +0.0032] | +0.0014 [-0.0070, +0.0095] |
| quora | 1000 | 0.9992 | -0.0005 [-0.0011, -0.0001] | -0.0001 [-0.0003, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0002 [-0.0006, +0.0000] |
| pooled | 3343 | 0.7745 | -0.0036 [-0.0074, -0.0000] | -0.0028 [-0.0059, +0.0002] | -0.0003 [-0.0018, +0.0010] | -0.0030 [-0.0064, +0.0003] |
