# Per-document learned fusion vs fixed-α hybrid

Dense model intfloat/e5-small-v2. Held-out nDCG@10, 5-fold CV by query within each dataset; hybrid α* tuned on the training folds.

| Dataset | n | Hybrid (α*) | Learned fusion | Learned − hybrid [95% CI] | Transfer: learned − hybrid (source α) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 0.3589 | 0.3465 | -0.0124 [-0.0208, -0.0044] | — |
| scifact | 300 | 0.7264 | 0.7439 | +0.0174 [-0.0008, +0.0364] | — |
| fiqa | 648 | 0.4029 | 0.3896 | -0.0133 [-0.0238, -0.0031] | — |
| cqadupstack-unix | 1072 | 0.3764 | 0.3750 | -0.0014 [-0.0090, +0.0064] | — |
| quora | 1000 | 0.9674 | 0.9666 | -0.0008 [-0.0044, +0.0030] | — |
| wands | 479 | 0.7650 | 0.7640 | -0.0010 [-0.0069, +0.0045] | — |
| cosqa | 500 | 0.2651 | 0.2837 | +0.0186 [-0.0055, +0.0428] | — |
| cqadupstack-android | 699 | 0.4688 | 0.4511 | -0.0177 [-0.0286, -0.0072] | — |
| cqadupstack-webmasters | 506 | 0.3827 | 0.3635 | -0.0192 [-0.0326, -0.0062] | — |
| arguana | 1401 | 0.6373 | 0.6321 | -0.0053 [-0.0142, +0.0033] | — |
| cqadupstack-mathematica | 804 | 0.2748 | 0.2534 | -0.0214 [-0.0326, -0.0104] | — |
| cqadupstack-english | 1570 | 0.4567 | 0.4522 | -0.0045 [-0.0097, +0.0007] | — |
| cqadupstack-gaming | 1595 | 0.5770 | 0.5748 | -0.0022 [-0.0073, +0.0030] | — |
| cqadupstack-physics | 1039 | 0.4218 | 0.4126 | -0.0092 [-0.0168, -0.0016] | — |
| cqadupstack-programmers | 876 | 0.4077 | 0.3928 | -0.0149 [-0.0230, -0.0070] | — |
| cqadupstack-gis | 885 | 0.3725 | 0.3617 | -0.0108 [-0.0199, -0.0017] | — |
| **pooled** | 13697 | 0.5010 | 0.4944 | -0.0066 [-0.0089, -0.0043] | — |
