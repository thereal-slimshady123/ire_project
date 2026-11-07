# Per-document learned fusion: frozen model applied to new datasets

One LambdaRank model (doc+query features) trained on 16 source datasets; applied unchanged. Source α = 0.6.

**Verdict (vs hybrid at source α): ACCEPTED**

| Dataset | n | Learned fusion | Hybrid (source α) | Hybrid (α* tuned on target) | Learned − hybrid (source α) | Learned − hybrid (tuned α*) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| cqadupstack-stats | 652 | 0.3459 | 0.3447 | 0.3447 | +0.0011 [-0.0036, +0.0061] | +0.0011 [-0.0036, +0.0061] |
| cqadupstack-wordpress | 541 | 0.3493 | 0.3438 | 0.3411 | +0.0055 [-0.0020, +0.0132] | +0.0082 [+0.0007, +0.0161] |
| cqadupstack-tex | 2906 | 0.2894 | 0.2878 | 0.2863 | +0.0016 [-0.0013, +0.0043] | +0.0031 [-0.0000, +0.0061] |
| scidocs | 1000 | 0.1882 | 0.1869 | 0.1869 | +0.0013 [-0.0013, +0.0038] | +0.0013 [-0.0013, +0.0038] |
| nq | 3452 | 0.9244 | 0.9202 | 0.9284 | +0.0043 [+0.0018, +0.0067] | -0.0040 [-0.0070, -0.0010] |
| hotpotqa | 7405 | 0.9011 | 0.8982 | 0.8990 | +0.0029 [+0.0019, +0.0040] | +0.0021 [+0.0010, +0.0031] |
| **pooled** | 15956 | 0.7086 | 0.7058 | 0.7076 | +0.0029 [+0.0019, +0.0038] | +0.0011 [+0.0000, +0.0021] |
