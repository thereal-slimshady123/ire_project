# Learning curve: arguana

Fixed test set n = 350 (112 with a dense/BM25 winner). Training subsets from the remaining 1051 queries; 5 repeats per size (1 at the largest).

| Training queries | Router − hybrid [95% CI] | Learned winner AUC′ [95% CI] |
| :---: | :---: | :---: |
| 250 | -0.0169 [-0.0262, -0.0078] | 0.552 [0.530, 0.633] |
| 500 | -0.0116 [-0.0213, -0.0021] | 0.582 [0.531, 0.686] |
| 1000 | -0.0081 [-0.0185, +0.0024] | 0.628 [0.535, 0.748] |
| 1051 | -0.0037 [-0.0168, +0.0094] | 0.639 [0.517, 0.768] |

Plateau contrasts (1051 vs 1000 training queries): router − hybrid change +0.0044 [-0.0052, +0.0140]; AUC′ change +0.011 [-0.036, +0.056].
