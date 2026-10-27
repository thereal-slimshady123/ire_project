# Oracle headroom over tuned hybrid (upper bounds)

nDCG@10; dense = e5-small-v2; each query uses its held-out fold's training α*. The noise oracle picks the best of hybrid at α* − 0.1, α*, α* + 0.1, so it measures what a per-query max gains from rank jitter alone. Noise-corrected headroom = oracle − noise oracle. None of these is evidence that a router can realize the gain.

| Dataset | n | Hybrid | Oracle | Oracle − hybrid | Oracle{BM25,dense} − hybrid | Noise oracle − hybrid | Noise-corrected headroom | ≥ 0.02? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nfcorpus | 323 | 0.3589 | 0.3877 | +0.0288 [+0.0228, +0.0352] | +0.0171 [+0.0095, +0.0248] | +0.0114 [+0.0085, +0.0145] | +0.0174 [+0.0124, +0.0226] | no |
| scifact | 300 | 0.7264 | 0.7911 | +0.0647 [+0.0476, +0.0835] | +0.0499 [+0.0306, +0.0704] | +0.0215 [+0.0133, +0.0306] | +0.0432 [+0.0287, +0.0588] | yes |
| fiqa | 648 | 0.4029 | 0.4465 | +0.0436 [+0.0364, +0.0512] | +0.0225 [+0.0131, +0.0321] | +0.0236 [+0.0189, +0.0284] | +0.0200 [+0.0136, +0.0266] | yes |
| cqadupstack-unix | 1072 | 0.3764 | 0.4182 | +0.0418 [+0.0354, +0.0486] | +0.0287 [+0.0209, +0.0366] | +0.0188 [+0.0148, +0.0231] | +0.0230 [+0.0176, +0.0286] | yes |
| quora | 1000 | 0.9674 | 0.9805 | +0.0131 [+0.0092, +0.0174] | +0.0105 [+0.0063, +0.0150] | +0.0040 [+0.0023, +0.0061] | +0.0091 [+0.0060, +0.0125] | no |
| **pooled** | 3343 | 0.5880 | 0.6224 | +0.0344 [+0.0311, +0.0378] | +0.0228 [+0.0190, +0.0268] | +0.0148 [+0.0130, +0.0169] | +0.0195 [+0.0167, +0.0224] | no |
