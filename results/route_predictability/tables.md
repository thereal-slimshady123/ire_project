# Route predictability (explanatory, post hoc)

Held-out (Experiment 2 folds): a LightGBM classifier on the 18 features + PCA-32 query embedding predicts "route beats hybrid". Precision is among queries whose nDCG@10 changes; break-even = mean loss / (mean gain + mean loss) over all queries.

| Route | Win rate | Mean gain / loss | Break-even precision | AUC | Most confident | Win / lose | Precision | Mean Δ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bm25 | 7.0% | +0.200 / −0.310 | 61% | 0.674 | 1% (34 q) | 21% / 41% | 33% | -0.0265 |
| bm25 | 7.0% | +0.200 / −0.310 | 61% | 0.674 | 2% (67 q) | 15% / 33% | 31% | -0.0225 |
| bm25 | 7.0% | +0.200 / −0.310 | 61% | 0.674 | 5% (168 q) | 19% / 35% | 35% | -0.0316 |
| bm25 | 7.0% | +0.200 / −0.310 | 61% | 0.674 | 10% (335 q) | 15% / 37% | 29% | -0.0537 |
| dense | 12.3% | +0.167 / −0.222 | 57% | 0.682 | 1% (34 q) | 18% / 38% | 32% | -0.0418 |
| dense | 12.3% | +0.167 / −0.222 | 57% | 0.682 | 2% (67 q) | 18% / 31% | 36% | -0.0258 |
| dense | 12.3% | +0.167 / −0.222 | 57% | 0.682 | 5% (168 q) | 18% / 29% | 39% | -0.0108 |
| dense | 12.3% | +0.167 / −0.222 | 57% | 0.682 | 10% (335 q) | 22% / 26% | 46% | -0.0060 |
