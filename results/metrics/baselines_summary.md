
# Retrieval Baselines Summary (5-fold CV Hybrid & RRF)

| Dataset | Queries | BM25 nDCG@10 | Dense nDCG@10 | Hybrid (CV) nDCG@10 | RRF nDCG@10 | Best Hybrid alpha |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `nfcorpus` | 323 | 0.3223 | 0.3255 | **0.3589** | 0.3509 | 0.5 |
| `scifact` | 300 | 0.6827 | 0.6875 | **0.7264** | 0.7318 | 0.5 |
| `fiqa` | 648 | 0.2528 | 0.3747 | **0.4029** | 0.3741 | 0.6 |
