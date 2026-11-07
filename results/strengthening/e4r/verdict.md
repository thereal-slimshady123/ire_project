# E4-R verdict (pre-registered in results/preregistration/e4r_climate_fever_20261107.md)

Climate-FEVER, e5-small-v2, 1,391 queries with >= 2 relevant documents (718 with a dense/BM25 winner in half B).

- (a) split-half reference AUC' = 0.507 [0.501, 0.556] -> < 0.65 and CI upper < 0.70: PASS
- (b) split-half kappa (3-way) = -0.004 -> < 0.20: PASS
- (c) nDCG@20 stability kappa = 0.759 -> >= 0.60: PASS

**Verdict: ACCEPTED** (predicted: accepted). The original E4 verdict (inconclusive) is unchanged and reported alongside.
