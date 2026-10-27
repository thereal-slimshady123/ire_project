# Confidence-Based Query Routing for Hybrid Retrieval

## Abstract

We study whether query features can improve retrieval quality by selecting BM25, dense retrieval, or a fixed-weight hybrid, and whether confidence based escalation can target cross-encoder reranking under a budget. Experiments use five BEIR/MTEB query sets and held-out, five-fold evaluation over 3,343 queries. The main held-out test is negative: a LightGBM route classifier scores 0.0061 nDCG@10 below always-hybrid (95% paired bootstrap CI [−0.0091, −0.0032]). The model does not beat random routing at the same route proportions. A separate skip-dense experiment meets its pre-registered pooled non-inferiority, savings, and random-control criteria, while a simple BM25 top-score ratio matches the learned classifier. The tested MiniLM-L6 reranker lowers mean quality, and selective policies do not significantly beat random escalation. These findings support a narrower conclusion: feature based routing among the tested cheap systems did not generalize, BM25 score separation can identify some queries where dense retrieval may be skipped, and useful escalation depends on having a reranker that improves the base ranking.

## 1. Introduction and Research Questions

Search systems commonly combine lexical matching with semantic retrieval. BM25 is effective for exact terms and identifiers; dense retrievers can find semantically related passages; hybrid retrieval combines both candidate lists. Cross-encoders can score query-document pairs jointly, but add inference cost. This project began with the hypothesis that inexpensive query and retrieval features could choose among BM25, dense, and hybrid, then escalate uncertain queries to a cross-encoder.

The initial proposal expected routing to outperform a fixed hybrid and selective reranking to retain most of the always-rerank gain at a low budget. The experiments below test those claims against held-out queries and matched random controls. We report negative results directly. In particular, a per-query oracle is an upper bound and does not demonstrate that a learnable signal can identify the winning route.

## 2. Method and Evaluation Protocol

### 2.1 Systems and features

The first-stage systems are BM25, a dense bi-encoder, and a convex hybrid over min-max-normalized BM25 and dense scores. The hybrid dense weight α is selected from a grid on training queries only. The BM25 normalization uses a zero floor when a truncated BM25 list implies that missing documents have score zero. This prevents a single BM25 hit from being normalized to the same score as an unretrieved document.

The original classifier uses 18 query and retrieval-signal features: query length, IDF and rare-term statistics, entity and out-of-vocabulary density, top-k list overlap, rank correlation, top scores, score gaps and entropy, top-score ratio, and query-term coverage. GBM labels use a margin rule that defaults near-ties to hybrid; training weights scale with the spread of route scores. The separate Experiment 3 skip model uses only query and BM25 signals available before dense retrieval.

### 2.2 Datasets and metrics

The evaluation covers NFCorpus (323 test queries), SciFact (300), FiQA (648), CQADupStack Unix (1,072), and a deterministic Quora subset (1,000). The Quora subset retains the sampled queries' relevant documents and 50,000 random distractors. Experiments 2–4 use `intfloat/e5-small-v2` for dense retrieval and top-100 candidate lists; Experiment 1 used `msmarco-distilbert-base-v4`, so its absolute values are not directly comparable with later experiments.

The primary measure is nDCG@10. MRR, MAP, and Recall@100 are additional retrieval metrics in the project evaluation code. Unless otherwise noted, held-out results use five-fold cross-validation within each dataset with seed 42 and a paired bootstrap with 10,000 resamples. Parameters and thresholds are selected from training folds. Pooled statistics combine per-query held-out predictions across the five datasets.

### 2.3 Registered comparisons

Experiment 2 compares GBM routing with always-hybrid and a random policy using the router's own route proportions. Experiment 3 asks whether BM25 can replace hybrid for selected queries while preserving quality and reducing dense calls. Experiment 4 compares predicted-gain and low-confidence escalation with random escalation at the same budget. Oracle results are reported only as upper bounds.

## 3. Results

### 3.0 Latency profile

The component timers report the following observed ranges across the profiled collections. FLOP counts are estimates, not hardware-independent measurements.

| Component | Mean (ms) | p50 (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: |
| BM25 search | 0.81–1.59 | 0.70–1.52 | 1.50–2.42 |
| Dense query encoding | 3.86–3.99 | 3.84–4.01 | 4.14–4.20 |
| FAISS search | 1.06–11.84 | 0.95–11.80 | 1.84–14.72 |
| Hybrid fusion | 1.17–1.51 | 1.09–1.54 | 1.71–2.59 |
| Feature extraction | 1.83–2.75 | 1.69–2.54 | 2.83–4.76 |
| Router inference | 0.01–0.02 | 0.01–0.02 | 0.02–0.03 |
| Cross-encoder reranking | 18.37–18.49 | 18.34–18.45 | 18.59–18.97 |

These measurements show that feature extraction adds measurable overhead, while router inference is negligible. End-to-end latency depends on corpus size and the selected policy; the skip-dense measurements and cross-encoder escalation results are reported with their corresponding experiments below.

### 3.1 Initial hybrid analysis and held-out route selection

Across NFCorpus, SciFact, and FiQA, the tuned hybrid was the best fixed system in mean nDCG@10. The original H1 “degraded query” prevalence compares hybrid with the per-query maximum of BM25 and dense; this selected maximum is biased upward by construction. The project issue log therefore withdraws the original H1 verdict and treats per-query oracle headroom as a bound. A separate, predefined query-type analysis on Unix technical/plain queries and Quora conversational queries found hybrid ahead of both single retrievers in each group; each group's best α was 0.6.

The decisive held-out comparison pooled 3,343 queries. GBM routing achieved 0.5819 nDCG@10 versus 0.5880 for always-hybrid, a difference of −0.0061 (95% CI [−0.0091, −0.0032]). Logistic routing scored 0.5632. GBM routing differed from random routing at the same route proportions by +0.0017 (95% CI [−0.0010, +0.0044]), which is inconclusive. The oracle's +0.0344 difference is an upper bound, not evidence that the available features can realize that gain.

### 3.2 Skipping dense retrieval

At the operating point chosen on training folds, the skip router omitted dense retrieval for 32.9% of held-out queries. Its pooled score was 0.5828 versus 0.5880 for always-hybrid, a difference of −0.0053 (95% CI [−0.0073, −0.0032]). It passed the registered criteria: the confidence interval lower bound exceeded −0.01, savings exceeded 20%, and the router beat expected random skipping by +0.0104 (95% CI [+0.0080, +0.0129]).

The gain is domain dependent. The held-out skip rate ranged from 4.6% on FiQA to 62.7% on Quora. The BM25 top-1/top-2 ratio heuristic was at least as effective as the learned classifier in the reported matched-rate curves. Transfer results did not show uniform quality non-inferiority. The result is evidence for a useful BM25 confidence signal in this evaluation, not a general dense-call reduction guarantee.

### 3.3 Cross-encoder escalation

The preregistered MiniLM-L6 reranker lowered pooled nDCG@10 from 0.5880 for always-hybrid to 0.5718 for always-rerank. At a 10% budget, predicted-gain escalation scored 0.5872 and random escalation 0.5864. Their difference was +0.0008 (95% CI [−0.0012, +0.0028]); router-confidence escalation also did not significantly beat random. Although the oracle-selected 10% reached 0.6169, the tested features did not identify those beneficial cases. Since always-rerank is worse than hybrid on average, “quality retained” relative to that baseline has no meaningful denominator and is reported as not applicable.

The result does not establish that stronger rerankers will help. Alternative checkpoints are configured for an exploratory matched-pool benchmark; their effect must be measured before making that claim. The preregistered MiniLM result remains the primary Experiment 4 result.

## 4. Discussion and Limitations

The first-stage routing hypothesis was not supported: the tested 18 surface features did not choose routes well enough to beat tuned hybrid on unseen queries. This is consistent with weak and dataset-dependent correlations and with the small, discontinuous per-query relevance signals in these benchmarks. Routing among BM25, dense, and hybrid should not be presented as a demonstrated quality improvement here.

The skip-dense result has a narrower interpretation. A sharp BM25 score head carries some information about whether dense retrieval is necessary, but the operating point and savings vary substantially across datasets. The learned model did not outperform the simple top-score ratio rule, and transfer was not consistently non-inferior. A deployment would need domain-specific validation and threshold calibration.

Selective escalation relies on the expensive backend having positive gain over its base system. MiniLM-L6 did not meet that prerequisite on these candidate pools. Stronger rerankers are a plausible next experiment, but model capacity alone does not guarantee improvement; they require matched candidate pools, held-out evaluation, and paired uncertainty estimates. The code includes an exploratory comparison path, separate from the preregistered result.

Other limits include small dataset query counts, heterogeneous benchmark difficulty, the reduced Quora corpus, and model differences between Experiments 1 and 2–4. Pooled results improve precision but can hide dataset-level variation. Oracle headroom and outcome-selected subsets are descriptive upper-bound or existence analyses and cannot be used as evidence of predictive generalization.

## 5. Conclusion

For the tested datasets and features, always-hybrid remains the strongest dependable cheap ranking policy. A BM25 score-ratio rule can support selective dense skipping on the pooled registered evaluation, with important domain variation. The tested cross-encoder and escalation policies did not improve quality. Future work should first validate a genuinely stronger reranker, then evaluate selective escalation with a fresh held-out protocol; richer query representations and larger natural training sets may also test whether route advantages can be predicted reliably.

## References

The project proposal and experiment design discuss BEIR, Adaptive-RAG, selective prediction, cross-encoder reranking, DAT, and query-adaptive hybrid retrieval. Bibliographic details should be completed from the cited source papers in the final submission.
