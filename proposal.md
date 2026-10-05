# Query Routing for Hybrid Retrieval: Confidence-Based Selection Among BM25, Dense, Hybrid, and Cross-Encoder Escalation

## Problem Statement

Train a lightweight query router that assigns each query to one of four buckets: **BM25**, **dense (semantic)**, **hybrid (fixed α)**, or **escalate** (rerank with a cross-encoder). The router predicts among the three cheap routes from query features and retrieval-signal features (BM25/dense score distributions, top-k overlap, rank correlation, query length, rare-term density). A query is **escalated when the router's confidence falls below a threshold τ**, so the escalation rate can be set directly (e.g. ≤10%). Escalation reranks the union of the BM25 and dense top-100 with a MiniLM-style cross-encoder. Evaluate on 2-3 BEIR datasets against fixed-α hybrid, reciprocal rank fusion (RRF), single-retriever, always-rerank, and oracle-router baselines.

**Hypotheses**
- H1: Routing improves ranking-quality metrics (nDCG@10, MRR, MAP) over fixed-α hybrid and RRF, with significance tested per dataset.
- H2: Recall@100 matches or slightly improves on fixed hybrid (reranking the union cannot exceed the union's recall, so large recall gains are not expected).
- H3: With ≤10% of queries escalated, the router retains most of the always-rerank quality gain at a small fraction of its latency.
- H4: Average per-query latency stays close to fixed hybrid and far below always-rerank and LLM-in-the-loop approaches such as DAT.

## Relevant Literature

### Adaptive / query-dependent weighting (closest overlap)

**DAT — Dynamic Alpha Tuning for Hybrid Retrieval in RAG (2025, arXiv:2503.23013)**
Uses an LLM to estimate the optimal per-query α by comparing top-1 results from sparse and dense retrieval. Its authors flag the LLM's latency, memory, and GPU cost as a limitation. Closest prior work on adaptive hybrid retrieval, and the main latency comparison point.

**Query-Adaptive Hybrid Search (2026, MDPI)**
Predicts α from query latent representations, with a lightweight and a full-capacity predictor variant, to avoid LLM online-evaluation latency. Similar in spirit to this project, but it regresses a continuous α and has no escalation mechanism.

### Query routing and selective prediction (added for the routing formulation)

**Adaptive-RAG (Jeong et al., 2024)**
Trains a classifier to route queries by complexity to different retrieval/generation strategies. It is the main precedent for query routing, but it routes between RAG strategies, not among sparse, dense, and hybrid retrieval.

**Selective classification (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017)**
Formalizes abstaining when confidence is below a threshold and trading coverage against risk. This is the framework behind confidence-based escalation and the risk-coverage curves.

**Cross-encoder reranking (Nogueira & Cho, 2019, and MS MARCO MiniLM cross-encoders)**
Cross-encoders jointly encode the query and passage for higher-precision reranking at higher per-pair cost. They serve as the escalation backend.

### Static / fixed-weight hybrid fusion (baselines)

**Sparse Meets Dense: A Hybrid Approach to Enhance Scientific Document Retrieval (AAAI-SDU 2024)**
Convex combination of sparse and dense scores with λ swept as a hyperparameter. Reference for the fixed-α baseline.

**Hybrid Dense-Sparse Retrieval for High-Recall IR (2026)**
Weighted score fusion with large Recall@10 gains over dense-only on MS MARCO. Justifies using simple hybrid fusion as the main baseline rather than chasing ColBERT- or SPLADE-style systems.

**Pinecone hybrid search docs (practical reference)**
BM25-style scores are unbounded while dense scores are bounded, so scores must be normalized before fusion.

### Foundational (background / related work)

- **BEIR (Thakur et al., 2021, arXiv:2104.08663)**: the evaluation suite; 2-3 datasets are chosen from it.
- **SPAR (Chen et al., 2021)**: distills a sparse model into a dense embedding; motivation for "why hybrid."
- **Reciprocal Rank Fusion (Cormack et al., 2009)**: rank-based fusion, the second baseline family.
- **DPR (Karpukhin et al., 2020) and Contriever (Izacard et al., 2021)**: standard pretrained dense retrievers, used as-is without retraining.

## a. Why is it important? What improves if solved?

Hybrid retrieval is the de facto standard in production search and RAG, but it treats every query the same way. Keyword-heavy or rare-entity queries favor sparse retrieval, paraphrastic or semantic queries favor dense retrieval, and some queries are hard for every cheap method. A single fixed fusion rule cannot serve all three cases, and always applying an expensive reranker to everything is costly.

If solved cheaply (no LLM call per query):

- Ranking quality (nDCG@10, MRR, MAP) improves over fixed hybrid and RRF without per-domain retuning.
- Expensive computation is spent only where it helps: the few queries where the cheap routes are uncertain.
- Average latency stays close to fixed hybrid and far below always-rerank or LLM-based adaptive approaches. The escalation rate is a tunable knob, giving deployers an explicit cost/quality tradeoff.
- This matters for RAG systems, enterprise search, and any latency- or cost-sensitive pipeline.

## b. Literature, gap acknowledgement, prior attempts, threads, and feasibility

### Does the literature acknowledge this gap?

Partly.

- DAT (2025) shows that static weighting is suboptimal and that per-query adaptation helps, but pays for it with an LLM call per query.
- Query-Adaptive Hybrid Search (2026) moves to a lightweight trained predictor, but predicts a continuous α from latent representations and has no notion of routing to a stronger backend when the cheap routes are uncertain.
- Adaptive-RAG shows routing by query type works, but not among sparse, dense, and hybrid retrieval.

The field is moving toward cheap, learned adaptation. The specific combination here (discrete routing among BM25, dense, and hybrid, plus confidence-based escalation to a cross-encoder under a controlled budget, evaluated with risk-coverage analysis) is what needs explicit differentiation.

### Prior attempts

| Approach | Method | Gap this project can target |
|---|---|---|
| DAT (2025) | LLM estimates α per query from top-1 sparse/dense results | Expensive; not reproducible without LLM access |
| Query-Adaptive Hybrid Search (2026) | Predicts α from query latent representations | Less interpretable; no escalation; no comparison with simple hand-crafted features |
| Adaptive-RAG (2024) | Classifier routes queries by complexity across RAG strategies | Does not route among sparse/dense/hybrid retrievers |
| Sparse Meets Dense (AAAI-SDU 2024) | Fixed λ swept globally | No per-query adaptivity; baseline only |
| Pinecone docs | Fixed α with score normalization | Practical reference, not adaptivity |
| RRF (Cormack et al., 2009) | Rank-based fusion, no learned weighting | Alternate fusion baseline |

### Design decisions

**Metrics.** The headline claim is on end-to-end retrieval quality: nDCG@10, MRR, MAP, and Recall@100. Router classification quality (macro-F1, per-class confusion matrices) is a diagnostic. Every metric is reported honestly, including any where the router only matches the baselines. Recall@100 is expected to match, not beat, hybrid, because reranking cannot add documents the union missed.

**Labels.** For each query, compute nDCG@10 under BM25, dense, and hybrid. The best route is the argmax only if it beats the runner-up by a margin ε; otherwise the label is hybrid (the safe default), which avoids training on tie noise. Raw per-route nDCG values are kept for regret-weighted training, so misrouting a high-gap query costs more than misrouting a near-tie. All thresholds (ε, τ) are tuned on training folds only.

**Escalation.** Confidence-based (selective classification): the router predicts among the three cheap routes, and if its maximum class probability is below τ, the query is escalated. Sweeping τ gives the escalation rate directly (5%, 10%, 15%) and yields risk-coverage curves. The backend is a MiniLM cross-encoder reranking the union of the BM25 and dense top-100, optionally truncated to the top 30-50 for lower latency.

**Controls.**
- Random escalation at the same rate (does the router pick *which* queries to escalate better than chance?)
- Always-hybrid plus rerank on a random subset of the same size
- Always-rerank (upper bound on cost and gain)
- Oracle router (upper bound on routing headroom)
- Fixed-α hybrid (tuned on the same folds) and RRF

**Latency.** Measured per query and reported as mean and p95, broken down by component (BM25, dense encode and search, feature extraction, router inference, cross-encoder). Compared against BM25-only, hybrid, always-rerank, and the reported cost profile of DAT.

**Features.**
- Query-only: length, max/mean IDF, rare-term and entity density, fraction of out-of-vocabulary terms
- Retrieval-signal: BM25/dense top-k overlap (Jaccard), rank correlation between the two lists, top-1 score, top1-top2 gap, and score entropy per retriever

Models: logistic regression and a GBM (LightGBM/XGBoost) with class weighting.

### Threads to pursue (future work that could become a paper)

1. **Hand-crafted vs. latent-representation routers.** The MDPI paper uses learned query embeddings; this project uses interpretable retrieval-signal features. Comparing the two is a specific research question and the main differentiation angle.
2. **Confidence-based vs. label-based escalation.** Compare a fourth "escalate" class against thresholding router confidence.
3. **Label-generation strategy.** Margin-based argmax vs. regret-weighted labels vs. DAT's top-1 heuristic.
4. **Cross-domain generalization.** Train the router on one BEIR dataset and test on another. Neither DAT nor the MDPI paper reports this, and label distributions differ a lot across datasets.
5. **Feature ablation.** Which features drive routing, and how much comes from query-only vs. retrieval-signal features.
6. **Efficiency framing.** Quality retained per unit of latency vs. always-rerank and DAT-style approaches.

Recommended order for a first pass: (1) + (2) + (5) as the core deliverable, with (4) as a stretch result.

### Does it justify a 3-person team?

Yes, split by lane after a shared infrastructure milestone (~week 2):

- **Person A, retrieval infra:** BM25 and dense pipelines, score normalization, fixed-α and RRF baselines, cross-encoder escalation backend, always-rerank baseline, latency benchmarking.
- **Person B, labels and analysis:** per-route nDCG for every query, margin-based labeling, oracle-router headroom analysis, significance testing.
- **Person C, router modeling:** feature extraction, router training (regret-weighted loss), τ sweep and risk-coverage curves, ablations, writeup coordination.

The lanes are close to independently parallelizable once infrastructure is shared. One person would bottleneck sequentially (infra → labels → model), and five or more would have too little independent work.

### Risks

- **Small query sets.** BEIR test sets are small (SciFact ~300 queries, NFCorpus ~323). Use k-fold cross-validation over queries and paired significance tests (paired bootstrap on per-query nDCG); gains of ~1 point may not be significant.
- **Insufficient headroom.** If the oracle router barely beats hybrid, there is little to gain. This is checked at the week-2 checkpoint, before investing in modeling.
- **Weak baseline tuning.** Fixed α must be tuned as carefully as the router, on the same folds.
- **Latency vs. BM25.** If features use both retrievers' scores, both retrievers run on every query, so average latency cannot fall below hybrid. The claim is therefore latency near hybrid, not near BM25-only.

## Resources (skills + hardware)

**Skills needed:** Python, basic ML (feature engineering, training a small classifier via sklearn/LightGBM), and IR fundamentals (BM25, embeddings, nDCG/MRR/MAP evaluation, reranking). The IR-specific pieces are the main new learning curve.

**Hardware:** the binding constraint is dense corpus encoding. Full MS MARCO (~8.8M passages) is not feasible on free or limited compute. Instead, smaller BEIR datasets are used:

- NFCorpus (~3.6K docs), SciFact (~5K docs), SciDocs (~25K docs), FiQA (~57K docs): all embeddable on a free-tier Colab/Kaggle GPU in well under an hour, with FAISS on CPU.
- TREC-COVID (~171K docs): borderline feasible, slower.
- The cross-encoder runs only on the escalated fraction of queries (top 30-100 candidates each), which is manageable on a free-tier GPU.

**Verdict:** feasible on free/limited compute if small-to-mid BEIR datasets are deliberately chosen. This is stated as a scoping decision, not an oversight.

## Rough Plan

1. **Read:** DAT, Query-Adaptive Hybrid Search, and Adaptive-RAG closely (define the gap and differentiation); skim the rest for background.
2. **Infra:** pick 2-3 small/mid BEIR datasets (e.g. NFCorpus, SciFact, FiQA). Stand up BM25 (Pyserini / rank_bm25), one dense retriever (Contriever / sentence-transformers), and a MiniLM cross-encoder. Confirm end-to-end nDCG@10 evaluation.
3. **Baselines:** BM25-only, dense-only, fixed-α hybrid (swept), RRF, always-rerank. Normalize scores before fusion.
4. **Week-2 checkpoint:** compute per-route nDCG for all queries and the oracle-router headroom over hybrid. If headroom is under ~2 nDCG points, revisit the dataset choice before continuing.
5. **Labels:** margin-based best-route labels (ε tuned on training folds), keeping raw per-route nDCG for regret weighting.
6. **Features:** query-only and retrieval-signal features as listed above.
7. **Router:** train logistic regression and a GBM over the three cheap routes; set escalation by confidence threshold τ; sweep τ for risk-coverage curves and cost/quality tradeoffs.
8. **Evaluate:** routing vs. all baselines and controls (random escalation, oracle, always-rerank) on nDCG@10, MRR, MAP, and Recall@100, plus latency (mean, p95, per component), per dataset and pooled, with paired significance tests. Ablate features.
9. **Stretch:** cross-domain generalization (train on one dataset, test on another) and a latent-representation router comparison.
10. **Write up:** report once the project is done.