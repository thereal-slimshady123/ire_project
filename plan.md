# Implementation Plan: Confidence-Based Query Router for Hybrid Retrieval
## Three-Person Independent Execution Blueprint

This document specifies the complete engineering architecture, experiment protocol, and **strictly decoupled 3-person division of labor** for the Information Retrieval (IRE) project. 

The division is architected to guarantee **maximum developer independence**: no contributor is blocked waiting on another's code or experimental run. Decoupling is enforced through **frozen interface contracts**, **standardized data artifacts (Parquet/JSON)**, **self-contained test suites**, and **dedicated file ownership**.

---

## 1. Principles of Independence & Decoupling Strategy

In multi-stage ML/IR pipelines, team members frequently suffer from serial blocking: *Modeling waits for Features, Features wait for Retrieval, and Evaluation waits for Modeling.* 

To completely eliminate these bottlenecks, the project enforces four architectural safeguards:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SHARED REPOSITORY CONTRACTS                                  │
│               (data/raw/, src/router_retrieval/data/schema.py, configs/*.yaml)                 │
└───────────────────┬────────────────────────────┬────────────────────────────┬──────────────────┘
                    │                            │                            │
                    ▼                            ▼                            ▼
┌────────────────────────────────┐ ┌────────────────────────────────┐ ┌────────────────────────────────┐
│            TRACK 1             │ │            TRACK 2             │ │            TRACK 3             │
│      [Person 1 Ownership]      │ │      [Person 2 Ownership]      │ │      [Person 3 Ownership]      │
│  Retrieval Infra, Baselines,   │ │    Feature Engineering,        │ │  Strong Escalation Backend,    │
│  Latency Profiling &           │ │    Synthetic Data Scaling &    │ │  Statistical Evaluation &      │
│  Skip-Dense Efficiency (Exp 3) │ │    Advantage Routing (Exp 2)   │ │  End-to-End Pipeline (Exp 4)   │
└────────────────┬───────────────┘ └────────────────┬───────────────┘ └────────────────┬───────────────┘
                 │                                  │                                  │
                 ▼                                  ▼                                  ▼
┌────────────────────────────────┐ ┌────────────────────────────────┐ ┌────────────────────────────────┐
│      OUTPUT ARTIFACTS 1        │ │      OUTPUT ARTIFACTS 2        │ │      OUTPUT ARTIFACTS 3        │
│  results/runs/{dataset}/       │ │  results/features/{dataset}/   │ │  results/labels/{dataset}/     │
│  results/latency/{dataset}.json│ │  results/models/{dataset}/     │ │  results/metrics/{dataset}/    │
│  results/experiment3/          │ │  results/synthetic/            │ │  results/experiment4/          │
└────────────────────────────────┘ └────────────────────────────────┘ └────────────────────────────────┘
```

1. **Contract-First Data Serialization**:
   All communication between tracks happens via strictly typed on-disk artifacts (Parquet / JSON). For example, Track 2 and Track 3 consume candidate rankings from `results/runs/{dataset}/{retriever}.parquet` using a frozen schema (`query_id`, `doc_id`, `score`, `rank`).
2. **Instant Parallel Start via Pre-Cached Runs**:
   Base runs are not committed (`results/runs/`, `results/indexes/` and `data/raw/` are gitignored). What is committed is the per-query route table `results/experiment2/cache/{dataset}.csv` (nDCG@10 for BM25, dense and every α, plus the 18 features), from which Tracks 2 and 3 run fully offline. The runs are regenerated from `data/raw` with one command per stage: `python scripts/download_data.py`, then `python scripts/run_retrieval_baselines.py` (BM25 + dense runs for NFCorpus, SciFact, FiQA) and `python scripts/run_experiment1b_query_types.py` (CQADupStack Unix and the Quora subset), both cached as `.trec` in `results/runs/{dataset}/`, then `python scripts/export_run_parquet.py` (Contract 1 parquet from those runs).
3. **Zero-Overlap File & Directory Ownership**:
   Each person has 100% exclusive write ownership of specific modules in `src/`, scripts in `scripts/`, test suites in `tests/`, and output folders in `results/`. Merge conflicts are structurally prevented.
4. **Offline Isolation & Mock Fixtures**:
   Every track includes standalone synthetic fixtures (in `tests/conftest.py`). A developer can run, test, and validate their entire subsystem on 5 mock queries and 10 mock documents in <2 seconds with zero GPU or network dependencies.

---

## 2. The 3-Person Division of Labor

### 2.1 Person 1 — Track 1: Retrieval Infrastructure, Latency Profiling & Skip-Dense Efficiency
*The "Retrieval & Systems Efficiency Engineer"*

* **Mission**: Own first-stage retrieval engines (lexical, dense, hybrid fusion, RRF), establish wall-clock latency measurement and FLOPs profiling (addressing Issue D3), and implement the **Skip-Dense Retrieval Engine** (Experiment 3: skipping dense vector search when BM25 is confident to slash latency without quality degradation).

#### Owned Modules & Files
| Category | File Paths Owned |
| :--- | :--- |
| **Source Modules** | `src/router_retrieval/retrieval/bm25.py`<br>`src/router_retrieval/retrieval/dense.py`<br>`src/router_retrieval/retrieval/normalization.py`<br>`src/router_retrieval/retrieval/hybrid.py`<br>`src/router_retrieval/retrieval/rrf.py`<br>`src/router_retrieval/pipeline/latency.py`<br>`src/router_retrieval/evaluation/latency_report.py` |
| **Scripts** | `scripts/build_indexes.py`<br>`scripts/run_retrieval_baselines.py`<br>`scripts/run_latency_benchmark.py`<br>`scripts/run_experiment3_skip_dense.py` |
| **Unit Tests** | `tests/test_bm25.py`<br>`tests/test_dense.py`<br>`tests/test_normalization.py`<br>`tests/test_hybrid.py`<br>`tests/test_rrf.py`<br>`tests/test_experiment3.py` |
| **Configurations** | `configs/retrievers.yaml`<br>`configs/hybrid.yaml` |
| **Results** | `results/indexes/`<br>`results/runs/`<br>`results/latency/`<br>`results/experiment3/` |
| **Report Sections** | **Section 3** (Retrieval Setup, Indexing, and Hyperparameter Grids)<br>**Section 4.2** (Latency Benchmarks, Wall-Clock Profiling, and Skip-Dense Efficiency Frontiers) |

#### Detailed Responsibilities & Milestones
1. **Tokenizer & Indexing Pipeline Robustness**:
   - Fix tokenizer stemming discrepancies in `tests/test_bm25.py` to ensure reproducible Lucene/Porter normalization across OS environments.
   - Implement clean CLI `scripts/build_indexes.py` supporting BM25 (Pyserini / rank_bm25) and Dense embeddings (FAISS flat IP index with `intfloat/e5-small-v2` and `facebook/contriever`).
2. **Score Normalization & Hybrid Baselines**:
   - Maintain min-max score normalization with floor anchoring (`retrieval/normalization.py`) so sparse matches are never dropped (Bug B1 prevention).
   - Implement convex combination `HybridRetriever(alpha)` and rank-based Reciprocal Rank Fusion (`retrieval/rrf.py`).
   - Run baseline sweeps across $\alpha \in [0.0, 1.0]$ in 5-fold CV to establish the non-strawman benchmark.
3. **Dedicated Latency & FLOPs Profiling Suite (Addressing Issue D3)**:
   - Build `pipeline/latency.py` context-manager timers measuring mean, median, p95, and p99 wall-clock latency per component: BM25 search, dense query encoding, FAISS index search, feature extraction, router forward pass, and cross-encoder reranking.
   - Run `scripts/run_latency_benchmark.py` across CPU vs. GPU environments on FiQA, NFCorpus, and SciFact, producing publication-ready latency distribution curves.
4. **Skip-Dense Efficiency Engine (Experiment 3 Deep-Dive)**:
   - Own `scripts/run_experiment3_skip_dense.py`. Evaluate routing policies that decide whether to execute dense retrieval based *only* on pre-retrieval and BM25 signals (top-1/top-2 score ratio, max-IDF, score entropy).
   - Compute Pareto frontiers of Quality (nDCG@10) vs. Dense Retrieval Calls Skipped (20%–60% compute reduction target). Compare learned LightGBM skip-classifier against one-feature heuristic baselines (addressing Issue D1/D2).

#### Independent Developer Playbook (How to work without waiting)
* Person 1 works directly from `data/raw/{dataset}/` and BEIR downloads.
* Runs can be tested locally using a 100-doc synthetic fixture or the lightweight `scifact` corpus.
* Person 1 is completely unblocked from router modeling or cross-encoders.

---

### 2.2 Person 2 — Track 2: Query Feature Engineering, Synthetic Scaling & Advantage Routing
*The "Router Modeling & Data Scaling Engineer"*

* **Mission**: Own query representations and interaction features, eliminate the sample-size statistical bottleneck (Issue A6) via **Synthetic Dataset Generation at Scale** (scaling from 3.3k to 30,000+ queries per `improvements.md §3`), and implement the **Dual Advantage Regression Router** (replacing noisy 3-class classification with continuous gain prediction per `improvements.md §2`).

#### Owned Modules & Files
| Category | File Paths Owned |
| :--- | :--- |
| **Source Modules** | `src/router_retrieval/features/query_features.py`<br>`src/router_retrieval/features/retrieval_signal_features.py`<br>`src/router_retrieval/features/feature_builder.py`<br>`src/router_retrieval/router/base.py`<br>`src/router_retrieval/router/logistic_router.py`<br>`src/router_retrieval/router/gbm_router.py`<br>`src/router_retrieval/router/oracle_router.py` |
| **Scripts** | `scripts/extract_features.py`<br>`scripts/train_router.py`<br>`scripts/run_experiment2_heldout_routing.py`<br>`scripts/demo_synthetic_routing.py`<br>`scripts/run_ablations.py`<br>`scripts/run_cross_domain.py` |
| **Unit Tests** | `tests/test_query_features.py`<br>`tests/test_retrieval_signal_features.py`<br>`tests/test_router_logistic.py`<br>`tests/test_router_gbm.py` |
| **Configurations** | `configs/router.yaml` |
| **Results** | `results/features/`<br>`results/models/`<br>`results/experiment2/`<br>`results/synthetic/` |
| **Report Sections** | **Section 2** (Methodology: Dual Advantage Regression, Feature Representations, and Synthetic Scaling)<br>**Section 4.3** (Feature Ablation Studies: Query-Only vs. Retrieval-Signal vs. Joint, and Out-of-Domain Generalization) |

#### Detailed Responsibilities & Milestones
1. **Feature Extraction Engine & Disk Caching**:
   - Maintain `features/query_features.py`: query length, max/mean IDF, rare-term density, entity density, and out-of-vocabulary (OOV) ratio.
   - Maintain `features/retrieval_signal_features.py`: BM25/dense top-10 Jaccard overlap, rank correlation (Kendall tau / Spearman rho), normalized top-1 scores, top-1/top-2 score gaps, score ratios, and Shannon score entropy.
   - Build high-speed Parquet disk caching in `features/feature_builder.py` and CLI `scripts/extract_features.py`.
2. **Dual Advantage Regression Formulation (Solving Issues A1, A2 & A5)**:
   - Implement continuous regressors in `router/gbm_router.py` predicting margins over hybrid:
     $$\Delta_{\text{BM25}}(q) = \text{nDCG@10}(\text{BM25}, q) - \text{nDCG@10}(\text{Hybrid}, q)$$
     $$\Delta_{\text{Dense}}(q) = \text{nDCG@10}(\text{Dense}, q) - \text{nDCG@10}(\text{Hybrid}, q)$$
   - Calibrate risk margins $\theta_{\text{bm25}}, \theta_{\text{dense}}$ on training folds to prevent false routing on near-ties, maintaining Hybrid as the safe default whenever predictions are ambiguous.
3. **Large-Scale Synthetic Dataset Scaling (Improvements §3 Protocol)**:
   - Scale `scripts/demo_synthetic_routing.py` into a production training generator producing 30,000+ synthetic queries across NFCorpus, SciFact, and FiQA.
   - Implement dual generation styles:
     - *Lexical / Entity Style (50%)*: High-IDF keyword/identifier queries favoring BM25.
     - *Semantic / Conceptual Style (50%)*: Paraphrastic/abstract questions favoring Dense.
   - Enforce strict test-set isolation: passages relevant to natural evaluation queries are strictly forbidden from synthetic training sets.
4. **Feature Ablations & Cross-Domain Generalization**:
   - Own `scripts/run_ablations.py`: benchmark router performance using (a) query-only features, (b) retrieval-signal features, and (c) all features combined.
   - Own `scripts/run_cross_domain.py`: train on dataset $X$ (e.g. FiQA) and test zero-shot on dataset $Y$ (e.g. SciFact) to quantify domain transfer degradation.

#### Independent Developer Playbook (How to work without waiting)
* Person 2 loads candidate runs directly from `results/runs/{dataset}/` which already exist in the repository.
* When experimenting with synthetic generation, Person 2 works directly from raw corpus texts (`data/raw/{dataset}/corpus.jsonl`).
* Person 2 does not need the cross-encoder or final evaluation scripts to train, ablate, and evaluate routers.

---

### 2.3 Person 3 — Track 3: Escalation Backend, Cross-Encoder Optimization, Rigorous Evaluation & Pipeline Integration
*The "Escalation, Statistical Evaluation & Integration Lead"*

* **Mission**: Own the selective cross-encoder escalation pipeline (Experiment 4), resolve **Issue D4** by upgrading the cross-encoder to a stronger reranker (e.g., `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-12-v2` to guarantee escalation produces a true positive quality headroom), implement budget-constrained escalation policies, build the unified end-to-end integration harness, execute rigorous statistical validation (paired bootstrap and Wilcoxon tests), and lead final report synthesis.

#### Owned Modules & Files
| Category | File Paths Owned |
| :--- | :--- |
| **Source Modules** | `src/router_retrieval/retrieval/cross_encoder.py`<br>`src/router_retrieval/router/escalation.py`<br>`src/router_retrieval/labeling/per_route_eval.py`<br>`src/router_retrieval/labeling/margin_labeler.py`<br>`src/router_retrieval/labeling/regret.py`<br>`src/router_retrieval/evaluation/ranking_metrics.py`<br>`src/router_retrieval/evaluation/router_metrics.py`<br>`src/router_retrieval/evaluation/significance.py`<br>`src/router_retrieval/evaluation/risk_coverage.py`<br>`src/router_retrieval/pipeline/controls.py`<br>`src/router_retrieval/pipeline/end_to_end.py` |
| **Scripts** | `scripts/generate_labels.py`<br>`scripts/run_oracle_headroom.py`<br>`scripts/sweep_escalation.py`<br>`scripts/run_experiment4_escalation.py`<br>`scripts/run_full_evaluation.py`<br>`scripts/build_hybrid_outperformed_sets.py` |
| **Unit Tests** | `tests/test_cross_encoder.py`<br>`tests/test_escalation.py`<br>`tests/test_margin_labeler.py`<br>`tests/test_ranking_metrics.py`<br>`tests/test_significance.py`<br>`tests/test_end_to_end_pipeline.py` |
| **Configurations** | `configs/cross_encoder.yaml`<br>`configs/escalation.yaml`<br>`configs/labels.yaml`<br>`configs/experiment.yaml` |
| **Results** | `results/labels/`<br>`results/metrics/`<br>`results/experiment4/`<br>`results/figures/` |
| **Report Sections** | **Abstract & Section 1** (Introduction, Core Hypotheses H1–H4, and Literature Positioning)<br>**Section 4.1** (End-to-End Retrieval Performance, Headroom Analysis, and Escalation Trade-offs)<br>**Section 5** (Discussion, Limitations, Analysis of Negative Findings, and Master Document Compilation) |

#### Detailed Responsibilities & Milestones
1. **Cross-Encoder Upgrade & Headroom Restoration (Solving Issue D4)**:
   - In `retrieval/cross_encoder.py`, integrate modern, high-capacity rerankers (configurable via `configs/cross_encoder.yaml`: `BAAI/bge-reranker-base` and `cross-encoder/ms-marco-MiniLM-L-12-v2`) in addition to default `MiniLM-L-6-v2`.
   - Recompute reranking on hybrid candidate pools to confirm that `Always-Rerank > Always-Hybrid` on the target benchmarks, restoring the essential prerequisite for escalation.
2. **Budgeted Selective Escalation (Experiment 4 Execution)**:
   - Own `router/escalation.py` and `scripts/run_experiment4_escalation.py`. Implement selective escalation policies at fixed budgets $B \in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]$:
     - *Policy 1 (Predicted-Gain)*: LightGBM regressor predicting $\text{nDCG}_{\text{Rerank}} - \text{nDCG}_{\text{Hybrid}}$.
     - *Policy 2 (Router Confidence)*: Escalating when $\max_r P(\text{route}=r) < \tau$.
     - *Policy 3 (Random Control)*: Escalating random queries at identical budget $B$.
     - *Policy 4 (Oracle Upper Bound)*: Escalating true highest-gain queries.
   - Generate Risk-Coverage and Quality-Retention curves across budget sweeps (`evaluation/risk_coverage.py`).
3. **Rigorous Statistical Validation & Diagnostics (Addressing Issues A1, A2, D6)**:
   - Standardize `evaluation/ranking_metrics.py` (nDCG@10, MRR@10, MAP, Recall@100) and `evaluation/significance.py` (paired bootstrap test with $B=10{,}000$ resamples and paired Wilcoxon signed-rank tests with 95% confidence intervals).
   - Format win/loss/tie contingency tables and confusion matrices (`evaluation/router_metrics.py`). Handle edge cases where denominators collapse gracefully (Issue D6 fix).
4. **End-to-End Orchestration & Report Lead**:
   - Complete `pipeline/end_to_end.py` and `scripts/run_full_evaluation.py` to produce unified comparison tables across all systems: BM25, Dense, Tuned Hybrid, RRF, Skip-Dense (from P1), Advantage Router (from P2), Selective Escalation (from P3), Always-Rerank, and Oracle.
   - Lead master compilation of `report/report.md`, aggregating figures, tables, and honest discussion of hypotheses.

#### Independent Developer Playbook (How to work without waiting)
* Person 3 uses existing cached runs in `results/runs/` to compute ground-truth labels and candidate pools for cross-encoder reranking.
* To test escalation and ranking metrics, Person 3 can run completely offline using mock candidate lists or pre-computed parquet files.
* Person 3 is not dependent on router training or feature extraction to benchmark rerankers and escalation policies.

---

## 3. Data Interface Contracts & File Schemas

To maintain complete autonomy, all data passed between tracks follows these immutable schemas:

### Contract 1: Candidate Pool Run Schema (`results/runs/{dataset}/{retriever}.parquet`)
*Produced by Person 1; Consumed by Person 2 & Person 3*
| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `query_id` | `str` | Unique query identifier (strictly string to avoid Bug B2) | `"302"` |
| `doc_id` | `str` | Unique document identifier in corpus | `"MED-1049"` |
| `score` | `float32` | Raw retriever score | `14.821` |
| `rank` | `int32` | 1-based rank position within top-$k$ | `1` |

### Contract 2: Feature Matrix Schema (`results/features/{dataset}.parquet`)
*Produced by Person 2; Consumed by Person 2 & Person 3*
| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `query_id` | `str` | Unique query identifier | `"302"` |
| `query_length` | `float32` | Word count of query string | `7.0` |
| `max_idf` | `float32` | Maximum BM25 corpus IDF of query terms | `8.41` |
| `mean_idf` | `float32` | Average BM25 corpus IDF of query terms | `4.12` |
| `rare_term_density` | `float32` | Fraction of query terms with IDF > threshold | `0.428` |
| `entity_density` | `float32` | Heuristic capitalized / alphanumeric entity density | `0.142` |
| `oov_fraction` | `float32` | Fraction of terms not in corpus vocabulary | `0.0` |
| `jaccard_overlap` | `float32` | Top-10 Jaccard overlap between BM25 and Dense | `0.25` |
| `rank_correlation` | `float32` | Spearman rank correlation over shared top candidates (alias of `rank_correlation_spearman`) | `0.38` |
| `bm25_top1_score` | `float32` | Raw (unnormalized) top-1 BM25 score | `14.82` |
| `bm25_top1_top2_ratio`| `float32`| Ratio of top-1 to top-2 BM25 score | `1.34` |
| `bm25_score_entropy` | `float32`| Shannon entropy of softmaxed candidate scores | `2.11` |
| `dense_top1_score` | `float32` | Raw top-1 dense (cosine) score | `0.86` |
| `dense_top1_top2_gap`| `float32`| Difference between top-1 and top-2 Dense scores | `0.08` |
| `dense_score_entropy`| `float32`| Shannon entropy of Dense candidate distribution | `2.84` |

### Contract 3: Router Predictions Schema (`results/models/{dataset}/predictions.parquet`)
*Produced by Person 2; Consumed by Person 3*
| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `query_id` | `str` | Unique query identifier | `"302"` |
| `fold` | `int32` | Cross-validation fold index | `2` |
| `route_predicted` | `str` | Predicted route (`"bm25"`, `"dense"`, `"hybrid"`) | `"dense"` |
| `p_bm25` | `float32` | Calibrated probability for BM25 route | `0.15` |
| `p_dense` | `float32` | Calibrated probability for Dense route | `0.65` |
| `p_hybrid` | `float32` | Calibrated probability for Hybrid route | `0.20` |
| `delta_pred_bm25` | `float32` | Advantage regression estimate $\hat{\Delta}_{\text{BM25}}$ | `-0.04` |
| `delta_pred_dense`| `float32` | Advantage regression estimate $\hat{\Delta}_{\text{Dense}}$ | `+0.07` |

### Contract 4: Escalation Candidates & Predictions (`results/experiment4/{dataset}_escalation.parquet`)
*Produced and Managed by Person 3*
| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `query_id` | `str` | Unique query identifier | `"302"` |
| `budget` | `float32` | Escalation budget fraction ($B$) | `0.10` |
| `policy` | `str` | Policy name (`"predicted_gain"`, `"confidence"`, `"random"`, `"oracle"`) | `"predicted_gain"` |
| `escalated` | `bool` | True if query was sent to cross-encoder | `True` |
| `ndcg_final` | `float32` | Final achieved nDCG@10 under policy | `0.421` |
| `latency_ms` | `float32` | Wall-clock query latency | `48.3` |

### Contract 5: Standardized Evaluation Metrics (`results/metrics/{dataset}/metrics.json`)
*Produced by Person 3; Consumed by All for Report*
```json
{
  "dataset": "fiqa",
  "system": "advantage_router_with_escalation_10pct",
  "ndcg_10": { "mean": 0.358, "ci_95": [0.342, 0.374] },
  "mrr_10": { "mean": 0.341, "ci_95": [0.325, 0.357] },
  "map": { "mean": 0.312, "ci_95": [0.298, 0.326] },
  "recall_100": { "mean": 0.692, "ci_95": [0.675, 0.709] },
  "latency_ms": { "mean": 18.4, "p50": 11.2, "p95": 54.1 },
  "headroom_vs_hybrid": { "delta": +0.021, "p_value": 0.004, "significant": true }
}
```

---

## 4. Synchronized Implementation Timeline & Parallel Milestones

```
Timeline      Person 1 (Retrieval & Efficiency)    Person 2 (Features & Routing)         Person 3 (Escalation & Evaluation)
────────────  ───────────────────────────────────  ────────────────────────────────────  ───────────────────────────────────
Week 1        • Fix test_bm25 tokenizer stems      • Extract features on existing runs   • Benchmark upgraded cross-encoders
              • Build CLI scripts/build_indexes    • Implement FeatureBuilder caching    • Confirm Always-Rerank > Hybrid
              • Verify score normalizations        • Implement Dual Advantage Regressor  • Build evaluation metrics wrapper

Week 2        • Instrument latency.py timers       • Train Advantage Router on Exp 2     • Run selective escalation (Exp 4)
              • Benchmark CPU vs GPU latencies     • Prototype synthetic query generator • Compute risk-coverage curves
              • Verify alpha sweep baselines       • Test query-style prompt templates   • Implement bootstrap significance

[MILESTONE 1] ─── DATA ARTIFACT HANDSHAKE: Index & Run Caches / Feature Parquets / Upgraded Cross-Encoder Pools ──────────

Week 3        • Implement Exp 3 Skip-Dense         • Scale synthetic queries to 30,000+  • Integrate End-to-End Pipeline
              • Test BM25 top ratio heuristic      • Re-train Advantage Regressor at 30k • Build control policies harness
              • Measure wall-clock skip savings    • Perform feature ablation studies    • Generate win/loss contingency tables

Week 4        • Map latency-quality Pareto curves  • Run cross-domain transfer tests     • Execute run_full_evaluation.py
              • Finalize results/latency/ tables   • Profile feature importances         • Pool metrics across 5 datasets
              • Draft Report Section 3 & 4.2       • Draft Report Section 2 & 4.3        • Draft Report Section 1, 4.1 & 5

[MILESTONE 2] ─── SYSTEM INTEGRATION FREEZE: All Experiments Complete & Figures Generated ────────────────────────────────

Week 5        • Cross-review experimental logs     • Review ablation and scaling tables  • Compile master report.md
              • Verify latency reproducibility     • Format synthetic data appendix      • Polish final camera-ready report
```

---

## 5. Team Governance, Git Strategy & Conflict Prevention

To maintain independent workflows in Git:

1. **Dedicated Track Branches**:
   - Person 1 commits to `feature/track1-retrieval-latency`
   - Person 2 commits to `feature/track2-router-synthetic`
   - Person 3 commits to `feature/track3-escalation-eval`
2. **Pull Request Protocol**:
   - Merges to `master` require green test suite execution (`pytest`).
   - PRs must only modify files within the author's designated ownership table (§2).
3. **Artifact Isolation**:
   - Intermediate model checkpoints and per-query CSVs land in dedicated subdirectories (`results/experiment3/`, `results/synthetic/`, `results/experiment4/`), preventing untracked merge collisions.
4. **Issue & Negative Finding Transparency**:
   - In accordance with the findings in `experiment_issues.md`, negative results (e.g. inability of cheap features to beat hybrid, or one-feature heuristics matching learned models) are treated as **first-class scientific contributions**. Team members document what was attempted, the confidence intervals, and the theoretical reason for the outcome without forcing artificial claims.

---

## 6. Report & Presentation Responsibility Matrix

The final project deliverable (`report/report.md` and presentation slides) is divided as follows:

| Section | Title / Content | Primary Owner | Reviewer |
| :--- | :--- | :--- | :--- |
| **Abstract** | Executive summary of research questions, key findings, and efficiency results | **Person 3** | Person 1, Person 2 |
| **Section 1** | **Introduction & Hypotheses**: Motivation, trade-offs, formal hypotheses H1–H4, literature positioning (DAT, Pinecone, Adaptive-RAG) | **Person 3** | Person 2 |
| **Section 2** | **Methodology & Architecture**: Dual Advantage Regression, Feature Engineering (Lexical & Interaction), and Synthetic Scaling Protocol | **Person 2** | Person 3 |
| **Section 3** | **Experimental Setup & Baselines**: Datasets (NFCorpus, SciFact, FiQA, Unix, Quora), retrievers, $\alpha$-sweeps, normalization, and test splits | **Person 1** | Person 3 |
| **Section 4.1** | **Retrieval Quality & Headroom**: nDCG@10/MRR/MAP, Headroom analysis, Selective Escalation results, and Bootstrap Significance | **Person 3** | Person 2 |
| **Section 4.2** | **Latency, FLOPs & Efficiency Trade-offs**: Wall-clock CPU/GPU timings, per-component breakdowns, and Skip-Dense Pareto frontiers | **Person 1** | Person 3 |
| **Section 4.3** | **Ablation Studies & Scaling Impact**: Feature group ablations, Synthetic dataset scaling curves (3k vs 30k), and Cross-domain transfer | **Person 2** | Person 1 |
| **Section 5** | **Discussion & Limitations**: Honest analysis of negative results (Issues A1–A9, D1–D6), failure modes, and deployment guidelines | **Person 3** | Person 1, Person 2 |
| **Presentation** | Slide Deck creation & visual design | **All Hands** (P1: Setup/Latency, P2: Method/Ablations, P3: Results/Escalation) | Lead: P3 |

---

## 7. Master Risk & Mitigation Register

| Identified Risk | Impacted Track | Pre-Emptive Mitigation |
| :--- | :---: | :--- |
| **Statistical power too low on small BEIR sets** (Issue A6) | Track 2 & 3 | Person 2 scales training via 30,000 synthetic queries; Person 3 runs 10,000 bootstrap resamples with pooled reporting. |
| **Cross-encoder weaker than hybrid** (Issue D4) | Track 3 | Person 3 benchmarks higher-capacity models (`bge-reranker-base`, `ms-marco-MiniLM-L-12-v2`) in Phase 1 before running escalation. |
| **Learned router matches simple heuristic** (Issue D1) | Track 1 & 2 | Document as an interpretability win: show that a transparent top-1/top-2 ratio heuristic yields 90%+ of the learned model's savings. |
| **Retrieval latency dominated by network/IO** | Track 1 | Person 1 isolates benchmark runs locally using in-memory FAISS indices and warms up PyTorch caches prior to timing. |
| **Merge conflicts on shared files** | All | Zero-overlap file ownership (§2) and contract-first architecture (§3). |
