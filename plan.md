# Implementation Plan: Confidence-Based Query Router for Hybrid Retrieval

This document translates `proposal.md` into a concrete, step-by-step engineering plan with a full repository layout. It is written so that any contributor (human or coding agent) can pick up a module, understand its inputs/outputs/contracts, and implement it without needing to re-derive design decisions already made in the proposal.

---

## 0. Guiding Contracts (read first)

These are fixed decisions from the proposal that every module must respect. Do not re-litigate them without updating this file.

1. **Datasets:** BEIR subsets only, chosen for feasibility: NFCorpus, SciFact, FiQA (primary); TREC-COVID optional/stretch. SciDocs optional.
2. **Retrievers (cheap routes):**
   - BM25 (Pyserini or `rank_bm25`)
   - Dense (Contriever or a `sentence-transformers` model, FAISS CPU index)
   - Hybrid = fixed-α convex combination of **normalized** BM25 and dense scores.
3. **Escalation backend:** MiniLM-style cross-encoder reranking the **union** of BM25 top-100 and dense top-100 (optionally truncated to top 30–50 candidates for latency).
4. **Router:** multi-class classifier over 3 cheap routes (BM25 / dense / hybrid). Escalation is **not** a 4th class in the main design — it is triggered when `max(class_probability) < τ`. (A 4th-class variant is a documented stretch thread, see §9.)
5. **Labels:** per-query argmax over {BM25, dense, hybrid} nDCG@10, with margin threshold ε — if the top two routes are within ε, label = hybrid (safe default). Raw per-route nDCG values are retained for regret-weighted loss.
6. **Models:** Logistic Regression (baseline router) and LightGBM/XGBoost (main router), both with class weighting.
7. **Thresholds (ε for labeling, τ for escalation):** tuned on training folds only, never on test folds.
8. **Baselines/controls:** BM25-only, dense-only, fixed-α hybrid (swept), RRF, always-rerank, random-escalation-at-same-rate, oracle router, always-hybrid+rerank-random-subset.
9. **Metrics:** nDCG@10, MRR, MAP, Recall@100 (primary); macro-F1 + confusion matrix (router diagnostic, secondary); latency mean/p95 per component.
10. **Validation:** k-fold cross-validation over queries + paired bootstrap significance testing on per-query nDCG@10.
11. **Week-2 checkpoint gate:** compute oracle-router headroom over hybrid. If headroom < ~2 nDCG points, revisit dataset choice before investing further in router modeling.
12. **Team lanes** (for parallel work once shared infra lands):
    - **Lane A — Retrieval infra:** BM25/dense pipelines, score normalization, fixed-α + RRF baselines, cross-encoder escalation backend, always-rerank baseline, latency benchmarking.
    - **Lane B — Labels & analysis:** per-query per-route nDCG, margin-based labeling, oracle headroom analysis, significance testing.
    - **Lane C — Router modeling:** feature extraction, router training (regret-weighted loss), τ sweep, risk-coverage curves, ablations, writeup coordination.

---

## 1. Repository File Structure

```
ire_project/
├── README.md                          # project overview, setup, how to reproduce results
├── plan.md                            # this file
├── proposal.md                        # original proposal (already in repo)
├── pyproject.toml                     # or requirements.txt — pinned deps
├── requirements.txt
├── .gitignore                         # data/, *.index, *.pt, wandb/, .cache/, __pycache__/
├── Makefile                           # convenience targets: make data, make baselines, make train, make eval
├── configs/
│   ├── datasets.yaml                  # dataset names, BEIR paths/URLs, doc/query counts
│   ├── retrievers.yaml                # BM25 params (k1, b), dense model name, FAISS index type
│   ├── hybrid.yaml                    # alpha grid for sweep, normalization method
│   ├── cross_encoder.yaml             # model name, batch size, candidate pool size (30/50/100)
│   ├── labels.yaml                    # epsilon grid, labeling strategy flag
│   ├── router.yaml                    # feature list, model type, hyperparams, class weights
│   ├── escalation.yaml                # tau grid (0.05, 0.10, 0.15 escalation-rate targets)
│   └── experiment.yaml                # top-level: which datasets/models/seeds to run end-to-end
│
├── data/                              # gitignored; populated by scripts/download_data.py
│   ├── raw/{dataset}/                 # corpus.jsonl, queries.jsonl, qrels/
│   └── processed/{dataset}/           # tokenized/cleaned copies if needed
│
├── src/
│   └── router_retrieval/              # installable package (pip install -e .)
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── beir_loader.py         # load BEIR corpus/queries/qrels into a common schema
│       │   ├── schema.py              # dataclasses: Document, Query, QRel, Candidate
│       │   └── splits.py              # k-fold CV split generation over queries (seeded)
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── bm25.py                # BM25Retriever: index(), search(query, k) -> List[Candidate]
│       │   ├── dense.py               # DenseRetriever: encode corpus, build FAISS index, search()
│       │   ├── normalization.py       # min-max / z-score normalization of BM25 & dense scores
│       │   ├── hybrid.py              # HybridRetriever(alpha): convex combination of normalized scores
│       │   ├── rrf.py                 # ReciprocalRankFusion: rank-based fusion baseline
│       │   └── cross_encoder.py       # CrossEncoderReranker: rerank(query, candidates) -> reranked list
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── query_features.py      # length, max/mean IDF, rare-term density, entity density, OOV frac
│       │   ├── retrieval_signal_features.py  # top-k Jaccard overlap, rank correlation (Kendall/Spearman),
│       │   │                                  # top-1 score, top1-top2 gap, score entropy per retriever
│       │   └── feature_builder.py     # orchestrates: query -> full feature vector (+ caching to disk)
│       │
│       ├── labeling/
│       │   ├── __init__.py
│       │   ├── per_route_eval.py      # compute nDCG@10/MRR/MAP/Recall@100 per query per route
│       │   ├── margin_labeler.py      # argmax-with-margin-epsilon labeling -> best_route label
│       │   └── regret.py              # regret weights = best_nDCG - route_nDCG, for weighted loss
│       │
│       ├── router/
│       │   ├── __init__.py
│       │   ├── base.py                # RouterModel interface: fit(), predict_proba(), predict()
│       │   ├── logistic_router.py     # sklearn LogisticRegression wrapper, class_weight='balanced'
│       │   ├── gbm_router.py          # LightGBM/XGBoost wrapper, regret-weighted sample_weight
│       │   ├── escalation.py          # apply tau threshold to predict_proba -> route or ESCALATE
│       │   └── oracle_router.py       # "cheats" using true labels; upper-bound baseline
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── end_to_end.py          # given a query -> route decision -> final ranked list
│       │   ├── controls.py            # random-escalation, always-hybrid+random-rerank, always-rerank
│       │   └── latency.py             # per-component timers (BM25, dense, feature extraction,
│       │                               # router inference, cross-encoder), context-manager based
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ranking_metrics.py     # nDCG@10, MRR, MAP, Recall@100 (reuse pytrec_eval or ir_measures)
│       │   ├── router_metrics.py      # macro-F1, confusion matrix, per-class precision/recall
│       │   ├── significance.py        # paired bootstrap test on per-query nDCG@10
│       │   ├── risk_coverage.py       # risk-coverage curve computation across tau sweep
│       │   └── latency_report.py      # aggregate mean/p95 latency by component, by dataset
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py              # YAML config loading + validation
│           ├── seeding.py             # global seed control for reproducibility
│           ├── logging_utils.py       # structured run logging
│           └── caching.py             # disk cache for embeddings/features/retrieval results
│
├── scripts/                           # thin CLI entry points, one responsibility each
│   ├── download_data.py               # fetch BEIR datasets into data/raw/
│   ├── build_indexes.py               # build BM25 + FAISS indexes for a dataset
│   ├── run_retrieval_baselines.py     # BM25-only, dense-only, hybrid-sweep, RRF -> results/
│   ├── run_oracle_headroom.py         # WEEK-2 CHECKPOINT: compute oracle vs hybrid headroom
│   ├── generate_labels.py             # per-route nDCG + margin labeling -> labels/{dataset}.parquet
│   ├── extract_features.py            # build feature matrix -> features/{dataset}.parquet
│   ├── train_router.py                # train logistic + GBM routers, save models/
│   ├── sweep_escalation.py            # sweep tau, produce risk-coverage curve data
│   ├── run_full_evaluation.py         # run router + all baselines/controls, compute all metrics
│   ├── run_latency_benchmark.py       # dedicated latency measurement pass
│   ├── run_ablations.py               # feature ablation study (query-only vs retrieval-signal vs both)
│   └── run_cross_domain.py            # stretch: train on dataset X, test on dataset Y
│
├── notebooks/                         # exploratory only; nothing load-bearing lives only here
│   ├── 00_data_exploration.ipynb
│   ├── 01_headroom_checkpoint.ipynb
│   ├── 02_feature_inspection.ipynb
│   ├── 03_router_error_analysis.ipynb
│   └── 04_risk_coverage_plots.ipynb
│
├── results/                           # gitignored or git-lfs; generated artifacts
│   ├── indexes/{dataset}/             # BM25 index files, FAISS index files
│   ├── labels/{dataset}.parquet
│   ├── features/{dataset}.parquet
│   ├── models/{dataset}/{router_type}.pkl
│   ├── metrics/{dataset}/*.json       # per-dataset metric tables
│   ├── metrics/pooled/*.json          # cross-dataset pooled results
│   ├── latency/{dataset}.json
│   ├── figures/                       # risk-coverage curves, confusion matrices, ablation bar charts
│   └── logs/                          # run logs per script invocation
│
├── tests/
│   ├── __init__.py
│   ├── test_bm25.py
│   ├── test_dense.py
│   ├── test_normalization.py
│   ├── test_hybrid.py
│   ├── test_rrf.py
│   ├── test_cross_encoder.py
│   ├── test_query_features.py
│   ├── test_retrieval_signal_features.py
│   ├── test_margin_labeler.py
│   ├── test_router_logistic.py
│   ├── test_router_gbm.py
│   ├── test_escalation.py
│   ├── test_ranking_metrics.py
│   ├── test_significance.py
│   └── test_end_to_end_pipeline.py
│
└── report/
    ├── report.md / report.tex         # final writeup
    ├── figures/                       # final polished figures for the report
    └── tables/                        # final result tables (csv/markdown) for the report
```

---

## 2. Step-by-Step Execution Plan

### Phase 0 — Setup (Week 1, all hands)

1. Initialize repo structure above; set up `pyproject.toml`/`requirements.txt` with pinned versions: `rank_bm25` or `pyserini`, `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `lightgbm`, `xgboost`, `pytrec_eval` or `ir_measures`, `pandas`, `numpy`, `pyyaml`, `pytest`.
2. Write `src/router_retrieval/data/schema.py` — shared dataclasses (`Document`, `Query`, `QRel`, `Candidate`) used by every downstream module so retrieval, labeling, and features all speak the same format.
3. Write `scripts/download_data.py` to pull NFCorpus, SciFact, FiQA (and optionally TREC-COVID) via the BEIR dataset distribution; cache to `data/raw/{dataset}/`.
4. Write `src/router_retrieval/data/beir_loader.py` to parse each dataset into the common schema, and `src/router_retrieval/data/splits.py` for seeded k-fold CV splits over queries.
5. Agree on `configs/*.yaml` defaults as a team (BM25 k1/b, dense model name — e.g. `facebook/contriever` or `sentence-transformers/msmarco-distilbert-base-v4`, cross-encoder model — e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`, alpha grid, epsilon grid, tau grid).

**Deliverable:** datasets loadable end-to-end into the common schema; configs agreed; repo skeleton committed.

### Phase 1 — Retrieval Infra (Lane A, Weeks 1–2)

6. Implement `retrieval/bm25.py`: index corpus, `search(query, k)` returning scored candidates.
7. Implement `retrieval/dense.py`: encode corpus with the chosen embedding model, build a FAISS CPU index, `search(query, k)`.
8. Implement `retrieval/normalization.py`: min-max and/or z-score normalization for BM25 and dense scores (BM25 is unbounded, dense is bounded — normalize before any fusion, per Pinecone reference in the proposal).
9. Implement `retrieval/hybrid.py`: `HybridRetriever(alpha)` combining normalized scores; implement the alpha sweep in `scripts/run_retrieval_baselines.py`.
10. Implement `retrieval/rrf.py`: standard reciprocal rank fusion.
11. Implement `retrieval/cross_encoder.py`: load MiniLM cross-encoder, `rerank(query, candidates)` over the BM25∪dense top-100 union (configurable truncation to top 30–50).
12. Implement `scripts/build_indexes.py` to build and persist BM25 + FAISS indexes per dataset into `results/indexes/`.
13. Implement `pipeline/latency.py`: context-manager timers wrapping each component (BM25 search, dense encode+search, feature extraction, router inference, cross-encoder rerank), logging per-query component latencies.
14. Run `scripts/run_retrieval_baselines.py` to produce BM25-only, dense-only, hybrid (swept α, pick best per dataset on training folds), and RRF results, with `evaluation/ranking_metrics.py` (nDCG@10, MRR, MAP, Recall@100).

**Deliverable:** all cheap retrieval baselines + cross-encoder backend working end-to-end with scores, metrics, and per-component latency.

### Phase 2 — Labels & Headroom Checkpoint (Lane B, Weeks 2–3)

15. Implement `labeling/per_route_eval.py`: compute nDCG@10 (and MRR/MAP) per query, per route (BM25, dense, hybrid-best-α) using qrels.
16. Implement `labeling/margin_labeler.py`: best route = argmax nDCG@10; if `top1 - top2 < epsilon`, label = `hybrid`. Tune ε on training folds only.
17. Implement `labeling/regret.py`: regret per route = `best_route_nDCG - this_route_nDCG`, retained alongside labels for regret-weighted training.
18. Implement `scripts/run_oracle_headroom.py`: oracle router = always pick the best route per query; compare oracle nDCG@10 vs. best fixed-α hybrid.
19. **WEEK-2 CHECKPOINT (hard gate):** if oracle headroom over hybrid is < ~2 nDCG points on all chosen datasets, pause and revisit dataset selection (e.g., swap in TREC-COVID or SciDocs) before investing further in router modeling. Document the headroom numbers in `results/metrics/` and in the report.
20. Run `scripts/generate_labels.py` to persist `labels/{dataset}.parquet` (query_id, best_route, regret_bm25, regret_dense, regret_hybrid, margin).

**Deliverable:** labeled dataset per BEIR dataset + go/no-go checkpoint decision documented.

### Phase 3 — Features (Lane C, Weeks 2–3, parallel with Phase 2)

21. Implement `features/query_features.py`: query length, max/mean IDF (computed from the BM25 index's corpus statistics), rare-term density, entity density (simple NER or heuristic), OOV fraction (vs. corpus vocabulary).
22. Implement `features/retrieval_signal_features.py`: BM25/dense top-k Jaccard overlap, rank correlation (Spearman or Kendall) between the two ranked lists, top-1 score (normalized), top1-top2 gap per retriever, score entropy per retriever.
23. Implement `features/feature_builder.py`: orchestrate full feature vector construction per query, with disk caching (features are expensive to recompute — depend on retrieval being run already).
24. Run `scripts/extract_features.py` to persist `features/{dataset}.parquet`, joinable with `labels/{dataset}.parquet` on `query_id`.

**Deliverable:** feature matrix per dataset, ready for router training.

### Phase 4 — Router Training (Lane C, Weeks 3–4)

25. Implement `router/base.py` interface (`fit`, `predict_proba`, `predict`).
26. Implement `router/logistic_router.py`: scikit-learn `LogisticRegression` with `class_weight='balanced'`, trained on features → 3-class label, k-fold CV.
27. Implement `router/gbm_router.py`: LightGBM or XGBoost multi-class classifier, with per-sample weights from regret (regret-weighted loss — misrouting a high-gap query costs more than a near-tie), class weighting, k-fold CV, hyperparameter search on training folds only.
28. Implement `router/escalation.py`: given `predict_proba` output, apply threshold τ — if `max(proba) < τ`, mark query as `ESCALATE`; else route to `argmax(proba)`.
29. Implement `router/oracle_router.py`: upper-bound control that always picks the query's true best route (no escalation).
30. Run `scripts/train_router.py`: train both router types per dataset, save models to `results/models/{dataset}/`.
31. Run `scripts/sweep_escalation.py`: sweep τ to hit target escalation rates (5%, 10%, 15%), log resulting coverage/risk via `evaluation/risk_coverage.py`.

**Deliverable:** trained routers + escalation mechanism + risk-coverage sweep data.

### Phase 5 — End-to-End Pipeline & Controls (Lane A + C, Week 4)

32. Implement `pipeline/end_to_end.py`: query → features → router decision → (cheap route OR escalate-to-cross-encoder) → final ranked list, with latency instrumentation at every step.
33. Implement `pipeline/controls.py`:
    - Random escalation at the same rate as the router (does the router beat random escalation selection?).
    - Always-hybrid + rerank on a random subset of equal size to the router's escalation set.
    - Always-rerank (every query through the cross-encoder).
34. Wire oracle router and fixed-α/RRF baselines into the same end-to-end harness so every method is evaluated identically.

**Deliverable:** single harness producing comparable results for router, all baselines, and all controls.

### Phase 6 — Full Evaluation (Lane B, Week 5)

35. Implement `evaluation/ranking_metrics.py` wrapper (reuse `pytrec_eval`/`ir_measures` rather than hand-rolling nDCG/MRR/MAP/Recall@100).
36. Implement `evaluation/router_metrics.py`: macro-F1, confusion matrix, per-class precision/recall (diagnostic only, reported alongside ranking metrics).
37. Implement `evaluation/significance.py`: paired bootstrap test on per-query nDCG@10 between router and each baseline, per dataset.
38. Implement `evaluation/latency_report.py`: aggregate mean/p95 latency by component, across BM25-only, hybrid, router (cheap path), router (escalated path), always-rerank; compare qualitatively against DAT's reported LLM-call latency.
39. Run `scripts/run_full_evaluation.py` to produce the full results table: per dataset + pooled, nDCG@10/MRR/MAP/Recall@100, significance, and latency — covering H1–H4 directly.
40. Run `scripts/run_latency_benchmark.py` as a dedicated, isolated pass if system load affects timing accuracy in the combined run.

**Deliverable:** complete results tables + significance tests + latency breakdown, mapped explicitly to hypotheses H1–H4.

### Phase 7 — Ablations & Stretch Threads (Weeks 5–6)

41. Run `scripts/run_ablations.py`: query-only features vs. retrieval-signal features vs. both, to attribute router performance to feature groups (Thread 5).
42. Optional/stretch — Thread 2: train a 4-class router (BM25/dense/hybrid/escalate) and compare against confidence-thresholding; implement as an alternate `router/` variant behind a config flag.
43. Optional/stretch — Thread 3: compare margin-based labels vs. regret-weighted-only vs. DAT's top-1 heuristic as alternate labeling strategies in `labeling/`.
44. Optional/stretch — Thread 4: `scripts/run_cross_domain.py` — train router on one BEIR dataset, evaluate on another, report generalization gap.
45. Optional/stretch — Thread 1: compare hand-crafted features against a latent-representation router (e.g., pooled query embedding fed to a small MLP/logistic head) as a differentiation study vs. the MDPI paper.

**Deliverable:** ablation results; any stretch threads attempted, clearly marked as such in the report.

### Phase 8 — Write-up (Week 6–7, all hands)

46. Compile all tables/figures into `report/`. Report every metric honestly, including cases where the router only ties baselines (per proposal's design-decision commitment).
47. Explicitly address each hypothesis H1–H4 with the corresponding evidence table/figure.
48. Document the Week-2 checkpoint outcome, any dataset changes made because of it, and all risks encountered (small query sets, tuning parity between router and fixed-α baseline, latency-vs-BM25 floor).

---

## 3. Testing Strategy

- Unit tests for every retrieval component (`tests/test_bm25.py`, `test_dense.py`, `test_hybrid.py`, `test_rrf.py`, `test_cross_encoder.py`) using a tiny synthetic corpus (3–5 docs) with known expected rankings.
- Unit tests for feature extraction correctness on hand-computed small examples.
- Unit tests for margin labeling edge cases (exact ties, ε boundary).
- Unit tests for escalation threshold logic (`τ` boundary conditions, all-escalate / no-escalate edge cases).
- Integration test (`test_end_to_end_pipeline.py`) running the full pipeline on a tiny fixture dataset to catch wiring bugs before running on real BEIR data.
- All randomized processes (splits, model training, bootstrap significance) must be seeded via `utils/seeding.py` for reproducibility.

## 4. Key Interfaces to Agree On Early (to enable lane parallelism)

- `Candidate` schema (query_id, doc_id, score, rank) — used by retrieval, labeling, and evaluation alike.
- `labels/{dataset}.parquet` schema (query_id, best_route, margin, regret_bm25, regret_dense, regret_hybrid) — contract between Lane B and Lane C.
- `features/{dataset}.parquet` schema (query_id, <feature columns>) — contract between Lane C's feature extraction and router training.
- Config file schemas in `configs/*.yaml` — agreed once in Phase 0 so no lane blocks another.

## 5. Risks to Track (carried over from proposal, with owners)

| Risk | Owner | Mitigation |
|---|---|---|
| Small query sets → noisy significance | Lane B | k-fold CV + paired bootstrap, report confidence intervals, not just point estimates |
| Oracle headroom too small | All | Week-2 checkpoint gate (step 19); documented go/no-go |
| Fixed-α baseline under-tuned vs. router | Lane A | Sweep α on same folds as router training, not a single default value |
| Latency floor near hybrid, not BM25 | Lane A | State this explicitly as the claim in report; don't overclaim |
