# Implementation Plan: Confidence-Based Query Router for Hybrid Retrieval

This document translates `proposal.md` into a concrete, step-by-step engineering plan with a full repository layout. It is written so that any contributor (human or coding agent) can pick up a module, understand its inputs/outputs/contracts, and implement it without needing to re-derive design decisions already made in the proposal.

---

## 0. Guiding Contracts (read first)

These are fixed decisions from the proposal that every module must respect. Do not re-litigate them without updating this file.

1. **Datasets:** BEIR subsets only, chosen for feasibility: NFCorpus, SciFact, FiQA (primary); TREC-COVID optional/stretch (test-only, so usable only for cross-domain evaluation). SciDocs optional.
2. **Splits (train → dev → test):** the router is trained and α\* is chosen on the official BEIR **train** queries; ε, router hyperparameters and τ are selected on **dev**; results are reported once on the untouched **test** queries. SciFact has no dev qrels, so a seeded 20% of its train queries is held out as dev. No k-fold cross-validation. (`configs/datasets.yaml`)
3. **Retrievers (cheap routes):**
   - BM25: `bm25s` with English stemming and stopword removal (k1=1.2, b=0.75).
   - Dense: `sentence-transformers/msmarco-distilbert-base-tas-b`, dot-product similarity (embeddings **not** normalized), exact FAISS `IndexFlatIP`.
   - Hybrid = fixed-α convex combination `α·norm_dense + (1−α)·norm_bm25` over the **union** of BM25 top-100 and dense top-100. For every union document both raw scores are computed exactly (no imputation), then min-max normalized per query over the union. α\* is chosen on train.
   - Document text everywhere (BM25, dense, cross-encoder) = `title + " " + text`.
4. **Escalation backend:** `cross-encoder/ms-marco-MiniLM-L-6-v2`. Pool = BM25 ∪ dense top-100 ordered by hybrid(α\*) score; the top **50** are reranked, the remaining hybrid-ordered documents are appended, and **100** results are returned.
5. **Router:** multi-class classifier over 3 cheap routes (BM25 / dense / hybrid). Escalation is **not** a 4th class: a query is escalated when `max(class_probability) < τ`. τ is calibrated on **dev** probabilities to hit each target escalation rate (5%, 10%, 15%); training-set probabilities are overconfident and must not be used. (Advantage regression and a 4th-class variant are documented future-work threads, see Phase 7 / `improvements.md`.)
6. **Labels:** per-query argmax over {BM25, dense, hybrid(α\*)} nDCG@10, with margin threshold ε: if `top1 − top2 < ε` (including exact ties and all-zero queries), label = hybrid (safe default). ε grid {0.02, 0.05, 0.10, 0.15}, selected on dev by routed nDCG@10. Raw per-route nDCG and regret are retained.
7. **Models and weighting:** Logistic Regression (baseline router, standardized features) and LightGBM (main router). Both use one sample weight: `class_balance_weight(label) × (margin + 0.05)`, rescaled to mean 1 (`labeling/regret.py::compute_sample_weights`). Routers add no extra `class_weight`.
8. **Baselines/controls:** BM25-only, dense-only, fixed-α hybrid (α\* from train), RRF (k=60), always-rerank, oracle router, **oracle escalation** (ceiling: with routing fixed, escalate the B% of queries with the largest true cross-encoder gain — evaluation-only, no model), random escalation (router's routes kept, exactly the router's number of escalated queries chosen at random), always-hybrid + random rerank (same count).
9. **Metrics:** every run is evaluated at depth 100. nDCG@10, MRR@10, MAP@100, Recall@100 (primary); macro-F1 + confusion matrix (router diagnostic, secondary); latency mean/p95 per component. Runs are passed to pytrec_eval with score = 1/rank.
10. **Validation:** paired bootstrap (10,000 resamples) on per-query test nDCG@10, router vs. every non-oracle system, with Holm-Bonferroni correction per dataset. No significance test against the oracle (oracle ≥ hybrid on every query by construction).
11. **Week-2 checkpoint gate (train + dev queries only):** headroom = oracle{BM25, dense, hybrid(α\*)} − hybrid(α\*); noise floor = oracle{hybrid(α\*), hybrid(α\*±0.1)} − hybrid(α\*). If `headroom − noise floor` < ~2 nDCG points, revisit dataset choice before investing further in router modeling.
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
├── pyproject.toml                     # single source of truth for dependencies
├── requirements.txt                   # `-e .[dev,demo]`
├── .gitignore                         # data/, *.index, *.pt, wandb/, .cache/, __pycache__/
├── Makefile                           # convenience targets (loop over DATASETS): make data, make baselines, make train, make eval
├── configs/
│   ├── datasets.yaml                  # dataset names, BEIR URLs, train/dev/test split roles
│   ├── retrievers.yaml                # bm25s params (k1, b, stemmer), TAS-B dense model, FAISS index type
│   ├── hybrid.yaml                    # alpha grid, normalization method, alpha* selection on train
│   ├── cross_encoder.yaml             # model name, batch size, rerank_depth 50, output_depth 100
│   ├── labels.yaml                    # epsilon grid (selected on dev), hybrid fallback
│   ├── router.yaml                    # feature list, model hyperparams, sample weighting
│   ├── escalation.yaml                # target escalation rates (5/10/15%), tau calibrated on dev
│   └── experiment.yaml                # top-level: datasets, seed, split protocol, depth, metrics
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
│       │   └── splits.py              # train/dev/test roles; seeded dev carve-out for SciFact
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── bm25.py                # BM25Retriever (bm25s): index(), search(), score_docs(), idf_vocab()
│       │   ├── dense.py               # DenseRetriever (TAS-B, dot product): index(), search(), score_docs()
│       │   ├── normalization.py       # min-max / z-score normalization of BM25 & dense scores
│       │   ├── hybrid.py              # HybridRetriever(alpha): convex combination over the exactly-scored union
│       │   ├── rrf.py                 # ReciprocalRankFusion: rank-based fusion baseline
│       │   └── cross_encoder.py       # CrossEncoderReranker: rerank top 50 of hybrid-ordered union, return 100
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
│       │   └── regret.py              # regret = best_nDCG - route_nDCG; compute_sample_weights()
│       │
│       ├── router/
│       │   ├── __init__.py
│       │   ├── base.py                # RouterModel interface: fit(), predict_proba(), predict()
│       │   ├── logistic_router.py     # sklearn LogisticRegression wrapper, weighted via sample_weight
│       │   ├── gbm_router.py          # LightGBM wrapper, weighted via sample_weight
│       │   ├── escalation.py          # apply tau to predict_proba -> route or ESCALATE; calibrate tau on dev
│       │   └── oracle_router.py       # oracle router + oracle_escalation (evaluation-only upper bounds)
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── end_to_end.py          # given a query -> route decision -> final ranked list
│       │   ├── controls.py            # set-level policies: random escalation (router routes kept),
│       │   │                           # always-hybrid+random-rerank, always-rerank; assemble_run()
│       │   └── latency.py             # per-component timers (BM25, dense, feature extraction,
│       │                               # router inference, cross-encoder), context-manager based
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ranking_metrics.py     # nDCG@10, MRR@10, MAP@100, Recall@100 (pytrec_eval, score = 1/rank)
│       │   ├── router_metrics.py      # macro-F1, confusion matrix, per-class precision/recall
│       │   ├── significance.py        # paired bootstrap test on per-query nDCG@10 + Holm-Bonferroni
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
│   ├── run_oracle_headroom.py         # WEEK-2 CHECKPOINT: oracle headroom minus noise floor (train+dev)
│   ├── generate_labels.py             # per-route nDCG + margin labeling -> labels/{dataset}.parquet
│   ├── extract_features.py            # build feature matrix -> features/{dataset}.parquet
│   ├── train_router.py                # train logistic + GBM routers, save models/
│   ├── sweep_escalation.py            # risk-coverage + nDCG vs escalation rate (tau from dev)
│   ├── run_full_evaluation.py         # run router + all baselines/controls, compute all metrics
│   ├── run_latency_benchmark.py       # dedicated latency measurement pass
│   ├── run_ablations.py               # feature ablation study (query-only vs retrieval-signal vs both)
│   ├── run_cross_domain.py            # stretch: train on dataset X, test on dataset Y
│   └── demo_synthetic_routing.py      # toy smoke test only (hand-written queries; not evidence)
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
│   ├── runs/{dataset}/                # candidate rankings per route/split (bm25, dense, hybrid, cross_encoder)
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

1. Initialize repo structure above; declare dependencies (minimum versions) in `pyproject.toml`: `bm25s` + `PyStemmer`, `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `lightgbm`, `pytrec-eval-terrier`, `pandas`, `numpy`, `pyarrow`, `pyyaml`, `pytest`. `requirements.txt` just installs `-e .[dev,demo]`.
2. Write `src/router_retrieval/data/schema.py` — shared dataclasses (`Document`, `Query`, `QRel`, `Candidate`) used by every downstream module so retrieval, labeling, and features all speak the same format.
3. Write `scripts/download_data.py` to pull NFCorpus, SciFact, FiQA (and optionally TREC-COVID) via the BEIR dataset distribution; cache to `data/raw/{dataset}/`.
4. Write `src/router_retrieval/data/beir_loader.py` to parse each dataset into the common schema, and `src/router_retrieval/data/splits.py` to resolve train/dev/test query IDs per dataset (seeded 20% dev carve-out from SciFact train).
5. `configs/*.yaml` defaults are fixed by §0 (bm25s k1=1.2/b=0.75, TAS-B, `cross-encoder/ms-marco-MiniLM-L-6-v2`, alpha grid, epsilon grid, target escalation rates).

**Deliverable:** datasets loadable end-to-end into the common schema; configs agreed; repo skeleton committed.

### Phase 1 — Retrieval Infra (Lane A, Weeks 1–2)

6. Implement `retrieval/bm25.py` with `bm25s` (stemming + stopwords): index corpus, `search(query, k)`, `score_docs()` for exact scores of arbitrary docs, `idf_vocab()`. Sanity check against published BEIR BM25 nDCG@10 (SciFact ≈0.665, NFCorpus ≈0.325, FiQA ≈0.236).
7. Implement `retrieval/dense.py`: encode corpus with TAS-B (dot product, no normalization; GPU for encoding), build a FAISS `IndexFlatIP`, `search(query, k)`, `score_docs()`.
8. Implement `retrieval/normalization.py`: min-max and/or z-score normalization for BM25 and dense scores (BM25 is unbounded, dense is bounded — normalize before any fusion, per Pinecone reference in the proposal).
9. Implement `retrieval/hybrid.py`: `HybridRetriever(alpha)` over the union of both top-100 lists, scoring missing documents exactly, min-max normalizing over the union; implement the alpha sweep in `scripts/run_retrieval_baselines.py`.
10. Implement `retrieval/rrf.py`: standard reciprocal rank fusion.
11. Implement `retrieval/cross_encoder.py`: load MiniLM cross-encoder; rerank the top 50 of the hybrid-ordered BM25∪dense union, append the rest in hybrid order, return 100 with strictly decreasing scores.
12. Implement `scripts/build_indexes.py` to build and persist BM25 + FAISS indexes per dataset into `results/indexes/`.
13. Implement `pipeline/latency.py`: context-manager timers wrapping each component (BM25 search, dense encode+search, feature extraction, router inference, cross-encoder rerank), logging per-query component latencies.
14. Run `scripts/run_retrieval_baselines.py` on train, dev and test queries to produce BM25-only, dense-only, hybrid (swept α, α\* = best mean nDCG@10 on **train**), and RRF results, with `evaluation/ranking_metrics.py` (nDCG@10, MRR@10, MAP@100, Recall@100).

**Deliverable:** all cheap retrieval baselines + cross-encoder backend working end-to-end with scores, metrics, and per-component latency.

### Phase 2 — Labels & Headroom Checkpoint (Lane B, Weeks 2–3)

15. Implement `labeling/per_route_eval.py`: compute nDCG@10 (and MRR@10/MAP@100/Recall@100) per query, per route (BM25, dense, hybrid at α\* from train) using qrels.
16. Implement `labeling/margin_labeler.py`: best route = argmax nDCG@10; if `top1 - top2 < epsilon` (ties included), label = `hybrid`. ε is selected on **dev** in `scripts/train_router.py` (one router per ε, keep the best routed nDCG@10).
17. Implement `labeling/regret.py`: regret per route = `best_route_nDCG - this_route_nDCG`, plus `compute_sample_weights()` (class balance × (margin + 0.05), mean 1).
18. Implement `scripts/run_oracle_headroom.py` on **train + dev** queries: headroom = oracle{BM25, dense, hybrid(α\*)} − hybrid(α\*); noise floor = oracle{hybrid(α\*), hybrid(α\*±0.1)} − hybrid(α\*). No significance test (oracle ≥ hybrid per query by construction).
19. **WEEK-2 CHECKPOINT (hard gate):** if headroom minus noise floor is < ~2 nDCG points on all chosen datasets, pause and revisit dataset selection (e.g., swap in TREC-COVID or SciDocs) before investing further in router modeling. Document the headroom numbers in `results/metrics/` and in the report.
20. Run `scripts/generate_labels.py` to persist `labels/{dataset}.parquet` (schema in §4) for train, dev and test queries; test labels are used for analysis only.

**Deliverable:** labeled dataset per BEIR dataset + go/no-go checkpoint decision documented.

### Phase 3 — Features (Lane C, Weeks 2–3, parallel with Phase 2)

21. Implement `features/query_features.py`: query length, max/mean IDF (computed from the BM25 index's corpus statistics), rare-term density, entity density (simple NER or heuristic), OOV fraction (vs. corpus vocabulary).
22. Implement `features/retrieval_signal_features.py`: BM25/dense top-10/top-20 Jaccard overlap, Spearman rank correlation over documents shared by both top-20 lists (0.0 if fewer than 2), raw top-1 score, top1-top2 gap per retriever, score entropy per retriever.
23. Implement `features/feature_builder.py`: orchestrate full feature vector construction per query, with disk caching (features are expensive to recompute — depend on retrieval being run already).
24. Run `scripts/extract_features.py` for train, dev and test queries to persist `features/{dataset}.parquet`, joinable with `labels/{dataset}.parquet` on `query_id`.

**Deliverable:** feature matrix per dataset, ready for router training.

### Phase 4 — Router Training (Lane C, Weeks 3–4)

25. Implement `router/base.py` interface (`fit`, `predict_proba`, `predict`).
26. Implement `router/logistic_router.py`: scikit-learn `LogisticRegression` on standardized features → 3-class label, weighted by `compute_sample_weights`, trained on train.
27. Implement `router/gbm_router.py`: LightGBM multi-class classifier weighted by `compute_sample_weights` (misrouting a high-gap query costs more than a near-tie), trained on train, hyperparameters selected on dev.
28. Implement `router/escalation.py`: given `predict_proba` output, apply threshold τ — if `max(proba) < τ`, mark query as `ESCALATE`; else route to `argmax(proba)`. `find_tau_for_target_rate` calibrates τ on **dev** probabilities.
29. Implement `router/oracle_router.py`: oracle router (always the true best cheap route, no escalation) and `oracle_escalation` (routing fixed, escalate the B% of queries with the largest true cross-encoder gain).
30. Run `scripts/train_router.py`: train both router types per dataset on train, select ε and hyperparameters on dev, calibrate τ on dev for 5/10/15%, save models and τ values to `results/models/{dataset}/`.
31. Run `scripts/sweep_escalation.py`: risk-coverage curve (`evaluation/risk_coverage.py`) and routed nDCG@10 vs. escalation rate on test, with τ from dev.

**Deliverable:** trained routers + escalation mechanism + risk-coverage sweep data.

### Phase 5 — End-to-End Pipeline & Controls (Lane A + C, Week 4)

32. Implement `pipeline/end_to_end.py`: query → features → router decision → (cheap route OR escalate-to-cross-encoder) → final ranked list, with latency instrumentation at every step.
33. Implement `pipeline/controls.py` (set-level policies over precomputed runs; "same rate" = exactly the router's number of escalated queries):
    - Random escalation: router's routes kept, the escalated queries chosen at random (does the router pick better queries to escalate than chance?).
    - Always-hybrid + random rerank: no routing, same number of random queries reranked (does routing add anything beyond the reranking budget?).
    - Always-rerank (every query through the cross-encoder).
    - `assemble_run()` builds a final depth-100 run from per-query decisions.
34. Wire oracle router, oracle escalation and fixed-α/RRF baselines into the same harness so every method is evaluated identically.

**Deliverable:** single harness producing comparable results for router, all baselines, and all controls.

### Phase 6 — Full Evaluation (Lane B, Week 5)

35. Implement `evaluation/ranking_metrics.py` wrapper over `pytrec_eval` (nDCG@10, MRR@10, MAP@100, Recall@100; runs converted with score = 1/rank).
36. Implement `evaluation/router_metrics.py`: macro-F1, confusion matrix, per-class precision/recall (diagnostic only, reported alongside ranking metrics).
37. Implement `evaluation/significance.py`: paired bootstrap test on per-query test nDCG@10 between router and each non-oracle system, with Holm-Bonferroni correction per dataset.
38. Implement `evaluation/latency_report.py`: aggregate mean/p95 latency by component, across BM25-only, hybrid, router (cheap path), router (escalated path), always-rerank; compare qualitatively against DAT's reported LLM-call latency.
39. Run `scripts/run_full_evaluation.py` on **test** queries to produce the full results table: per dataset + pooled, nDCG@10/MRR@10/MAP@100/Recall@100, significance, oracle-escalation ceiling, and latency — covering H1–H4 directly.
40. Run `scripts/run_latency_benchmark.py` as a dedicated, isolated pass if system load affects timing accuracy in the combined run.

**Deliverable:** complete results tables + significance tests + latency breakdown, mapped explicitly to hypotheses H1–H4.

### Phase 7 — Ablations & Stretch Threads (Weeks 5–6)

41. Run `scripts/run_ablations.py`: query-only features vs. retrieval-signal features vs. both, to attribute router performance to feature groups (Thread 5).
42. Optional/stretch — Thread 2: train a 4-class router (BM25/dense/hybrid/escalate) and compare against confidence-thresholding; implement as an alternate `router/` variant behind a config flag.
43. Optional/stretch — Thread 3: compare margin-based labels vs. regret-weighted-only vs. DAT's top-1 heuristic as alternate labeling strategies in `labeling/`.
44. Optional/stretch — Thread 4: `scripts/run_cross_domain.py` — train router on one BEIR dataset, evaluate on another, report generalization gap.
45. Optional/stretch — Thread 1: compare hand-crafted features against a latent-representation router (e.g., pooled query embedding fed to a small MLP/logistic head) as a differentiation study vs. the MDPI paper.
46. Optional/stretch — Thread 6: advantage regression (predict Δ_BM25, Δ_Dense vs. hybrid) and synthetic training queries, as designed in `improvements.md` §2–3. Not part of the main method.

**Deliverable:** ablation results; any stretch threads attempted, clearly marked as such in the report.

### Phase 8 — Write-up (Week 6–7, all hands)

47. Compile all tables/figures into `report/`. Report every metric honestly, including cases where the router only ties baselines (per proposal's design-decision commitment).
48. Explicitly address each hypothesis H1–H4 with the corresponding evidence table/figure.
49. Document the Week-2 checkpoint outcome, any dataset changes made because of it, and all risks encountered (small query sets, tuning parity between router and fixed-α baseline, latency-vs-BM25 floor).

---

## 3. Testing Strategy

- Unit tests for every retrieval component (`tests/test_bm25.py`, `test_dense.py`, `test_hybrid.py`, `test_rrf.py`, `test_cross_encoder.py`) using a tiny synthetic corpus (3–5 docs) with known expected rankings.
- Unit tests for feature extraction correctness on hand-computed small examples.
- Unit tests for margin labeling edge cases (exact ties, ε boundary).
- Unit tests for escalation threshold logic (`τ` boundary conditions, all-escalate / no-escalate edge cases, τ calibration hitting the target rate, oracle escalation).
- Unit tests for controls (exact escalated counts; random escalation keeps router routes).
- Test placeholders are module-level `pytest.mark.skip` until their module is implemented; remove the marker when implementing.
- Integration test (`test_end_to_end_pipeline.py`) running the full pipeline on a tiny fixture dataset to catch wiring bugs before running on real BEIR data.
- All randomized processes (splits, model training, bootstrap significance) must be seeded via `utils/seeding.py` for reproducibility.

## 4. Key Interfaces to Agree On Early (to enable lane parallelism)

- `Candidate` schema (query_id, doc_id, score, rank) — used by retrieval, labeling, and evaluation alike.
- `labels/{dataset}.parquet` schema (query_id, split, epsilon, best_route, margin, top1_score, top2_score, ndcg_bm25, ndcg_dense, ndcg_hybrid, regret_bm25, regret_dense, regret_hybrid) — one row per query per ε; contract between Lane B and Lane C.
- `features/{dataset}.parquet` schema (query_id, split, <feature columns from configs/router.yaml>) — contract between Lane C's feature extraction and router training.
- Config file schemas in `configs/*.yaml` — agreed once in Phase 0 so no lane blocks another.

## 5. Risks to Track (carried over from proposal, with owners)

| Risk | Owner | Mitigation |
|---|---|---|
| Small test sets → noisy significance | Lane B | Train on BEIR train splits, paired bootstrap on test with Holm-Bonferroni, report confidence intervals, not just point estimates |
| Oracle headroom too small (or only noise) | All | Week-2 checkpoint gate on headroom minus noise floor (step 19); documented go/no-go |
| Fixed-α baseline under-tuned vs. router | Lane A | Sweep α on the same train split the router uses, not a single default value |
| Latency floor near hybrid, not BM25 | Lane A | State this explicitly as the claim in report; don't overclaim |
| Confidence-based escalation misses queries where every cheap route fails (they tie, get labelled hybrid, may be confident) | Lane C | Report the oracle-escalation ceiling next to the router; analyse the gap in the report |
