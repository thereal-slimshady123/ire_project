# Confidence-Based Query Router for Hybrid Retrieval

An empirical information retrieval framework that dynamically routes queries between sparse (BM25), dense, and fixed-convex hybrid retrievers, escalating low-confidence queries to a cross-encoder reranker.

**Write-ups:** Phase 1 in [report/report.md](report/report.md) (§6 summarises Phase 2), mid-submission in
[report/midsubmission.md](report/midsubmission.md), Phase 2 in full in [report/phase2_report.md](report/phase2_report.md).
[plan.md](plan.md) and [proposal.md](proposal.md) are the original plans (historical); [improvements.md](improvements.md)
holds Experiment 1 and the label/synthetic designs (its pre-registered §3–4 survive in `results/preregistration/`).

## Project Structure

```
ire_project/
├── configs/            # Experiment and component configuration files (hash-pinned by the Phase 2 pre-registrations)
├── data/               # BEIR corpora, queries, and qrels (gitignored); data/processed/natural-niche/ = Phase 2 exploration diagnostics
├── notebooks/          # Exploratory analysis notebooks
├── report/             # Reports, figures, and tables
├── results/            # Per-experiment outputs; results/preregistration/ = hashed pre-registrations (do not edit)
├── scripts/            # Executable CLI workflows
├── src/                # Core Python package (`router_retrieval`)
└── tests/              # Test suite
```

## Setup

```bash
# Clone and create virtual environment
python -m venv venv
source venv/bin/activate  # Or on Windows: .\venv\Scripts\activate

# Install package in editable mode with dev dependencies
pip install -e .[dev]
```

## Phase 1 workflow

1. **Download Data**: `python scripts/download_data.py`
2. **Build Indexes**: `python scripts/build_indexes.py`
3. **Retrieval Baselines**: `python scripts/run_retrieval_baselines.py`
4. **Experiments 1–3**: `python scripts/run_experiment1_hybrid_degradation.py`, `run_experiment1b_query_types.py`, `run_experiment2_heldout_routing.py`, `run_experiment3_skip_dense.py`
5. **Oracle Headroom** (all datasets, noise-corrected): `python scripts/run_oracle_headroom.py`
6. **Route Labels** (training-fold α*/ε): `python scripts/generate_labels.py`
7. **Escalation**: `python scripts/run_experiment4_escalation.py` (pre-registered), then `python scripts/run_tau_escalation.py` (router + τ escalation, risk-coverage)
8. **Reranker Benchmark**: `python scripts/benchmark_cross_encoders.py`
9. **Measured Policy Latency**: `python scripts/run_policy_latency.py` (real GPU timings). `scripts/run_latency_benchmark.py` and `results/latency/` are a deprecated *simulated* benchmark; do not cite them.
10. **Full Evaluation + Figures**: `python scripts/run_full_evaluation.py --reranker_benchmark_summary results/reranker_benchmark/summary.json && python scripts/make_report_figures.py`
11. **Synthetic Routing Demo** (60-doc pilot): `python scripts/demo_synthetic_routing.py`
12. **Track 2 — features and advantage routing** (all run offline from the committed `results/experiment2/cache/`):
    `python scripts/extract_features.py` (Contract 2 parquet; `--source runs` recomputes from raw data),
    `python scripts/train_router.py` (Contract 3 `results/models/{dataset}/predictions.parquet`),
    `python scripts/run_experiment2b_advantage.py`, `python scripts/run_ablations.py`, `python scripts/run_cross_domain.py`,
    later variants `run_experiment2c_embedding_router.py`, `run_experiment2d_cost_aware_router.py` (the final router),
    `run_experiment2e_ce_feature_router.py`, `run_experiment4b_interpolated_rerank.py`, `run_experiment4c_topk_blended_rerank.py`
13. **Synthetic scaling** (needs `data/raw`, torch, GPU; not run at scale): `python scripts/generate_synthetic_queries.py --n_total 30000`, then
    `python scripts/run_experiment2b_advantage.py --synthetic_dir results/synthetic --n_synthetic 0 1000 3000 10000 30000`.
    This writes to `results/synthetic/`; the 288-query set used by Phase 2's E1 lives separately in
    `results/phase2_validation/synthetic/` so it cannot be overwritten.

## Phase 2 workflow

Every confirmatory step has a pre-registration in `results/preregistration/` (with `SHA256SUMS.txt` and `*.pins.txt`;
check with `sha256sum -c results/preregistration/SHA256SUMS.txt`). Names follow `report/phase2_report.md` (E1–E8, E4-R);
note that the first pre-registration calls the skip-dense test "E4" (= E3/E3′ in the report).

1. **Exploration datasets**: `python scripts/build_niche_datasets.py`; diagnostics `scripts/diagnose_router_lock.py`,
   `scripts/analyze_niche_existence.py`, `scripts/build_natural_niche.py` → `data/processed/natural-niche/`.
2. **E1–E2 validation** (CQADupStack mathematica/english/gaming): `run_experiment2_heldout_routing.py` and
   `run_experiment2d_cost_aware_router.py` on those datasets, then
   `python scripts/run_phase2_validation.py --synthetic_dir results/phase2_validation/synthetic`
   → `results/phase2_validation/validation_results.md`.
3. **E3′ matched skip rate** (physics/programmers/gis): `scripts/build_e4_fresh_datasets.py`, `scripts/run_e4_matched_rate.py` → `results/e4_fresh/`.
4. **Strengthening (E2 replications, E4–E7, learning curve)**: `bash scripts/run_strengthening_pipeline.sh` → `results/strengthening/fresh/`
   (exploration outputs in `results/strengthening/explore/`). Learned-fusion latency: `scripts/run_fusion_latency.py`.
5. **E8 boundary test**: `bash scripts/run_boundary_pipeline.sh` → `results/strengthening/boundary/`.
6. **E4-R (Climate-FEVER)**: `bash scripts/run_e4r_pipeline.sh` → `results/strengthening/e4r/`.
7. **Post hoc cost table without the laptop-suspend outlier**:
   `python scripts/posthoc_cost_frontier_clean.py --timings results/strengthening/fresh/cost_frontier/timings_cpu.csv --output_dir results/strengthening/fresh/cost_frontier`.

Scripts, `configs/` and `src/` listed in the `*.pins.txt` files are hash-pinned: do not edit them; add new, clearly
labelled post hoc scripts instead.

## Tests

`pytest tests/`

Results: `results/*/tables.md`, `results/metrics/combined.md`, `results/strengthening/**/tables.md`.
