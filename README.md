# Confidence-Based Query Router for Hybrid Retrieval

An empirical information retrieval framework that dynamically routes queries between sparse (BM25), dense, and fixed-convex hybrid retrievers, escalating low-confidence queries to a cross-encoder reranker.

## Project Structure

Refer to [plan.md](plan.md), [proposal.md](proposal.md), and [improvements.md](improvements.md) for architectural contracts, experimental design, and synthetic scaling protocols.

```
ire_project/
├── configs/            # Experiment and component configuration files
├── data/               # BEIR corpora, queries, and qrels (gitignored)
├── notebooks/          # Exploratory analysis notebooks
├── report/             # Final report, figures, and tables
├── results/            # Checkpoints, generated models, labels, metrics
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

## Workflow Overview

1. **Download Data**: `python scripts/download_data.py`
2. **Build Indexes**: `python scripts/build_indexes.py`
3. **Retrieval Baselines**: `python scripts/run_retrieval_baselines.py`
4. **Experiments 1–3**: `python scripts/run_experiment1_hybrid_degradation.py`, `run_experiment1b_query_types.py`, `run_experiment2_heldout_routing.py`, `run_experiment3_skip_dense.py`
5. **Oracle Headroom** (all datasets, noise-corrected): `python scripts/run_oracle_headroom.py`
6. **Route Labels** (training-fold α*/ε): `python scripts/generate_labels.py`
7. **Escalation**: `python scripts/run_experiment4_escalation.py` (pre-registered), then `python scripts/run_tau_escalation.py` (router + τ escalation, risk-coverage)
8. **Reranker Benchmark**: `python scripts/benchmark_cross_encoders.py`
9. **Measured Policy Latency**: `python scripts/run_policy_latency.py`
10. **Full Evaluation + Figures**: `python scripts/run_full_evaluation.py --reranker_benchmark_summary results/reranker_benchmark/summary.json && python scripts/make_report_figures.py`
11. **Synthetic Routing Demo** (60-doc pilot): `python scripts/demo_synthetic_routing.py`
12. **Track 2 — features and advantage routing** (all run offline from the committed `results/experiment2/cache/`):
    `python scripts/extract_features.py` (Contract 2 parquet; `--source runs` recomputes from raw data),
    `python scripts/train_router.py` (Contract 3 `results/models/{dataset}/predictions.parquet`),
    `python scripts/run_experiment2b_advantage.py`, `python scripts/run_ablations.py`, `python scripts/run_cross_domain.py`
13. **Synthetic scaling** (needs `data/raw`, torch, GPU): `python scripts/generate_synthetic_queries.py --n_total 30000`, then
    `python scripts/run_experiment2b_advantage.py --synthetic_dir results/synthetic --n_synthetic 0 1000 3000 10000 30000`
14. **Run Tests**: `pytest tests/`

Results and the write-up: `results/*/tables.md`, `results/metrics/combined.md`, `report/report.md`.
