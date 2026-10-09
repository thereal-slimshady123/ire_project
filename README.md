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
4. **Oracle Headroom**: `python scripts/run_oracle_headroom.py`
5. **Generate Labels**: `python scripts/generate_labels.py`
6. **Extract Features**: `python scripts/extract_features.py`
7. **Train Router**: `python scripts/train_router.py`
8. **Sweep Escalation**: `python scripts/sweep_escalation.py`
9. **Full Evaluation**: `python scripts/run_full_evaluation.py`
10. **Synthetic Routing Demo**: `python scripts/demo_synthetic_routing.py`
11. **Run Tests**: `pytest tests/`
