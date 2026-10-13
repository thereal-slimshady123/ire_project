# Confidence-Based Query Router for Hybrid Retrieval

An empirical information retrieval framework that routes each query to sparse (BM25), dense, or fixed-α hybrid retrieval with a 3-class classifier, and escalates queries where the classifier's confidence is below τ to a cross-encoder reranker.

**Status:** skeleton. Interfaces, configs and test placeholders are in place; most modules still raise `NotImplementedError` and tests are skipped until implemented.

## Project Structure

- [plan.md](plan.md) — binding design contracts (§0) and the step-by-step plan.
- [proposal.md](proposal.md) — research proposal and hypotheses H1–H4.
- [improvements.md](improvements.md) — hybrid-degradation analysis (§1, part of the main project) and stretch threads (§2 advantage regression, §3 synthetic queries).

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

# Install package in editable mode with dev dependencies (+ rank-bm25 for the demo)
pip install -e .[dev,demo]
```

## Protocol at a Glance

- Datasets: NFCorpus, SciFact, FiQA (TREC-COVID optional, test-only).
- Splits: train the router and tune α\* on BEIR **train**; select ε, hyperparameters and τ on **dev** (SciFact: seeded 20% of train); report once on **test**.
- Retrievers: BM25 (`bm25s`, stemmed), dense (TAS-B, dot product), hybrid over the exactly-scored union of both top-100 lists.
- Escalation: cross-encoder reranks the top 50 of the hybrid-ordered union; 100 results returned.
- Metrics: nDCG@10, MRR@10, MAP@100, Recall@100; paired bootstrap with Holm-Bonferroni.

## Workflow Overview

Per-dataset scripts take `--dataset`; the Makefile loops over `DATASETS` (default `nfcorpus scifact fiqa`), e.g. `make index DATASETS="scifact"`.

1. **Download Data**: `make data`
2. **Build Indexes**: `make index`
3. **Retrieval Baselines** (also selects α\* on train): `make baselines`
4. **Oracle Headroom** (Week-2 gate, train + dev): `make headroom`
5. **Generate Labels**: `make labels`
6. **Extract Features**: `make features`
7. **Train Router** (train; select on dev; calibrate τ on dev): `make train`
8. **Sweep Escalation**: `make sweep`
9. **Full Evaluation** (test): `make eval`
10. **Latency Benchmark**: `make latency`
11. **Feature Ablations**: `make ablations`
12. **Run Tests**: `make test`
13. **Toy Smoke Test** (hand-written queries; not evidence): `make demo`
