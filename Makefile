.PHONY: help install data index baselines headroom labels features train sweep escalation rerankers policy-latency eval latency ablations test clean

PYTHON = python

help:
	@echo "Available targets:"
	@echo "  install    - Install package and dependencies"
	@echo "  data       - Download BEIR datasets"
	@echo "  index      - Build BM25 and FAISS indexes"
	@echo "  baselines  - Run cheap retrieval baselines"
	@echo "  headroom   - Run oracle headroom checkpoint"
	@echo "  labels     - Generate route labels"
	@echo "  features   - Extract query and retrieval features"
	@echo "  train      - Train router models"
	@echo "  sweep      - Sweep escalation threshold tau"
	@echo "  escalation - Experiment 4 + router/tau escalation (needs Experiment 2 artifacts)"
	@echo "  rerankers  - Matched-pool cross-encoder checkpoint benchmark"
	@echo "  policy-latency - Measured per-policy latency matched to held-out quality"
	@echo "  eval       - Combined held-out evaluation + report figures"
	@echo "  latency    - Run latency benchmark"
	@echo "  ablations  - Run feature ablation studies"
	@echo "  test       - Run unit and integration tests"
	@echo "  clean      - Clean build artifacts and cache"

install:
	pip install -e .[dev]

data:
	$(PYTHON) scripts/download_data.py

index:
	$(PYTHON) scripts/build_indexes.py

baselines:
	$(PYTHON) scripts/run_retrieval_baselines.py

headroom:
	$(PYTHON) scripts/run_oracle_headroom.py

labels:
	$(PYTHON) scripts/generate_labels.py

features:
	$(PYTHON) scripts/extract_features.py

train:
	$(PYTHON) scripts/train_router.py

sweep:
	$(PYTHON) scripts/sweep_escalation.py

escalation:
	$(PYTHON) scripts/run_experiment4_escalation.py
	$(PYTHON) scripts/run_tau_escalation.py

rerankers:
	$(PYTHON) scripts/benchmark_cross_encoders.py

policy-latency:
	$(PYTHON) scripts/run_policy_latency.py

eval:
	$(PYTHON) scripts/run_full_evaluation.py --reranker_benchmark_summary results/reranker_benchmark/summary.json
	$(PYTHON) scripts/make_report_figures.py

latency:
	$(PYTHON) scripts/run_latency_benchmark.py

ablations:
	$(PYTHON) scripts/run_ablations.py

test:
	pytest tests/

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
