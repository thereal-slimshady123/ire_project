.PHONY: help install data index baselines headroom labels features train sweep eval latency ablations test clean

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
	@echo "  eval       - Run full pipeline evaluation"
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

eval:
	$(PYTHON) scripts/run_full_evaluation.py

latency:
	$(PYTHON) scripts/run_latency_benchmark.py

ablations:
	$(PYTHON) scripts/run_ablations.py

test:
	pytest tests/

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
