.PHONY: help install data index baselines headroom labels features train sweep eval latency ablations test demo clean

VENV ?= ./venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)
DATASETS ?= nfcorpus scifact fiqa

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
	@echo "  demo       - Run the toy smoke test (needs the demo extra)"
	@echo "Override datasets with: make <target> DATASETS=\"scifact\""
	@echo "  clean      - Clean build artifacts and cache"

install:
	pip install -e .[dev,demo]

data:
	$(PYTHON) scripts/download_data.py --datasets $(DATASETS)

index:
	for d in $(DATASETS); do $(PYTHON) scripts/build_indexes.py --dataset $$d || exit 1; done

baselines:
	for d in $(DATASETS); do $(PYTHON) scripts/run_retrieval_baselines.py --dataset $$d || exit 1; done

headroom:
	for d in $(DATASETS); do $(PYTHON) scripts/run_oracle_headroom.py --dataset $$d || exit 1; done

labels:
	for d in $(DATASETS); do $(PYTHON) scripts/generate_labels.py --dataset $$d || exit 1; done

features:
	for d in $(DATASETS); do $(PYTHON) scripts/extract_features.py --dataset $$d || exit 1; done

train:
	for d in $(DATASETS); do $(PYTHON) scripts/train_router.py --dataset $$d || exit 1; done

sweep:
	for d in $(DATASETS); do $(PYTHON) scripts/sweep_escalation.py --dataset $$d || exit 1; done

eval:
	$(PYTHON) scripts/run_full_evaluation.py --datasets $(DATASETS)

latency:
	for d in $(DATASETS); do $(PYTHON) scripts/run_latency_benchmark.py --dataset $$d || exit 1; done

ablations:
	for d in $(DATASETS); do $(PYTHON) scripts/run_ablations.py --dataset $$d || exit 1; done

test:
	$(PYTHON) -m pytest tests/

demo:
	$(PYTHON) scripts/demo_synthetic_routing.py

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
