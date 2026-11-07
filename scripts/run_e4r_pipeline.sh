#!/usr/bin/env bash
# Fixed pipeline for results/preregistration/e4r_climate_fever_20261107.md. Run once, from the project root:
#   bash scripts/run_e4r_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
R=results/strengthening/e4r
mkdir -p "$R/logs"
run() { local name=$1; shift; echo "[$(date '+%F %T')] $name"; "$@" > "$R/logs/$name.log" 2>&1; }

run build python scripts/build_e4r_dataset.py
run exp2 python scripts/run_experiment2_heldout_routing.py --datasets climate-fever --output_dir $R/experiment2
run reliability python scripts/analyze_label_reliability.py --datasets climate-fever --exp2_dir $R/experiment2 --output_dir $R/label_reliability
echo "[$(date '+%F %T')] done"
