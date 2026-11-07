#!/usr/bin/env bash
# Fixed pipeline for results/preregistration/e8_boundary_20261107.md. Run once, from the project root:
#   bash scripts/run_boundary_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

COMP="musique 2wikimultihopqa"
ALT="dbpedia"
B=results/strengthening/boundary
mkdir -p "$B/logs"
run() { local name=$1; shift; echo "[$(date '+%F %T')] $name"; "$@" > "$B/logs/$name.log" 2>&1; }

run build python scripts/build_boundary_datasets.py
run exp2 python scripts/run_experiment2_heldout_routing.py --datasets $COMP $ALT --output_dir $B/experiment2
run matched python scripts/run_e4_matched_rate.py --datasets $COMP $ALT --exp2_dir $B/experiment2 --output_dir $B/matched_rate
run e8 python scripts/compare_boundary.py --complementary $COMP --alternative $ALT --exp2_dir $B/experiment2 \
    --matched_dir $B/matched_rate --output_dir $B/e8
run reliability python scripts/analyze_label_reliability.py --datasets $COMP $ALT --exp2_dir $B/experiment2 --output_dir $B/label_reliability
echo "[$(date '+%F %T')] done"
