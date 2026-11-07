#!/usr/bin/env bash
# Fixed pipeline for results/preregistration/phase2_strengthening_20261107.md. Run once, from the project root:
#   bash scripts/run_strengthening_pipeline.sh
# Every step writes under results/strengthening/fresh/ (logs in logs/). Nothing here is tuned on the fresh data.
set -euo pipefail
cd "$(dirname "$0")/.."

FORUMS="cqadupstack-stats cqadupstack-wordpress cqadupstack-tex"
NONFORUM="scidocs nq hotpotqa"
ALL="$FORUMS $NONFORUM"
SEEN="nfcorpus scifact fiqa cqadupstack-unix quora wands cosqa cqadupstack-android cqadupstack-webmasters arguana cqadupstack-mathematica cqadupstack-english cqadupstack-gaming cqadupstack-physics cqadupstack-programmers cqadupstack-gis"
SEEN_DIRS="results/experiment2 results/niche/experiment2 results/phase2_validation/experiment2 results/e4_fresh/experiment2"
S=results/strengthening/fresh
E=results/strengthening/explore
mkdir -p "$S/logs"
run() { local name=$1; shift; echo "[$(date '+%F %T')] $name"; "$@" > "$S/logs/$name.log" 2>&1; }

# 1. Data (downloads happen here, after the pre-registration is hashed)
run build python scripts/build_strengthening_datasets.py

# 2. e5-small retrieval + Exp 2 / Exp 2d
run exp2_small python scripts/run_experiment2_heldout_routing.py --datasets $ALL --output_dir $S/experiment2
run exp2d_small python scripts/run_experiment2d_cost_aware_router.py --datasets $ALL --exp2_dir $S/experiment2 \
    --exp2c_dir $S/experiment2c --output_dir $S/experiment2d

# 3. E2 replication (e5-small): forums, non-forum
run e2r_forums python scripts/run_e2_replication.py --datasets $FORUMS --exp2_dir $S/experiment2 --exp2d_dir $S/experiment2d \
    --label "fresh forums, e5-small-v2" --output_dir $S/e2_forums_small
run e2r_nonforum python scripts/run_e2_replication.py --datasets $NONFORUM --exp2_dir $S/experiment2 --exp2d_dir $S/experiment2d \
    --label "fresh non-forum, e5-small-v2" --output_dir $S/e2_nonforum_small

# 4. E7 + E3 continuity: matched skip rates, then measured cost on an otherwise idle machine (CPU, then GPU)
run matched python scripts/run_e4_matched_rate.py --datasets $ALL --exp2_dir $S/experiment2 --output_dir $S/matched_rate
run cost_cpu python scripts/run_cost_frontier.py --datasets $ALL --exp2_dir $S/experiment2 --matched_dir $S/matched_rate \
    --device cpu --output_dir $S/cost_frontier
run cost_gpu python scripts/run_cost_frontier.py --datasets $ALL --exp2_dir $S/experiment2 --matched_dir $S/matched_rate \
    --device cuda --output_dir $S/cost_frontier

# 5. E5: frozen niches (frozen in results/strengthening/explore/niche_discovery/frozen_niches.json)
run e5_niche python scripts/discover_niche.py --datasets $ALL --exp2_dir $S/experiment2 --output_dir $S/niche_test \
    --frozen $E/niche_discovery/frozen_niches.json

# 6. E6: per-document learned fusion, one model trained on the 16 seen datasets, applied unchanged
run e6_fusion python scripts/run_learned_fusion.py --features doc+query --datasets $SEEN --exp2_dir $SEEN_DIRS \
    --apply_datasets $ALL --apply_exp2_dir $S/experiment2 --output_dir $S/learned_fusion

# 7. Learning curve (descriptive, pre-recorded prediction)
run learning_curve python scripts/run_learning_curve.py --dataset hotpotqa --exp2_dir $S/experiment2 --output_dir $S/learning_curve

# 8. e5-base robustness: retrieval + Exp 2 / 2d + E2 on all six
run exp2_base python scripts/run_experiment2_heldout_routing.py --datasets $ALL --dense_model intfloat/e5-base-v2 \
    --output_dir $S/experiment2_e5base
run exp2d_base python scripts/run_experiment2d_cost_aware_router.py --datasets $ALL --dense_model intfloat/e5-base-v2 \
    --exp2_dir $S/experiment2_e5base --exp2c_dir $S/experiment2c_e5base --output_dir $S/experiment2d_e5base
run e2r_base python scripts/run_e2_replication.py --datasets $ALL --exp2_dir $S/experiment2_e5base --exp2d_dir $S/experiment2d_e5base \
    --label "fresh, all six, e5-base-v2" --output_dir $S/e2_all_base

# 9. E4: label reliability (after step 8 so the e5-base model swap is included)
run e4_reliability python scripts/analyze_label_reliability.py --datasets $ALL --exp2_dir $S/experiment2 --output_dir $S/label_reliability
run e4_reliability_nohotpot python scripts/analyze_label_reliability.py --datasets $FORUMS scidocs nq --exp2_dir $S/experiment2 \
    --output_dir $S/label_reliability_excl_hotpotqa

echo "[$(date '+%F %T')] done"
