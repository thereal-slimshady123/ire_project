"""Write Contract 1 run files (`results/runs/{dataset}/{bm25,dense}.parquet`) from the cached TREC runs.

Parquet is the canonical run format (plan.md Contract 1); the experiment scripts cache runs as `.trec`,
and this converts every dataset's BM25 and dense TREC run without re-running retrieval. Existing parquet
files are left alone unless --overwrite is given.

Usage:
  python scripts/export_run_parquet.py
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


baselines = _load_script("baselines", "run_retrieval_baselines.py")
log = get_logger("export_runs")
DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]


def parse_args():
    p = argparse.ArgumentParser(description="Convert cached TREC runs to Contract 1 parquet")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--runs_dir", type=Path, default=Path("results/runs"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bm25_cfg = load_config(args.config_dir / "retrievers.yaml")["bm25"]
    for ds in args.datasets:
        run_dir = args.runs_dir / ds
        sources = {
            "bm25": run_dir / f"bm25_k1-{bm25_cfg['k1']}_b-{bm25_cfg['b']}.trec",
            "dense": run_dir / f"dense_{args.dense_model.replace('/', '__')}.trec",
        }
        for name, trec in sources.items():
            target = run_dir / f"{name}.parquet"
            if target.exists() and not args.overwrite:
                log.info(f"[{ds}] {target} exists, skipped")
            elif not trec.exists():
                log.warning(f"[{ds}] {trec} missing; run the retrieval scripts first")
            else:
                baselines.save_run_parquet(baselines.read_trec_run(trec), target)
                log.info(f"[{ds}] {trec.name} -> {target}")


if __name__ == "__main__":
    main()
