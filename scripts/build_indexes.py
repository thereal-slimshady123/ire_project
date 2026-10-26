"""Build BM25 and FAISS indexes for a dataset or all evaluation benchmarks."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.beir_loader import load_corpus
from router_retrieval.data.schema import Document
from router_retrieval.retrieval.bm25 import BM25Retriever
from router_retrieval.retrieval.dense import DenseRetriever
from router_retrieval.utils.config import load_config
from router_retrieval.utils.logging_utils import get_logger

log = get_logger("build_indexes")
DEFAULT_DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS dense indexes for BEIR datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name (e.g. nfcorpus, scifact, fiqa, cqadupstack-unix, quora) or 'all'",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Root data directory containing raw/ or processed/ datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/indexes"),
        help="Directory to save generated index files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrievers.yaml"),
        help="Path to retrievers config YAML",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["all", "bm25", "dense"],
        default="all",
        help="Which index to build ('all', 'bm25', 'dense')",
    )
    parser.add_argument(
        "--dense_model",
        type=str,
        default=None,
        help="Dense embedding model name (e.g., intfloat/e5-small-v2, facebook/contriever)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('auto', 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if saved index files already exist",
    )
    return parser.parse_args()


def get_corpus_path(data_dir: Path, dataset: str) -> Path:
    """Locate corpus.jsonl in processed/ (for subsamples like quora) or raw/."""
    processed = data_dir / "processed" / f"{dataset}-sub" / "corpus.jsonl"
    if processed.exists():
        return processed
    raw = data_dir / "raw" / dataset / "corpus.jsonl"
    if raw.exists():
        return raw
    fallback = data_dir / dataset / "corpus.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find corpus.jsonl for dataset '{dataset}' under {data_dir}")


def build_bm25_index(
    dataset: str, corpus: Dict[str, Document], output_dir: Path, bm25_cfg: dict, force: bool
) -> Path:
    save_path = output_dir / dataset / "bm25.pkl"
    if save_path.exists() and not force:
        log.info(f"[{dataset}] BM25 index already exists at {save_path}; skipping (use --force to overwrite)")
        return save_path

    t0 = time.perf_counter()
    k1 = float(bm25_cfg.get("k1", 1.2))
    b = float(bm25_cfg.get("b", 0.75))
    log.info(f"[{dataset}] Building BM25 index (k1={k1}, b={b}) over {len(corpus)} documents...")
    bm25 = BM25Retriever(k1=k1, b=b)
    bm25.index(corpus)
    bm25.save(save_path)
    elapsed = time.perf_counter() - t0
    log.info(
        f"[{dataset}] BM25 index built in {elapsed:.2f}s: vocab={len(bm25.vocab)} terms, saved to {save_path}"
    )
    return save_path


def build_dense_index(
    dataset: str,
    corpus: Dict[str, Document],
    output_dir: Path,
    dense_cfg: dict,
    model_override: str,
    device_override: str,
    force: bool,
) -> Path:
    model_name = model_override or dense_cfg.get("model_name", "intfloat/e5-small-v2")
    device = device_override or dense_cfg.get("device", "auto")
    slug = model_name.replace("/", "__")
    save_dir = output_dir / dataset / slug

    if (save_dir / "embeddings.npy").exists() and (save_dir / "meta.json").exists() and not force:
        log.info(f"[{dataset}] Dense index already exists at {save_dir}; skipping (use --force to overwrite)")
        return save_dir

    t0 = time.perf_counter()
    passage_prefix = "passage: " if "e5" in model_name.lower() else ""
    query_prefix = "query: " if "e5" in model_name.lower() else ""

    kwargs = {
        "model_name": model_name,
        "batch_size": int(dense_cfg.get("batch_size", 64)),
        "device": device,
        "faiss_index_type": dense_cfg.get("faiss_index_type", "IndexFlatIP"),
        "max_length": dense_cfg.get("max_length", 512),
        "passage_prefix": passage_prefix,
        "query_prefix": query_prefix,
    }
    log.info(f"[{dataset}] Initializing dense retriever with {model_name} on {device}...")
    dense = DenseRetriever(**kwargs)
    log.info(f"[{dataset}] Encoding {len(corpus)} documents into FAISS index...")
    dense.index(corpus, show_progress_bar=True)
    dense.save(save_dir)
    elapsed = time.perf_counter() - t0
    log.info(f"[{dataset}] Dense index built in {elapsed:.2f}s: saved to {save_dir}")
    return save_dir


def main():
    args = parse_args()
    ret_cfg = load_config(args.config) if args.config.exists() else {}
    bm25_cfg = ret_cfg.get("bm25", {"k1": 1.2, "b": 0.75})
    dense_cfg = ret_cfg.get("dense", {"model_name": "intfloat/e5-small-v2", "device": "auto"})

    datasets = DEFAULT_DATASETS if args.dataset.lower() == "all" else [args.dataset]

    for ds in datasets:
        log.info(f"=== Processing dataset: {ds} ===")
        corpus_path = get_corpus_path(args.data_dir, ds)
        log.info(f"[{ds}] Loading corpus from {corpus_path}...")
        corpus = load_corpus(corpus_path)
        log.info(f"[{ds}] Loaded {len(corpus)} documents.")

        if args.retriever in ("all", "bm25"):
            build_bm25_index(ds, corpus, args.output_dir, bm25_cfg, args.force)

        if args.retriever in ("all", "dense"):
            build_dense_index(
                ds, corpus, args.output_dir, dense_cfg, args.dense_model, args.device, args.force
            )

    log.info("Index building process completed.")


if __name__ == "__main__":
    main()
