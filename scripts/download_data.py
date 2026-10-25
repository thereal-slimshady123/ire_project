"""Fetch BEIR datasets into data/raw/."""

import argparse
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

from router_retrieval.utils.config import load_config
from router_retrieval.utils.logging_utils import get_logger

logger = get_logger("download_data")


class DownloadProgressBar(tqdm):
    """Progress bar for urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    """Download a file with curl (with resume support) or fallback to urllib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {url} to {output_path}...")

    # If curl is available, use it with resume (-C -)
    curl_path = shutil.which("curl")
    if curl_path:
        cmd = [
            curl_path,
            "-C", "-",
            "-L",
            "--retry", "5",
            "--retry-delay", "2",
            "-o", str(output_path),
            url,
        ]
        ret = os.system(" ".join(cmd))
        if ret == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return

    # Fallback to urllib with progress bar
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_and_organize(zip_path: Path, output_dir: Path, dataset_name: str, beir_name: str) -> Path:
    """
    Extract zip archive and ensure files reside at output_dir / dataset_name.
    """
    logger.info(f"Extracting {zip_path}...")
    target_dir = output_dir / dataset_name
    temp_extract_dir = output_dir / f"_temp_{dataset_name}"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_extract_dir)

    # Identify where corpus.jsonl landed
    found_corpus = list(temp_extract_dir.rglob("corpus.jsonl"))
    if not found_corpus:
        raise FileNotFoundError(f"corpus.jsonl not found inside {zip_path}")

    actual_dataset_root = found_corpus[0].parent

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.move(str(actual_dataset_root), str(target_dir))
    shutil.rmtree(temp_extract_dir, ignore_errors=True)

    # Remove temporary zip file to save disk space
    if zip_path.exists():
        zip_path.unlink()

    return target_dir


def is_dataset_ready(dataset_dir: Path) -> bool:
    """Check if corpus.jsonl, queries.jsonl, and qrels/ exist and are non-empty."""
    corpus = dataset_dir / "corpus.jsonl"
    queries = dataset_dir / "queries.jsonl"
    qrels = dataset_dir / "qrels"
    return (
        corpus.is_file()
        and corpus.stat().st_size > 0
        and queries.is_file()
        and queries.stat().st_size > 0
        and qrels.is_dir()
        and any(qrels.glob("*.tsv"))
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Download BEIR datasets to data/raw/")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nfcorpus", "scifact", "fiqa"],
        help="List of BEIR dataset names to download",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save raw datasets",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to datasets configuration file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    datasets_cfg = config.get("datasets", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        if dataset_name not in datasets_cfg:
            logger.error(f"Dataset '{dataset_name}' not defined in {args.config}. Skipping.")
            continue

        d_info = datasets_cfg[dataset_name]
        url = d_info.get("url")
        beir_name = d_info.get("beir_name", dataset_name)
        target_dir = args.output_dir / dataset_name

        if not args.force and is_dataset_ready(target_dir):
            logger.info(f"Dataset '{dataset_name}' already exists at {target_dir}. Skipping. (use --force to re-download)")
            continue

        if not url:
            logger.error(f"No URL provided for dataset '{dataset_name}'. Skipping.")
            continue

        zip_path = args.output_dir / f"{dataset_name}.zip"
        try:
            if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
                download_file(url, zip_path)
            else:
                logger.info(f"Using existing archive {zip_path}")
            dataset_dir = extract_and_organize(zip_path, args.output_dir, dataset_name, beir_name)
            logger.info(f"Successfully set up dataset '{dataset_name}' at {dataset_dir}")
        except Exception as e:
            logger.error(f"Failed to process dataset '{dataset_name}': {e}", exc_info=True)
            if zip_path.exists():
                zip_path.unlink()
            raise


if __name__ == "__main__":
    main()
