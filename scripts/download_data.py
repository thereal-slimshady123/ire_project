"""Fetch BEIR datasets into data/raw/."""

import argparse
import io
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import yaml


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
        help="Path to datasets config YAML (provides download URLs)",
    )
    return parser.parse_args()


def fetch_bytes(url: str) -> bytes:
    """Fetch a URL; fall back to curl when Python's TLS stack rejects the certificate chain.

    The BEIR mirror does not send its intermediate certificate, which OpenSSL-based urllib
    cannot recover from but OS-native TLS stacks (e.g. curl on Windows) can.
    """
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.URLError as err:
        if "CERTIFICATE_VERIFY_FAILED" not in str(err):
            raise
        print("[warn] Python TLS verification failed; retrying with curl")
        return subprocess.run(["curl", "-sSfL", url], check=True, capture_output=True).stdout


def download_dataset(name: str, url: str, output_dir: Path) -> Path:
    """Download and unpack one BEIR zip to output_dir/name (skipped if already present)."""
    target = output_dir / name
    if (target / "corpus.jsonl").exists():
        print(f"[skip] {name}: already present at {target}")
        return target
    print(f"[download] {name}: {url}")
    payload = fetch_bytes(url)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(output_dir)
    # BEIR zips unpack to a folder named after the dataset; normalize if the name differs.
    if not target.exists():
        roots = {Path(n).parts[0] for n in zipfile.ZipFile(io.BytesIO(payload)).namelist()}
        if len(roots) == 1:
            shutil.move(str(output_dir / roots.pop()), str(target))
    print(f"[done] {name} -> {target}")
    return target


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["datasets"]
    print(f"Downloading datasets: {args.datasets} to {args.output_dir}")
    for name in args.datasets:
        if name not in cfg:
            raise KeyError(f"Dataset '{name}' not in {args.config}")
        download_dataset(name, cfg[name]["url"], args.output_dir)


if __name__ == "__main__":
    main()
