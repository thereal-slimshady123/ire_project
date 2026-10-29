"""Smoke tests: the Track 2 CLI scripts run end to end on a tiny fixture results directory."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from router_retrieval.synthetic import save_synthetic
from tests.track2_fixtures import make_route_table

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["script", *argv])
    mod.main()


@pytest.fixture()
def workspace(tmp_path):
    results = tmp_path / "results"
    cache = results / "experiment2" / "cache"
    cache.mkdir(parents=True)
    for i, name in enumerate(["toya", "toyb", "toyc"]):
        make_route_table(120, seed=i, name=name).to_csv(cache / f"{name}.csv")
    cfg = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", cfg)
    router = yaml.safe_load((cfg / "router.yaml").read_text())
    router["models"]["gbm"].update(n_estimators=15, num_leaves=7)
    router["advantage"].update(n_estimators=25, learning_rate=0.1, min_child_samples=8)
    (cfg / "router.yaml").write_text(yaml.safe_dump(router))
    exp = yaml.safe_load((cfg / "experiment.yaml").read_text())
    exp["k_folds"] = 3
    (cfg / "experiment.yaml").write_text(yaml.safe_dump(exp))
    return tmp_path, results, cfg


COMMON = lambda results, cfg: ["--results_dir", str(results), "--config_dir", str(cfg), "--n_bootstrap", "200"]


def test_extract_features_cache_mode_writes_contract2(workspace, monkeypatch):
    _, results, cfg = workspace
    _run(_load("extract_features"), ["--datasets", "toya", "--source", "cache", "--results_dir", str(results),
                                      "--config_dir", str(cfg), "--output_dir", str(results / "features")], monkeypatch)
    df = pd.read_parquet(results / "features" / "toya.parquet")
    assert len(df) == 120 and {"query_id", "jaccard_overlap", "rank_correlation", "max_idf"} <= set(df.columns)
    # the exported features flow back into the route table loader
    from router_retrieval.router.cv import load_route_table
    assert load_route_table("toya", results).shape[0] == 120


def test_train_router_writes_contract3_and_models(workspace, monkeypatch):
    _, results, cfg = workspace
    out = results / "models"
    _run(_load("train_router"), ["--datasets", "toya", "toyb", "--kind", "all", "--output_dir", str(out),
                                  *COMMON(results, cfg)], monkeypatch)
    pred = pd.read_parquet(out / "toya" / "predictions.parquet")
    assert list(pred.columns) == ["query_id", "fold", "route_predicted", "p_bm25", "p_dense", "p_hybrid",
                                  "delta_pred_bm25", "delta_pred_dense"] and len(pred) == 120
    assert (out / "toya" / "router_advantage.pkl").exists() and (out / "toya" / "feature_importances.csv").exists()
    assert (out / "toya" / "predictions_gbm_cls.parquet").exists()
    summary = json.loads((out / "toya" / "summary.json").read_text())
    assert summary["kinds"]["advantage"]["tests"]["routed - hybrid"]["delta_mean"] > 0.02   # planted signal found
    assert (out / "cv_summary.csv").exists()


def test_ablations_cross_domain_and_experiment2b(workspace, monkeypatch):
    _, results, cfg = workspace
    abl = results / "experiment2" / "ablations"
    _run(_load("run_ablations"), ["--datasets", "toya", "toyb", "--kinds", "advantage", "gbm_cls", "--output_dir", str(abl),
                                   *COMMON(results, cfg)], monkeypatch)
    s = json.loads((abl / "summary.json").read_text())
    full = s["advantage"]["all"]["pooled"]["tests"]["routed - hybrid"]["delta_mean"]
    assert full > 0.02
    assert "Gain importance" in (abl / "tables.md").read_text() and (abl / "feature_importances.csv").exists()

    cd = results / "experiment2" / "cross_domain"
    _run(_load("run_cross_domain"), ["--source_dataset", "toya", "--target_dataset", "toyb", "--output_dir", str(cd),
                                      *COMMON(results, cfg)], monkeypatch)
    m = pd.read_csv(cd / "matrix.csv")
    assert list(m["source"]) == ["toya"] and list(m["target"]) == ["toyb"] and m.loc[0, "gap_src_alpha"] > 0.0

    # synthetic scaling: the same planted signal, pooled in-domain
    syn_dir = results / "synthetic"
    for name in ("toya", "toyb"):
        syn = make_route_table(150, seed=11, name=name)
        syn.index = [f"syn-{name}-{i}" for i in range(len(syn))]
        syn["style"] = ["lexical", "semantic"] * 75
        syn["source_doc"], syn["query_text"] = "d", "q"
        save_synthetic(syn, syn_dir / f"{name}.parquet")
    out = results / "experiment2" / "advantage"
    _run(_load("run_experiment2b_advantage"), ["--datasets", "toya", "toyb", "--synthetic_dir", str(syn_dir),
                                                "--n_synthetic", "0", "100", "--output_dir", str(out),
                                                *COMMON(results, cfg)], monkeypatch)
    summ = json.loads((out / "summary.json").read_text())
    assert [r["label"].split(" ")[0] for r in summ["scaling"]] == ["0", "100"]
    assert summ["scaling"][1]["label"] == "100 per dataset (used: toya 100, toyb 100)"
    assert summ["scaling"][1]["n_synthetic"] == {"toya": 100, "toyb": 100}
    assert "Scaling with synthetic training queries" in (out / "tables.md").read_text()


def _write_tiny_beir(root: Path, n_docs=80, n_queries=8):
    """Tiny BEIR-format dataset: distinctive per-document vocabulary so lexical queries are retrievable."""
    (root / "qrels").mkdir(parents=True)
    docs = [{"_id": f"D{i}", "title": f"Topic{i}",
             "text": " ".join(f"word{(i * 7 + j) % 53}x{i} common{j % 5} term{i}" for j in range(12))} for i in range(n_docs)]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    qs = [{"_id": f"nq{i}", "text": f"topic{i} word{i} question"} for i in range(n_queries)]
    (root / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"nq{i}\tD{i}\t1\n" for i in range(n_queries)), encoding="utf-8")


def test_generate_synthetic_queries_end_to_end_without_models(tmp_path, monkeypatch):
    from router_retrieval.retrieval.bm25 import BM25Retriever
    from router_retrieval.synthetic import load_synthetic

    gen = _load("generate_synthetic_queries")
    _write_tiny_beir(tmp_path / "raw" / "tiny")

    class StandInDense:  # BM25 with other parameters in place of the e5 model
        def __init__(self, corpus):
            self.r = BM25Retriever(k1=2.0, b=0.2)
            self.r.index(corpus)

        def batch_search(self, queries, k=100):
            return self.r.batch_search(queries, k)

    monkeypatch.setattr(gen, "load_dense", lambda args, ds, corpus, ret_cfg: StandInDense(corpus))
    out = tmp_path / "synthetic"
    _run(gen, ["--datasets", "tiny", "--n_total", "50", "--lexical_only", "--min_passage_words", "5",
               "--data_dir", str(tmp_path / "raw"), "--results_dir", str(tmp_path / "results"),
               "--output_dir", str(out), "--config_dir", str(ROOT / "configs")], monkeypatch)
    table = load_synthetic(out / "tiny.parquet")
    manifest = json.loads((out / "manifest.json").read_text())
    assert 0 < len(table) <= 50 and set(table["style"]) == {"lexical"}
    # leakage control: no synthetic query may be sourced from a passage relevant to a natural test query
    assert not set(table["source_doc"]) & {f"D{i}" for i in range(8)}
    assert manifest["datasets"]["tiny"]["n_excluded_passages"] == 8
    assert table["ndcg_bm25"].mean() > 0.5                      # distinctive keywords retrieve their source
    assert {"ndcg_hyb_0.0", "ndcg_hyb_1.0", "bm25_top1_top2_gap"} <= set(table.columns)
