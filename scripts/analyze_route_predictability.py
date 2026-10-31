"""Explanatory analysis (post hoc): can the router features predict when BM25 / dense beats hybrid?

For each held-out query (Experiment 2 folds, training-fold alpha*), a LightGBM classifier trained on the
training fold predicts "route beats hybrid" from the 18 Experiment 2 features + PCA-32 of the query
embedding (Experiment 2c features). Reported: held-out AUC, and for the most confident 1/2/5/10% of
queries the win/loss shares, the precision among queries whose score changes, and the break-even
precision mean_loss / (mean_gain + mean_loss) computed from the pooled data.

Usage:
  python scripts/analyze_route_predictability.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp2 = _load_script("exp2", "run_experiment2_heldout_routing.py")
TOP_SHARES = [0.01, 0.02, 0.05, 0.10]
N_PCA = 32


def parse_args():
    p = argparse.ArgumentParser(description="Held-out predictability of route wins over hybrid")
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--exp2c_dir", type=Path, default=Path("results/experiment2c"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--output_dir", type=Path, default=Path("results/route_predictability"))
    return p.parse_args()


def main():
    import lightgbm as lgb

    args = parse_args()
    seed = load_config(args.config_dir / "experiment.yaml")["seed"]
    alphas = [float(a) for a in load_config(args.config_dir / "hybrid.yaml")["alpha_grid"]]
    rows = []
    for ds in exp2.DATASETS:
        df = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        emb = pd.read_parquet(args.exp2c_dir / "cache" / f"{ds}_query_emb.parquet").loc[df.index]
        for itr, ite in create_kfold_splits(list(df.index), 5, seed):
            tr, te = df.loc[itr], df.loc[ite]
            hyb = f"ndcg_hyb_{max(alphas, key=lambda a: tr[f'ndcg_hyb_{a}'].mean())}"
            pca = PCA(N_PCA, random_state=seed).fit(emb.loc[itr])

            def features(part, ids):
                return part[exp2.FEATURES].join(pd.DataFrame(pca.transform(emb.loc[ids]), index=ids).add_prefix("pc"))

            X_tr, X_te = features(tr, itr), features(te, ite)
            out = pd.DataFrame({"dataset": ds}, index=te.index)
            for r in ("bm25", "dense"):
                m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                                       random_state=seed, verbose=-1).fit(X_tr, (tr[f"ndcg_{r}"] > tr[hyb]).astype(int))
                out[f"{r}_p"] = m.predict_proba(X_te)[:, 1]
                out[f"{r}_delta"] = te[f"ndcg_{r}"] - te[hyb]
            rows.append(out)
    H = pd.concat(rows)

    summary, lines = {}, [
        "# Route predictability (explanatory, post hoc)", "",
        "Held-out (Experiment 2 folds): a LightGBM classifier on the 18 features + PCA-32 query embedding predicts "
        "\"route beats hybrid\". Precision is among queries whose nDCG@10 changes; break-even = mean loss / "
        "(mean gain + mean loss) over all queries.", "",
        "| Route | Win rate | Mean gain / loss | Break-even precision | AUC | Most confident | Win / lose | Precision | Mean Δ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for r in ("bm25", "dense"):
        d = H[f"{r}_delta"]
        gain, loss = float(d[d > 0].mean()), float(-d[d < 0].mean())
        need = loss / (gain + loss)
        auc = roc_auc_score((d > 0).astype(int), H[f"{r}_p"])
        summary[r] = {"win_rate": float((d > 0).mean()), "mean_gain": gain, "mean_loss": loss,
                      "break_even_precision": need, "auc": float(auc), "top": {}}
        for share in TOP_SHARES:
            sel = H[f"{r}_p"] >= H[f"{r}_p"].quantile(1 - share)
            s = d[sel]
            prec = float((s[s != 0] > 0).mean())
            summary[r]["top"][str(share)] = {"n": int(sel.sum()), "win": float((s > 0).mean()),
                                             "lose": float((s < 0).mean()), "precision": prec, "mean_delta": float(s.mean())}
            lines.append(f"| {r} | {(d > 0).mean():.1%} | +{gain:.3f} / −{loss:.3f} | {need:.0%} | {auc:.3f} | {share:.0%} "
                         f"({int(sel.sum())} q) | {(s > 0).mean():.0%} / {(s < 0).mean():.0%} | {prec:.0%} | {s.mean():+.4f} |")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
