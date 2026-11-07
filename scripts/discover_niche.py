"""Search for a query niche where something other than the tuned fixed-alpha hybrid wins, and (with --frozen) test
frozen niches on new data.

Niche = one or two conditions on the 18 router features, each feature converted to a within-dataset percentile rank
(so a niche transfers to new corpora; no relevance labels are used to define it). Condition = bottom or top
10% / 20% / 33% of a feature. Action = BM25 only, dense only, or hybrid at any alpha of the grid.
Baseline = the dataset's best fixed alpha (alpha* tuned on all of that dataset's queries -- the strongest fixed hybrid,
which makes it harder, not easier, for a niche to win).

Discovery (exploration data): candidates with n >= 200 queries in >= 5 datasets (>= 20 queries each) are ranked by
the normal-approximation 95% lower bound of mean(action - hybrid*) inside the niche; a candidate must also have a
positive mean in both halves of a seeded random split of the datasets (internal replication). Frozen: the top K
distinct niches, plus the best-ranked distinct niche with n >= 1000 (more power to replicate).

Test (--frozen <json>): each frozen niche's action - hybrid* inside the niche, paired bootstrap (10,000, seed 42),
Holm-corrected across the frozen niches; plus the niche rule's overall effect (niche -> action, else hybrid*).

Usage:
  python scripts/discover_niche.py --exp2_dir <dir ...> --datasets <ds ...> --output_dir <dir> [--top_k 3]
  python scripts/discover_niche.py --exp2_dir <dir ...> --datasets <ds ...> --output_dir <dir> --frozen <frozen.json>
"""

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402

spec = importlib.util.spec_from_file_location("exp2", ROOT / "scripts" / "run_experiment2_heldout_routing.py")
exp2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp2)

FEATURES = list(exp2.FEATURES)
QUANTS = [0.10, 0.20, 0.33]
ALPHAS = [round(a / 10, 1) for a in range(11)]
MIN_N, MIN_DS, MIN_PER_DS, TOP_K, LARGE_N, SEED, N_BOOT = 200, 5, 20, 3, 1000, 42, 10000


def load(datasets, dirs):
    frames = []
    for ds in datasets:
        src = next(d for d in dirs if (d / "cache" / f"{ds}.csv").exists())
        df = pd.read_csv(src / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = ds + ":" + df.index.astype(str)
        a_star = max(ALPHAS, key=lambda a: df[f"ndcg_hyb_{a}"].mean())
        df["hybrid_star"] = df[f"ndcg_hyb_{a_star}"]
        df["alpha_star"] = a_star
        for f in FEATURES:  # within-dataset percentile rank; ties get their average rank
            df[f"pct_{f}"] = df[f].rank(pct=True, method="average")
        frames.append(df)
    return pd.concat(frames)


def actions(df):
    acts = {"bm25": df["ndcg_bm25"], "dense": df["ndcg_dense"], **{f"hybrid@{a}": df[f"ndcg_hyb_{a}"] for a in ALPHAS}}
    return {k: (v - df["hybrid_star"]).to_numpy() for k, v in acts.items()}


def conditions():
    for f in FEATURES:
        for q in QUANTS:
            yield (f, "low", q)
            yield (f, "high", q)


def cond_mask(df, c):
    f, side, q = c
    p = df[f"pct_{f}"].to_numpy()
    return p <= q if side == "low" else p > 1 - q


def niche_name(conds):
    return " AND ".join(f"{f} in {'bottom' if s == 'low' else 'top'} {q:.0%}" for f, s, q in conds)


def niche_mask(df, conds):
    m = np.ones(len(df), bool)
    for c in conds:
        m &= cond_mask(df, tuple(c))
    return m


def discover(df, top_k):
    singles = list(conditions())
    cands = [(c,) for c in singles] + [(a, b) for a, b in itertools.combinations(singles, 2) if a[0] != b[0]]
    M = np.stack([niche_mask(df, c) for c in cands]).astype(np.float32)  # candidates x queries
    A = actions(df)
    names = list(A)
    D = np.stack([A[k] for k in names], axis=1).astype(np.float64)  # queries x actions
    n = M.sum(axis=1)
    s1, s2 = M @ D, M @ (D ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n[:, None]
        sd = np.sqrt(np.maximum(s2 / n[:, None] - mean ** 2, 0) * n[:, None] / np.maximum(n[:, None] - 1, 1))
        lb = mean - 1.96 * sd / np.sqrt(n[:, None])
    ds_codes, ds_names = pd.factorize(df["dataset"])
    per_ds = np.stack([M[:, ds_codes == i].sum(axis=1) for i in range(len(ds_names))], axis=1)
    n_ds = (per_ds >= MIN_PER_DS).sum(axis=1)
    rng = np.random.default_rng(SEED)
    half = rng.permutation(len(ds_names)) % 2
    halves = []
    for h in (0, 1):
        cols = np.isin(ds_codes, np.flatnonzero(half == h))
        nh = M[:, cols].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            halves.append((M[:, cols] @ D[cols]) / nh[:, None])
    ok = (n[:, None] >= MIN_N) & (n_ds[:, None] >= MIN_DS) & (halves[0] > 0) & (halves[1] > 0)
    score = np.where(ok, lb, -np.inf)
    order = np.argsort(-score, axis=None)
    picked, seen, large = [], set(), None
    rows = []
    for flat in order:
        i, j = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[i, j]):
            break
        rows.append({"niche": niche_name(cands[i]), "action": names[j], "n": int(n[i]), "datasets": int(n_ds[i]),
                     "mean": float(mean[i, j]), "lb95": float(lb[i, j]), "half0": float(halves[0][i, j]),
                     "half1": float(halves[1][i, j])})
        key = cands[i]  # one action per niche: the K frozen niches are distinct query sets
        entry = {"conditions": [list(c) for c in cands[i]], "action": names[j], "name": niche_name(cands[i])}
        if len(picked) < top_k and key not in seen:
            seen.add(key)
            picked.append({**entry, "rule": f"top-{top_k} by 95% LB"})
        elif large is None and n[i] >= LARGE_N and key not in seen:
            large = {**entry, "rule": f"best with n >= {LARGE_N}"}
        if len(rows) >= 50 and len(picked) == top_k and large is not None:
            break
    if large is not None:
        picked.append(large)
    meta = {"n_candidates": len(cands), "n_actions": len(names), "n_tests": int(len(cands) * len(names)),
            "n_eligible": int(np.isfinite(score).sum()), "split_half": {str(ds_names[i]): int(half[i]) for i in range(len(ds_names))}}
    return picked, pd.DataFrame(rows), meta


def test_frozen(df, frozen):
    A = actions(df)
    res = []
    for fz in frozen:
        m = niche_mask(df, fz["conditions"])
        d = A[fz["action"]][m]
        t = paired_bootstrap_test(d, np.zeros(len(d)), N_BOOT, SEED)
        overall = np.where(m, A[fz["action"]], 0.0)
        o = paired_bootstrap_test(overall, np.zeros(len(overall)), N_BOOT, SEED)
        per_ds = {ds: {"n": int(g.sum()), "mean": float(A[fz["action"]][g].mean()) if g.sum() else None}
                  for ds in df["dataset"].unique() for g in [m & (df["dataset"] == ds).to_numpy()]}
        res.append({**fz, "n": int(m.sum()), "share": float(m.mean()), "inside": t, "overall": o, "per_dataset": per_ds})
    p = np.array([r["inside"]["p_value"] for r in res])
    order = np.argsort(p)
    holm, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):  # Holm step-down adjusted p-values
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        holm[i] = running
    for r, h in zip(res, holm):
        r["p_holm"] = float(h)
        r["supported"] = bool(h < 0.05 and r["inside"]["ci_95_low"] > 0)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--exp2_dir", type=Path, nargs="+", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--frozen", type=Path)
    args = ap.parse_args()
    df = load(args.datasets, args.exp2_dir)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    if args.frozen:
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))["frozen"]
        res = test_frozen(df, frozen)
        (out / "niche_test.json").write_text(json.dumps({"datasets": args.datasets, "results": res}, indent=2), encoding="utf-8")
        L = ["# Frozen-niche test", "", f"Datasets: {', '.join(args.datasets)} (n = {len(df)}). Baseline: each dataset's best fixed α.", "",
             "| # | Niche → action | n (share) | Inside niche: action − hybrid* | Holm p | Overall rule − hybrid* | Supported |",
             "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |"]
        for i, r in enumerate(res, 1):
            L.append(f"| {i} | {r['name']} → **{r['action']}** | {r['n']} ({r['share']:.1%}) | {ci(r['inside'])} | {r['p_holm']:.3g} "
                     f"| {ci(r['overall'])} | {'**yes**' if r['supported'] else 'no'} |")
        (out / "niche_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))
        return
    picked, table, meta = discover(df, args.top_k)
    (out / "frozen_niches.json").write_text(json.dumps({"datasets": args.datasets, "meta": meta, "frozen": picked}, indent=2),
                                            encoding="utf-8")
    table.to_csv(out / "top_candidates.csv", index=False)
    L = ["# Niche discovery (exploration data)", "",
         f"Datasets: {', '.join(args.datasets)} (n = {len(df)}). {meta['n_candidates']} niches × {meta['n_actions']} actions = "
         f"{meta['n_tests']} candidates; {meta['n_eligible']} eligible (n ≥ {MIN_N}, ≥ {MIN_DS} datasets, positive in both halves).", "",
         "| Rank | Niche | Action | n | Datasets | Mean − hybrid* | 95% LB | Half A | Half B |",
         "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for i, r in enumerate(table.head(20).itertuples(), 1):
        L.append(f"| {i} | {r.niche} | {r.action} | {r.n} | {r.datasets} | {r.mean:+.4f} | {r.lb95:+.4f} | {r.half0:+.4f} | {r.half1:+.4f} |")
    L += ["", "Frozen for the confirmatory test: " + "; ".join(f"({i}) {p['name']} → {p['action']}" for i, p in enumerate(picked, 1))]
    (out / "discovery.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
