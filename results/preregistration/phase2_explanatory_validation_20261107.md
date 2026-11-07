# Phase 2: explanatory hypotheses E1–E2, pre-registered validation

Written and hashed on 2026-11-07, **before** the validation datasets below were downloaded, built or
scored. The validation run happens once and is reported whatever the result.

## Where these hypotheses come from (not blind to the exploration data)

The mid-submission report listed tentative Phase 2 hypotheses P2-H1–H4 ("subject to change"). Exploratory
analysis on 13 datasets (5 BEIR datasets from Phase 1; WANDS, CoSQA, CQADupStack-Android/-Webmasters,
ArguAna, BRIGHT-biology/-economics/-psychology; plus the 288-query synthetic set) refined them into two
explanatory hypotheses about **why Phase 1's router did not beat tuned hybrid retrieval**:

| Mid-submission | Refined into |
| :--- | :--- |
| P2-H1: routing needs query groups whose best retriever differs; natural sets have none | **E2** |
| P2-H2: the 18 features separate keyword from paraphrase queries, so route labels are learnable | **E1** (true on synthetic queries only, by construction) |
| P2-H3 (one α a poor compromise), P2-H4 (U-shaped gain vs style mix) | Not carried forward |

Exploration evidence (all on data already seen; this is why a held-out validation set is needed):
- Synthetic-trained router − hybrid(α=0.6) on 4,856 real queries: −0.0294 [−0.0337, −0.0252].
- In 26 pre-retrieval query groups (dataset, family, length, form, identifier tokens), BM25 and dense were
  below hybrid in every group; Exp 2d router − hybrid (13 datasets, n=7,235) 90% CI [−0.0015, −0.0001].
- Best single-feature AUC for dense-winner vs BM25-winner queries: 0.341 [0.312, 0.371] (real) vs
  query_length AUC 1.000 (synthetic).
- Power check (1,000 resamples of n=4,000 from CQADupStack unix/android/webmasters): P(E1 accepted) ≈ 1.00,
  P(E2 accepted) ≈ 0.94. This only models sampling noise, not differences between subforums.

## Validation data (held out)

- **CQADupStack mathematica, english, gaming** (`mteb/cqadupstack-*` on Hugging Face). Chosen by a fixed rule
  before download: the three CQADupStack subforums not used in exploration with the smallest corpus files.
- Built with `scripts/build_niche_datasets.py --datasets cqadupstack-mathematica cqadupstack-english
  cqadupstack-gaming --max_docs 1000000000` (full corpora, no subsampling). Official test qrels.
- These datasets are not used for anything before this file is hashed, and not used to change any
  setting after.

## Fixed pipeline (unchanged code; hashes below)

- BM25 (k1 = 1.2, b = 0.75) and `intfloat/e5-small-v2` dense, top-100, as in `configs/retrievers.yaml`.
- `scripts/run_experiment2_heldout_routing.py` → `scripts/run_experiment2d_cost_aware_router.py` on the 3
  datasets: 5-fold CV within each dataset, seed 42, α\* chosen per training fold, cost-aware GBM.
- Synthetic-trained router: the Exp 2d recipe (`choose_epsilon` with fallback, `fit_router`, margin to
  hybrid) trained only on the 288 synthetic queries with hybrid α = 0.6, applied unchanged to the
  validation queries (as in `scripts/diagnose_router_lock.py`, system "(a) Synthetic-trained 2d GBM").
- Metric nDCG@10. Paired bootstrap, 10,000 resamples, seed 42, **pooled over the 3 datasets**.
  Practical margin **δ = 0.005**.
- A new script may be written only to point the existing functions at the validation data; it must not
  change any definition below.

## E1 — Distribution shift

**Claim.** Routing gains on the synthetic set come from how it was constructed (zero-overlap paraphrases,
rare-term keyword bags), which makes the winning retriever trivially predictable from query features;
this does not hold for natural queries, so a router trained on synthetic queries is worse than hybrid on
natural ones.

**Primary test.** Synthetic-trained router − hybrid (α = 0.6), pooled validation queries.
- **Accept** if the 95% CI upper bound < −0.005.
- **Reject** if the 95% CI lower bound > −0.005.
- **Inconclusive** otherwise.

**Secondary (reported, no verdict).** Winner-prediction AUC (dense-winner vs BM25-winner, definition as in
E2(c)) on synthetic vs validation queries; per-dataset results; route shares.

## E2 — No pre-retrieval-identifiable niche

**Claim.** Per-query headroom exists (some queries are better served by BM25 or dense alone), but no group
of queries identifiable before retrieval has a single retriever beating hybrid, the Exp 2d router gains
nothing, and the 18 features cannot predict the per-query winner. This is why the router falls back to
hybrid.

**(a) Groups.** Hybrid = Exp 2d's per-fold α\*. Groupings, as implemented in
`scripts/analyze_niche_existence.py`: dataset; query length in words (1–3, 4–7, 8–15, 16–50, >50); question
form (first word in its question-word list or contains "?"); identifier-like token (its `IDENT` regex).
Only groups with n ≥ 300 are tested. For each group, BM25 − hybrid and dense − hybrid, Holm-corrected across
all tested (group, retriever) pairs. A **niche exists** if any pair is Holm-significant (p < 0.05) with
point estimate ≥ +0.005.

**(b) Router.** Exp 2d cost-aware router − hybrid, pooled: 95% CI upper bound < +0.005.

**(c) Signal.** Queries labelled dense-winner (nDCG_dense > hybrid(α=0.6) and > nDCG_bm25) vs BM25-winner
(symmetric). For each of the 18 features, AUC' = max(AUC, 1 − AUC); best AUC' < 0.65 with its 95% bootstrap
CI upper bound < 0.70.

- **Accept E2** if (a) finds no niche **and** (b) holds **and** (c) holds.
- **Reject E2** if (a) finds a niche, **or** best AUC' ≥ 0.70.
- **Inconclusive** otherwise.

**Secondary (reported, no verdict).** TOST equivalence of router − hybrid at ±0.005 (90% CI); oracle −
hybrid headroom; share of queries where a single retriever beats hybrid by ≥ 0.10; fallback-gate rate.

## E4 — Skipping dense retrieval (secondary, not one of the explanatory hypotheses)

Exp 3's protocol and success criteria are those pre-registered in `improvements.md §4.1`
(hash in `SHA256SUMS.txt`), unchanged: skip rate ≥ 20%, skip-router − hybrid CI low ≥ −0.01, skip-router −
random (same rate) CI low > 0.
- **Pilot on 10 exploration datasets** (already run, `results/niche/experiment3_pilot/`): skip 29.7%;
  − hybrid −0.0059 [−0.0076, −0.0043]; − random +0.0068 [+0.0051, +0.0085] → meets the §4.1 criteria.
  Reported as an extension of the §4.1 protocol to more datasets.
- **Validation set:** reported descriptively. Prediction recorded in advance from the pilot: on forum
  queries (CQADupStack) the skip rate will be **below 20%** (pilot CQADupStack mean 15.6%), so the §4.1
  criteria are expected **not** to be met there.

## Pinned files (SHA-256 at time of writing; git HEAD 9b2c6940019c121b5711fa94656bfebb361d0872 + untracked files)

```
9234a45fe0c6adb31c257c5ef0a7d135ea5ddcf3b959af194df5460a7a1c4d33  scripts/build_niche_datasets.py
f51d3da3d3d696ad4069f790c1d78a07059f775661955ad6b45db81f1ee01690  scripts/diagnose_router_lock.py
1e87f6e1475842831e657f47b5b1906952b4f69b6261056fc80cbcae278af4e5  scripts/analyze_niche_existence.py
8eef1aad1a7a6d2e94a73790d2216f37c7670a29c0f1446edf62461a730bb08d  scripts/run_experiment2_heldout_routing.py
bbb2c7bd309b18387a2de73c29ac253d56168031c3cece59e7f0d24f396a2d93  scripts/run_experiment2d_cost_aware_router.py
b55108297547bbb34a6fc5dd387dc553f56d42ec3b01e383a62a743dea94373c  scripts/run_experiment3_skip_dense.py
a72a749920bdb8c29614e6b30b4500a6f8624b601cea0932ea2aa6881656a252  configs/experiment.yaml
7131099ebb321adae405e2d55ffbea57abb587a5bab5cd69f01493eab4bdd129  configs/hybrid.yaml
e376279e4ba2cfd9387da48b833662ce7623aacf46726115b827154a7ac44be1  configs/labels.yaml
2c77a686f59914f14f2a5a8a833e52aec7a3c644e6cf2d1108a7b47412f79605  configs/retrievers.yaml
47070a98de397fa278781a7af9f41f4a9d58353999110f96b9577cb05869b8fc  configs/router.yaml
65322ce517a06e51b6b89614c68a4d43798cb2b6ed74008339b32c3068b2974c  synthetic/nfcorpus.parquet (288-query set)
f31c78b6d80d4b452f6bdef96f1077b4a508be8904b7c444a622c2fd1d416ed6  synthetic/scifact.parquet
```
