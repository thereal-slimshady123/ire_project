# E4 amendment: skipping dense retrieval at matched skip rates, pre-registered validation

Written, hashed and committed on 2026-11-07, **before** the fresh datasets below were downloaded, built or
scored. The run happens once and is reported whatever the result.

## Why E4 is amended

The E4 in `phase2_explanatory_validation_20261107.md` reused Exp 3's §4.1 criteria (`improvements.md`),
including **skip rate ≥ 20% at a −0.005 training loss budget**. Under that rule the skip rate is an
*outcome* that mostly reflects how far BM25 trails hybrid on a dataset, not how well the router picks
queries. The original pre-registration already predicted that forum data would fail that criterion
(pilot CQADupStack 14.2–16.7%), and it did (validation 12.0%). So the run could not tell us whether
the router works. That E4 result stands as reported ("§4.1 not met, as predicted"). This amendment
asks a different question, on data nobody has looked at yet: **at a fixed operating point, does the
skip router pick the right queries?**

## Exploration evidence (data already seen; not part of the test)

Computed with `scripts/run_e4_matched_rate.py` (pinned below). Pooled, 20% skip rate:

| Data | n | Router − hybrid | Router − random | FOG router | FOG router − H1 | Would-be verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Pilot, 10 datasets | 6,928 | −0.0084 [−0.0103, −0.0066] | +0.0040 [+0.0022, +0.0057] | 0.154 [0.110, 0.195] | −0.037 [−0.086, +0.011] | INCONCLUSIVE |
| Phase 2 validation (mathematica, english, gaming) | 3,969 | −0.0067 [−0.0091, −0.0043] | +0.0106 [+0.0081, +0.0130] | 0.214 [0.163, 0.264] | −0.046 [−0.099, +0.006] | ACCEPTED |
| All 6 seen CQADupStack subforums | 6,246 | −0.0080 [−0.0101, −0.0060] | +0.0084 [+0.0064, +0.0104] | 0.199 [0.158, 0.240] | −0.051 [−0.094, −0.006] | INCONCLUSIVE |

Outputs: `results/niche/e4_matched_rate_pilot/`, `results/phase2_validation/e4_matched_rate_posthoc/`,
`results/niche/e4_matched_rate_forums_seen/`. On the pilot, a fixed 20% rate is too aggressive for fiqa
and cosqa (router − hybrid ≈ −0.03 and −0.02). A fixed rate moves the dataset dependence rather than
removing it, so the test below is limited to one domain (forums). The rate stays at §4.1's 20% target.
It was **not** lowered to a value the seen data would pass (e.g. 15%).

## Fresh held-out data

- **CQADupStack physics, programmers, gis** (`mteb/cqadupstack-*` on Hugging Face). Fixed rule, the same
  as for the Phase 2 validation set: the three CQADupStack subforums not yet used (exploration or
  validation) with the smallest `corpus.jsonl` (HF metadata, checked before download: physics
  33.3 MB, programmers 35.4 MB, gis 40.0 MB; then stats 46.7, wordpress 57.0, tex 93.8).
- Built with `scripts/build_e4_fresh_datasets.py`, which calls `build_niche_datasets.py`'s unchanged
  `beir_like` / `write_beir` on full corpora with the official test qrels.

## Fixed pipeline

1. `python scripts/build_e4_fresh_datasets.py`
2. `python scripts/run_experiment2_heldout_routing.py --datasets cqadupstack-physics cqadupstack-programmers cqadupstack-gis --output_dir results/e4_fresh/experiment2`
3. `python scripts/run_e4_matched_rate.py --datasets cqadupstack-physics cqadupstack-programmers cqadupstack-gis --exp2_dir results/e4_fresh/experiment2 --output_dir results/e4_fresh/e4_matched_rate`
4. (descriptive, for continuity) `python scripts/run_experiment3_skip_dense.py --datasets <same> --exp2_dir results/e4_fresh/experiment2 --output_dir results/e4_fresh/experiment3`

Method (unchanged from Exp 3 unless stated): BM25 (k1 = 1.2, b = 0.75) and `intfloat/e5-small-v2`, top-100;
5-fold CV per dataset, seed 42; hybrid α\* chosen per training fold; label "BM25 suffices" =
nDCG@10(BM25) ≥ nDCG@10(hybrid) − 0.02; Exp 3's 12 pre-dense features and LightGBM settings.
**Changed:** in each test fold, a method skips dense on the top round(r · n_fold) held-out queries by its
score. No test labels are used, and there is no loss-budget threshold. Heuristic direction = the sign with the
higher training-fold mean nDCG@10 at r = 20%. Random skipping = expected value at the same realised rate.
nDCG@10; paired bootstrap, 10,000 resamples, seed 42, **pooled over the 3 datasets**.

## E4′ — The skip router selects the right queries

**P1 (r = 20%).** (a) router − hybrid, 95% CI low ≥ −0.01; (b) router − random, 95% CI low > 0.
**P2 (curve).** Fraction of oracle gain captured, FOG = Σ_r (router_r − random_r) / Σ_r (oracle_r − random_r),
r ∈ {5%, 10%, …, 50%}, sums over queries; bootstrap 95% CI low > 0.

- **Accept** if P1a, P1b and P2 all hold.
- **Reject** if router − random CI high ≤ 0, **or** FOG CI high ≤ 0, **or** router − hybrid CI high < −0.01.
- **Inconclusive** otherwise.

**Secondary (reported, no verdict).** Router − H1 (BM25 top-1/top-2 ratio) at 20% and FOG(router) −
FOG(H1); H2 (max-IDF); per-dataset results; the full curve; Exp 3's §4.1 table on the fresh data.

**Predictions (from the exploration table above).** P1b and P2 will hold. P1a is borderline: on the six seen
subforums the point estimate is −0.0080, and with n ≈ 2,800 the CI low is expected near −0.0105. A
normal approximation with sampling noise only gives P(P1a holds) ≈ 0.25, so **INCONCLUSIVE is the more
likely verdict**, ACCEPTED is possible, and REJECTED is unlikely. Secondary: the router will **not** beat
H1 (FOG difference ≤ 0).

## Pinned files (SHA-256 at time of writing; listed in `SHA256SUMS.txt`)

See `e4_matched_rate_20261107.pins.txt` (written by `sha256sum` in the same commit).
