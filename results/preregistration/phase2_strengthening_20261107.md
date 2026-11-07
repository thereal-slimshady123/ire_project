# Phase 2 strengthening: mechanism, generality, and the search for where routing can win (pre-registered)

Written and hashed on 2026-11-07, **before** any of the fresh datasets below were downloaded, built or scored. The
pipeline (`scripts/run_strengthening_pipeline.sh`) runs once and every result is reported, whatever it is.

## Status going in

| | Status (held-out, pre-registered) |
| :--- | :--- |
| E1 distribution shift | Accepted (CQADupStack mathematica/english/gaming) |
| E2 no pre-retrieval-identifiable niche | Accepted (same) |
| E3 skip dense, §4.1 criteria | Not met, as predicted (skip rate 12.0%; 13.2% on physics/programmers/gis) |
| E3′ skip dense at matched 20% rate | Inconclusive, as predicted (physics/programmers/gis) |

This document adds four hypotheses (E4–E7) and replications of E2, all tested on data nobody has scored yet.

## Exploration (data already seen, not part of any test)

All on the 16 datasets with cached runs (Phase 1, niche exploration, both earlier validation sets; outputs in
`results/strengthening/explore/`). Every exploratory analysis that was run is listed here, including the ones that
did not work.

1. **Label reliability** (`label_reliability/`, 16 datasets, 13,697 queries). The winner label barely changes under
   measurement changes: κ (3-way) 0.85 / 0.86 for α ∓ 0.1, 0.82 / 0.84 for nDCG@5 / @20, 0.86 for MRR@10. It
   collapses when the relevance judgments are split in half at random: κ 0.032. The half-A margin predicts the
   half-B binary winner with AUC′ **0.569 [0.543, 0.594]**, about the same as the best query feature (0.597 on the
   validation set). 72.6% of queries have a single relevant document. Swapping in e5-base (7 datasets) gives κ 0.50
   (58% of binary winners retained).
2. **Pooling artefact.** On the 7 e5-base exploration datasets, E2(c) as defined (raw features pooled across
   datasets) gives best AUC′ 0.702 for e5-base **and 0.704 for e5-small**, driven by `bm25_top1_score`, whose scale
   differs between corpora. With within-dataset percentile features it is 0.557 (e5-base) / 0.571 (e5-small). On
   the homogeneous validation forums both definitions agree (0.597 / 0.598).
3. **E2 with e5-base** (7 seen datasets): no niche; router − hybrid −0.0000 [−0.0009, +0.0009]; AUC′ as in (2).
4. **Niche search** (`niche_discovery/`, 19 datasets, 14,004 queries): 5,616 niches × 13 actions = 73,008
   candidates; 682 eligible (n ≥ 200, ≥ 5 datasets, positive in both random halves of the datasets). A shuffle null
   (feature rows permuted within dataset, 5 runs) gives 119–248 eligible, with a best 95% LB of −0.0000 to +0.0038.
   The real best LB is +0.0040, so there is some structure, but the best single candidate is barely above the null.
   Every surviving niche changes **α**, never "switch to one retriever": lean dense when BM25 is unsure, lean BM25
   when BM25 is confident.
5. **Per-document learned fusion** (`learned_fusion*/`, 16 datasets). Within-dataset 5-fold CV loses to tuned
   hybrid for all three feature sets (all −0.0066, doc −0.0062, doc+query −0.0080 pooled). Trained across datasets
   (leave-one-dataset-out, vs hybrid at the source α): doc +0.0001 [−0.0012, +0.0014], **doc+query +0.0019
   [+0.0005, +0.0034]** (the "all" feature set was not run across datasets). A frozen doc+query model trained on 13 datasets and applied to physics/programmers/gis:
   +0.0022 [−0.0007, +0.0053] vs source α, +0.0033 [−0.0001, +0.0068] vs target-tuned α*. Five variants were run
   and one was selected, so a winner's curse is expected.
6. **Learning curve** (arguana, 1,401 queries): router − hybrid −0.0169 → −0.0037 and learned winner AUC′
   0.552 → 0.639 from 250 to 1,051 training queries. It is still rising.
7. **Cost** (`cost_frontier/`, physics/programmers/gis, CPU). Always-hybrid 21.9 ms per query. At a 20% skip rate,
   the learned router takes 25.5 ms (slower than hybrid: it breaks even only at a 37.3% skip rate), while H1 (BM25
   top-1/top-2 ratio) takes 17.6 ms (−19.3%, break-even at 0.06%). Quality at 20%: H1 − hybrid −0.0071
   [−0.0104, −0.0038], H1 − random +0.0086 [+0.0053, +0.0117], router − H1 −0.0029 [−0.0067, +0.0008]. GPU timings
   are nearly identical (e5-small at batch size 1).

## Fresh held-out data (selected by fixed rules from Hugging Face metadata only; nothing downloaded)

- **Forums:** CQADupStack **stats, wordpress, tex**. These are all the remaining unused subforums. Full corpora,
  official test qrels.
- **Non-forum:** the unused BEIR datasets on `mteb/` with ≥ 300 test queries, taking the three smallest
  `corpus.jsonl`: **scidocs** (33.0 MB), **nq** (1,460.9 MB), **hotpotqa** (1,790.0 MB). Next was dbpedia
  (1,792.1 MB); trec-covid was excluded with 50 queries. All test queries with a positive judgment are used. Corpus =
  every judged document plus a seed-42 reservoir sample of distractors, up to 50,000 documents (the Phase 1 quora
  protocol).
- Built by `scripts/build_strengthening_datasets.py`. Pooled n ≈ 16,000 queries (hotpotqa ≈ 7,400 of them).

## Fixed pipeline

`bash scripts/run_strengthening_pipeline.sh` (pinned below), with unchanged settings everywhere: BM25
(k1 = 1.2, b = 0.75); e5-small-v2 (and e5-base-v2 in step 8); top-100; min-max fusion; 5-fold CV by query, seed 42;
paired bootstrap 10,000 resamples, seed 42; nDCG@10; δ = 0.005.

## E2 replications (same claim and rules as the original E2)

Run three times: fresh forums (e5-small), fresh non-forum (e5-small), all six (e5-base).
**One amendment, decided from exploration item 2 before seeing fresh data:** E2(c)'s verdict uses **within-dataset
percentile** features. The original raw-feature AUC′ is always reported alongside. Reason: on mixed pools the raw
version measures which dataset a query comes from. On homogeneous data the two agree.
**Prediction:** E2 accepted in all three.

## E4 — Mechanism: the winner is a property of the query–document pair, not of the query

**Claim.** Which retriever "wins" a query is decided mostly by which relevant documents happen to be judged, not by
the query. So no query-level signal can predict the winner much better than chance, which explains E2(c) and the
router's failure.

On the six fresh datasets, pooled (e5-small), from `analyze_label_reliability.py`:
- (a) **qrels split-half reference AUC′** (half-A dense − BM25 margin → half-B binary winner; queries resampled
  whole): point estimate < 0.65 **and** 95% CI upper < 0.70;
- (b) **qrels split-half κ** (3-way, query-weighted over datasets) < 0.20;
- (c) **measurement-stability contrast:** κ for nDCG@20 vs nDCG@10 ≥ 0.60. This shows the instability comes from
  which documents are relevant, not from metric noise.

**Accept** if (a), (b) and (c) all hold. **Reject** if the split-half AUC′ ≥ 0.75 **or** the split-half κ ≥ 0.40.
**Inconclusive** otherwise.
Secondary: the same without hotpotqa (it supplies most split-half queries); per dataset; the e5-base model swap
(κ, retention); the single-relevant-document share. **Prediction:** accepted.

## E5 — A frozen query niche where something other than fixed-α hybrid wins

The four niches frozen in `results/strengthening/explore/niche_discovery/frozen_niches.json` (top 3 distinct by
95% LB, plus the best with n ≥ 1,000):

1. `bm25_top1_score` bottom 10% AND `bm25_top1_top2_ratio` bottom 10% → hybrid α = 0.7
2. `bm25_top1_top2_gap` top 10% AND `dense_top1_score` bottom 33% → hybrid α = 0.5
3. `mean_idf` bottom 33% AND `entity_density` top 10% → hybrid α = 0.7
4. `bm25_top1_top2_gap` top 33% AND `bm25_score_entropy` bottom 20% → hybrid α = 0.5

Percentiles are computed within each fresh dataset (no labels used). Baseline = each dataset's best fixed α (tuned
on all of its queries, the strongest fixed hybrid). Test: action − hybrid\* inside the niche, pooled over the six
datasets, paired bootstrap, Holm across the four niches.
**A niche is supported** if Holm p < 0.05 and CI low > 0. **E5 accepted** if any niche is supported, **rejected** if
none is.
**Prediction:** power per niche ≈ 0.50–0.74 if the explored effect is real, ≈ 0.11–0.18 if it is half that size
(winner's curse). Most likely outcome: **rejected** (no niche supported).

## E6 — Per-document learned fusion beats fixed-α hybrid

**Claim (design implication of E4).** If wins are decided per document, the BM25/dense decision should be made per
document. Frozen system: one LightGBM LambdaRank model (router.yaml GBM settings; doc+query features, i.e. min-max
scores, ranks, retrieved-by flags, rank difference, and the 18 query features) trained on all 16 exploration
datasets, applied unchanged to the six fresh datasets.
Primary: learned fusion − hybrid at the source α (best pooled α on the 16 training datasets; same information),
pooled.
**Accept** if CI low > 0; **reject** if CI high < 0; **inconclusive** otherwise.
Secondary: vs hybrid with α\* tuned on the fresh dataset's own training folds (uses target labels); per dataset.
**Prediction:** effect ≈ +0.002; with n ≈ 16,000, P(accept) ≈ 0.5.

## E7 — A free skip rule is at least as good as the learned skip router

At a 20% skip rate, pooled over the six fresh datasets (`run_e4_matched_rate.py` per-query file, plus
`run_cost_frontier.py` timings):
- (a) H1 − router: 95% CI low > −0.005 (non-inferior);
- (b) H1 − random skipping: CI low > 0;
- (c) measured CPU latency of H1 < always-hybrid, and < the learned router.

**Accept** if (a), (b) and (c) all hold; **reject** if H1 − router CI high < −0.005 or H1 − random CI high ≤ 0;
**inconclusive** otherwise.
Secondary (no verdict): H1 − hybrid against −0.01 (predicted borderline on forums, failing on non-forum where BM25
trails dense more); GPU timings; break-even rates. **Prediction:** accepted.

## Learning curve (descriptive, prediction recorded)

hotpotqa, fixed 25% test set, training sizes 250 … 4,000 and all (≈ 5,550). **Prediction:** at the largest size,
router − hybrid 95% CI upper < +0.005 (still no gain), and the AUC′ gain (largest − 1,000) < 0.05. Exploration
(arguana) showed a rising curve, so this is uncertain.

## Pinned files

SHA-256 in `phase2_strengthening_20261107.pins.txt`, and the hash of this file and of the pins file in
`SHA256SUMS.txt`.
