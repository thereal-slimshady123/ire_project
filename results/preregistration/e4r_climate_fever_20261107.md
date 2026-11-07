# E4-R — E4 re-tested with a corrected design on new data (pre-registered)

Written and hashed on 2026-11-07, **before** the Climate-FEVER corpus or queries were downloaded or scored. Read
beforehand: Hugging Face file sizes, and the judgment file `qrels/test.tsv` only, to count relevant documents per
query (1,535 queries; 1,391 have ≥ 2 relevant documents; mean 3.05). No retrieval was run. The pipeline
(`scripts/run_e4r_pipeline.sh`) runs once and the result is reported whatever it is. **The original E4 verdict
(inconclusive) stands and is reported alongside this one.**

## Why

E4 (`phase2_strengthening_20261107.md`) pooled the split-half test over six datasets. HotpotQA supplied 46% of the
queries and drove the pooled AUC′ to 0.750 (inconclusive), while every other fresh dataset was 0.53–0.60 and the
pre-declared check without HotpotQA gave 0.522. E8 then showed that multi-hop data can behave differently. In
hindsight, the claim should have been tested **per dataset, on data whose relevant documents are alternatives**,
not pooled across dataset types. That design flaw is corrected here, on data nobody has scored.

## Data (fixed rule)

The smallest unused mteb datasets with ≥ 300 queries having ≥ 2 relevant documents are reasoning- or code-based
collections (FreshStack, BRIGHT, R2MED), where relevant documents tend to be complements, so they would not test
this claim cleanly. **Climate-FEVER** (`mteb/climate-fever`) is the unused BEIR dataset with the most such queries
whose evidence pages independently support or refute a claim (alternatives). All test queries with a positive
judgment; corpus = judged documents plus a seed-42 reservoir sample of distractors, up to 50,000 documents. Built by
`scripts/build_e4r_dataset.py`.

## Test

`analyze_label_reliability.py`, unchanged; e5-small-v2; the same definitions as E4:
- (a) split-half reference AUC′ < 0.65 **and** 95% CI upper < 0.70;
- (b) split-half κ (3-way) < 0.20;
- (c) nDCG@20 stability κ ≥ 0.60.

**Accept** if (a), (b) and (c) all hold. **Reject** if AUC′ ≥ 0.75 **or** κ ≥ 0.40. **Inconclusive** otherwise.

**Prediction:** accepted. Every non-multi-hop dataset so far has been 0.52–0.60. Risk: some climate claims need
evidence from more than one page together (complementary), which could raise the AUC′.

## Pinned files

SHA-256 in `e4r_climate_fever_20261107.pins.txt`, and the hash of this file and of the pins file in
`SHA256SUMS.txt`.
