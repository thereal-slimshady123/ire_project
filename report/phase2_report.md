# Why Per-Query Retriever Routing Fails, and What Works Instead

## Abstract

Phase 1 found that a learned router choosing BM25, dense or hybrid retrieval per query does not beat a tuned
fixed-α hybrid. Phase 2 asks why, and what follows. Every confirmatory test was pre-registered (written and hashed
before its data was downloaded) and run once on held-out datasets selected by fixed rules: 16 held-out datasets over
six domains (forums, scientific literature, open-domain QA, multi-hop QA, entity search, fact-checking), with two
dense encoders. Three results stand out. (1) The null result is robust: no query group identifiable
before retrieval is better served by a single retriever, and the router gains nothing (replicated on fresh forum and
non-forum data and with e5-base). (2) The reason is that retrieval "wins" are decided per document, not per query:
splitting a query's relevant documents in half, the winner on one half barely predicts the winner on the other
(reference AUC′ 0.52 on fresh non-multi-hop data and 0.507 in a corrected re-test on Climate-FEVER, where 0.5 is
chance). Some multi-hop datasets are an exception. (3) Acting on this, a per-document learned fusion model, trained
once on 16 exploration datasets and applied unchanged, beats fixed-α hybrid pooled over six fresh datasets
(+0.0029 [+0.0019, +0.0038] nDCG@10; positive on all six, individually significant on two) at about 5 ms extra per
query. Effects are small, and four of thirteen pre-registered predictions missed; all are reported.

## 1. From Phase 1 to Phase 2

Phase 1 (`report/report.md`) tested four hypotheses on NFCorpus, SciFact, FiQA, CQADupStack-Unix and a Quora subset:

| | Hypothesis | Outcome |
| :--- | :--- | :--- |
| H1 | Query-adaptive routing beats fixed-α hybrid and RRF | Not supported vs hybrid (best router −0.0012 [−0.0025, +0.0002]); beats RRF, but only because tuned hybrid does |
| H2 | Recall@100 matches hybrid | Supported (equivalence within ±0.01) |
| H3 | Budgeted cross-encoder escalation keeps most of the rerank gain | Not supported (16% retained at a 10% budget) |
| H4 | Latency stays near hybrid | Partly (router +5–13 ms; escalation 2.5–7× hybrid) |

The mid-submission proposed tentative Phase 2 hypotheses (P2-H1–H4). Exploration refined P2-H1 into E2 and P2-H2 into
E1; P2-H3 (one α is a poor compromise) was folded into the niche search (E5); P2-H4 (U-shaped gain with style mix)
was not carried forward.

## 2. Method

**Retrieval.** BM25 (k1 = 1.2, b = 0.75); dense `intfloat/e5-small-v2` (and `e5-base-v2` for robustness); top-100;
hybrid = α·dense + (1 − α)·BM25 over min-max scores, α tuned on training folds. Metric nDCG@10; 5-fold CV by query,
seed 42; paired bootstrap with 10,000 resamples (2,000 for the AUC′ confidence intervals in E2(c), E4, E4-R and E8);
practical margin δ = 0.005.

**Naming.** The first pre-registration (`phase2_explanatory_validation_20261107.md`) calls the skip-dense test "E4",
and `e4_matched_rate_20261107.md` amends it. In this report those are **E3** and **E3′**, and **E4** is the split-half
test of `phase2_strengthening_20261107.md`. Result tables written by `run_e4_matched_rate.py` keep the old title
"E4 (amended)".

**Pre-registration.** Each confirmatory test has a file in `results/preregistration/` stating the data rule, the
exact criteria (accept / reject / inconclusive) and a prediction. The file and every script it uses are hashed in
`SHA256SUMS.txt` before the test data is downloaded; each test set is used once. Exploration on already-seen data is
disclosed in the same files, including variants that did not work. The pre-registration files were committed to git
together with the results (commit 840350b), so the repository history alone cannot show that they were written
first; the hashes show only that neither the files nor the pinned scripts changed afterwards.

**Data.**

| Role | Datasets |
| :--- | :--- |
| Exploration (seen) | 19 sets: NFCorpus, SciFact, FiQA, Quora, ArguAna, WANDS, CoSQA, BRIGHT biology/economics/psychology, CQADupStack unix/android/webmasters, mathematica/english/gaming (after their validation run), physics/programmers/gis (after E3′). The E6 model and the E4 exploration use the 16 of them without BRIGHT |
| Validation (E1–E3) | CQADupStack mathematica, english, gaming (3,969 queries) |
| Fresh (E2-R, E4–E7) | Forums: CQADupStack stats, wordpress, tex. Non-forum: SciDocs, NQ, HotpotQA (15,956 queries) |
| Boundary (E8) | MuSiQue, 2WikiMultihopQA (multi-hop); DBpedia-Entity (control) |
| Corrected E4 (E4-R) | Climate-FEVER |

Fresh datasets were chosen by fixed rules from Hugging Face metadata (e.g. "the unused CQADupStack subforums";
"the three smallest unused BEIR corpora with ≥ 300 test queries"). Large corpora keep every judged document plus a
seeded random sample of distractors up to 50,000 documents.

## 3. Why routing fails

**E1 — Distribution shift. Accepted.** A router trained on 288 synthetic queries, built so the winning retriever is
obvious (feature AUC′ 1.000), is worse than hybrid on natural queries: −0.0209 [−0.0256, −0.0162]. On natural
queries the best feature predicts the winner with AUC′ 0.597.

**E2 — No pre-retrieval-identifiable niche. Accepted, replicated three times.** Across groups defined by dataset,
query length, question form and identifier tokens (Holm-corrected), neither BM25 nor dense beats hybrid anywhere.
The cost-aware router stays within ±0.002 of hybrid, and the best feature AUC′ stays below 0.60:

| Data | Router − hybrid | Best AUC′ (within-dataset percentiles) | Verdict |
| :--- | :---: | :---: | :---: |
| Validation forums | +0.0001 [−0.0004, +0.0005] | 0.598 | Accepted |
| Fresh forums | −0.0011 [−0.0019, −0.0002] | 0.583 | Accepted |
| Fresh non-forum | −0.0004 [−0.0008, −0.0001] | 0.588 | Accepted |
| All fresh, e5-base | −0.0003 [−0.0006, +0.0000] | 0.571 | Accepted |

*Pooling artefact.* Pooling raw features across heterogeneous datasets inflates AUC′ (0.70 on seven exploration
datasets, 0.633 on fresh non-forum) because some features, such as the top BM25 score, mainly identify the dataset.
Within-dataset percentiles remove this; the change was pre-registered before the fresh data was scored.

**E4 — Mechanism: wins are decided per document. Inconclusive overall; holds on every non-multi-hop dataset.**
For each query with ≥ 2 relevant documents, the relevant set is split in half at random (5 times). If the winning
retriever were a property of the query, both halves would agree. The half-A dense − BM25 margin is used to predict
the half-B winner (reference AUC′):

| Data | Split-half AUC′ | Split-half κ | nDCG@20 κ (control) |
| :--- | :---: | :---: | :---: |
| Exploration, 16 datasets | 0.569 [0.543, 0.594] | 0.03 | 0.84 |
| Fresh, all six (pre-registered primary) | **0.750 [0.730, 0.769]** | −0.07 | 0.89 |
| Fresh without HotpotQA (pre-declared secondary) | 0.522 [0.501, 0.559] | 0.03 | 0.87 |
| Corrected test, Climate-FEVER (E4-R) | **0.507 [0.501, 0.556]** | −0.00 | 0.76 |

Measurement choices barely move the label (κ 0.80–0.89 for α ± 0.1, nDCG@5/@20), so the instability comes from
which documents are relevant, not from metric noise. Swapping e5-small for e5-base (κ 0.47) turns 44% of
dense/BM25 winners into "no clear winner", but flips only 1.8% to the other retriever, and the e5-small margin still
predicts the e5-base winner with AUC′ 0.95: the encoder swap mostly changes *whether* there is a clear winner, not
*which* retriever it is.
The pooled primary failed its AUC′ < 0.65 criterion because HotpotQA (46% of fresh queries, AUC′ 0.838) dominated
it; on SciDocs, NQ, stats and tex the per-dataset AUC′ was 0.53–0.60 (wordpress had too few eligible queries to
compute it). The pooled AUC′ was 0.7498, just below the 0.75 reject threshold. Pooling across dataset types was a
design flaw. It is corrected in E4-R (§6), which was accepted on new data. E4's own verdict stays inconclusive.
The split-half test needs at least two relevant documents, so it covers only part of each dataset (74–84% of
forum queries have a single relevant document); for those queries the per-document explanation is untested.

**E8 — Boundary condition: multi-hop data. Inconclusive; partly supported.** Prediction: when relevant documents are
complements (all needed, as in multi-hop QA) rather than alternatives, the winner is more query-dependent.

| Group | Split-half AUC′ |
| :--- | :---: |
| Multi-hop (MuSiQue + 2Wiki) | 0.690 [0.664, 0.715] (needed ≥ 0.70) |
| Alternatives (DBpedia) | 0.600 [0.549, 0.651] |
| Difference | +0.090 [+0.032, +0.145] |

2WikiMultihopQA (0.819) behaves like HotpotQA; MuSiQue (0.601), built to block reasoning shortcuts, does not.
Multi-hop data is more query-dependent on average, but the boundary is dataset-specific, not clean. The multi-hop
group pools two datasets that behave differently, the same issue E4-R corrects for E4.
Both pre-registered secondary predictions failed. The learned skip router was predicted to beat H1 on the multi-hop
group, as on HotpotQA: router − H1 at a 20% skip rate was +0.0014 [−0.0001, +0.0029] on multi-hop, but
+0.0166 [+0.0021, +0.0318] on DBpedia, the control. The E2(c) feature AUC′ was predicted to be higher on multi-hop
data: 0.642 vs 0.643 (`results/strengthening/boundary/e8/tables.md`).
Disclosure: one team member had run an exploratory per-source α check on DBpedia-Entity on 2026-11-05, before this
pre-registration (no split-half analysis; files not used here). DBpedia was therefore not entirely unseen by the team.

**Learning curve (HotpotQA).** More training queries make the router better at *predicting* the winner (AUC′
0.632 → 0.749 from 1,000 to 5,554 queries) but not better at *ranking*: router − hybrid goes from −0.0025 to −0.0060.
The pre-registered prediction had two parts: still no gain at the largest size held (CI upper −0.0037 < +0.005), but
"AUC′ gain < 0.05" missed (+0.118 [+0.070, +0.166]; `results/strengthening/fresh/learning_curve/tables.md`).

## 4. What works instead: per-document fusion (E6, accepted)

If wins are decided per document, the BM25/dense decision should be made per document. A LightGBM LambdaRank model
scores each candidate in the union of the BM25 and dense top-100 using its normalised scores, ranks, retrieved-by
flags and the query's 18 router features. It was trained once on the 16 exploration datasets and applied unchanged
to the six fresh datasets. Exploration had shown that the same model trained within a single dataset loses to hybrid
(−0.006 to −0.008), so it needs cross-dataset training data.

| Dataset | Learned − hybrid (source α) | Learned − hybrid (α tuned on target) |
| :--- | :---: | :---: |
| stats | +0.0011 [−0.0036, +0.0061] | +0.0011 |
| wordpress | +0.0055 [−0.0020, +0.0132] | +0.0082 |
| tex | +0.0016 [−0.0013, +0.0043] | +0.0031 |
| scidocs | +0.0013 [−0.0013, +0.0038] | +0.0013 |
| nq | +0.0043 [+0.0018, +0.0067] | −0.0040 |
| hotpotqa | +0.0029 [+0.0019, +0.0040] | +0.0021 |
| **pooled** | **+0.0029 [+0.0019, +0.0038]** | +0.0011 [+0.0000, +0.0021] |

Against the pre-registered baseline the point estimate is positive on all six datasets, but individually only NQ and
HotpotQA have confidence intervals above zero; the accepted claim is the pooled one. Against the stronger baseline,
which tunes α on each target's own labels, the pooled gain is smaller and only just above zero, and NQ is
significantly worse (−0.0040 [−0.0070, −0.0010]). Cost (CPU medians,
`results/strengthening/fresh/fusion_latency/tables.md`): query features 2.1 ms + candidate features 2.6 ms + model
1.4 ms, replacing hybrid's 0.9 ms min-max fusion step, so 27.9 ms against 22.7 ms for hybrid (+23%). The technique is learning-to-rank over fusion
features and is not new; the contribution is the evidence for why it beats routing.

**E5 — Query niche. Rejected (as predicted).** Of 73,008 niche × action candidates searched on exploration data, four
were frozen. None survived Holm correction on fresh data (closest: +0.0115 [+0.0016, +0.0229], Holm p = 0.14). A
shuffle null had shown the best exploration candidate was barely above chance.

## 5. Efficiency: skipping dense retrieval

| | Claim | Verdict |
| :--- | :--- | :--- |
| E3 | ≥ 20% of queries skippable at ≤ 0.01 loss (Phase 1 §4.1 criteria) | Not met, as predicted (12–13% on forums) |
| E3′ | At a fixed 20% skip rate the router picks the right queries | Inconclusive, as predicted: beats random (+0.0056), costs −0.0100 vs hybrid |
| E7 | A free rule (BM25 top-1/top-2 ratio, H1) is as good as the learned skip router | Inconclusive: H1 − router −0.0047 [−0.0058, −0.0036] pooled; H1 ≥ router on 5 of 6 datasets, but not HotpotQA (+0.0131 for the router) |

Measured CPU latency (fresh datasets): hybrid about 23 ms per query. The learned skip router's overhead (≈ 7.8 ms) means
it is slower than hybrid unless it skips more than about 35% of queries (33–38% per dataset). H1 costs ≈ 0.01 ms and
saves about 19% latency at a 20% skip rate for −0.0084 nDCG@10 pooled. One CPU measurement was corrupted by the laptop
suspending (one HotpotQA query recorded at 56.7 minutes). The pre-registered table
(`results/strengthening/fresh/cost_frontier/tables_cpu.md`) averages over it, so its HotpotQA and pooled latency rows
are wrong (e.g. pooled hybrid 236 ms). The numbers above come from a post hoc recomputation without that one query
(`tables_cpu_clean_posthoc.md`, `scripts/posthoc_cost_frontier_clean.py`); medians give the same picture, the GPU run
was unaffected, and no verdict changes.

## 6. Corrected E4 (E4-R)

E4's pooled design let one multi-hop dataset dominate. The corrected design tests the claim on a single dataset
whose relevant documents are alternatives, chosen by a fixed rule and pre-registered before download
(`results/preregistration/e4r_climate_fever_20261107.md`). Climate-FEVER evidence pages each support or refute a claim
independently; 1,391 of its 1,535 test queries have ≥ 2 relevant documents.

| Criterion | Result | |
| :--- | :---: | :---: |
| (a) split-half AUC′ < 0.65, CI upper < 0.70 | 0.507 [0.501, 0.556] | Pass |
| (b) split-half κ < 0.20 | −0.004 | Pass |
| (c) nDCG@20 stability κ ≥ 0.60 | 0.759 | Pass |

**E4-R: accepted (as predicted).** With alternative relevant documents, the winning retriever on one half of a
query's relevant documents predicts the winner on the other half at chance level, while the label is stable under a
change of cut-off. The original E4 verdict (inconclusive) is not changed by this; both are reported.

## 7. Prediction record

| Test | Verdict | Predicted |
| :--- | :--- | :---: |
| E1 | Accepted | ✓ |
| E2 (validation; fresh forums; fresh non-forum; e5-base) | Accepted ×4 | ✓ |
| E3 / E3′ | Not met / inconclusive | ✓ |
| E4 | Inconclusive | ✗ (predicted accepted) |
| E5 | Rejected | ✓ |
| E6 | Accepted | ✓ (predicted about 50/50) |
| E7 | Inconclusive | ✗ (predicted accepted) |
| E8 | Inconclusive | ✗ (predicted accepted) |
| E4-R (corrected E4, Climate-FEVER) | Accepted | ✓ |
| Learning curve (HotpotQA, descriptive) | No gain held; AUC′ kept rising (+0.118) | ✗ (predicted AUC′ gain < 0.05) |

9 of 13 predictions correct. All four misses involve multi-hop data (HotpotQA; MuSiQue). E8's two secondary
predictions also failed (§3).

## 8. Limitations

- Effects are small: +0.003 nDCG@10 for per-document fusion, about +0.001 against a target-tuned hybrid.
- One encoder family (e5), one fusion form (min-max convex), one candidate depth (top-100).
- Pooled tests are dominated by the largest dataset (HotpotQA); E4's pooled design was a mistake (§3).
- E7's non-inferiority margin (0.005) was stricter than the project's other margins (0.01); with 0.01 it would have
  passed. It is reported as pre-registered.
- Non-forum corpora were capped at 50,000 documents (judged documents plus random distractors), which makes
  retrieval easier than on the full collections.
- Learned fusion was deliberately kept untuned (default settings, no cross-encoder features); its gain is likely a
  lower bound.
- Latency was measured on one laptop, single-query, CPU.
- The split-half test (E4, E4-R, E8) only uses queries with at least two relevant documents. Climate-FEVER's
  judgments are known to be noisy, and noisy judgments alone would also push its split-half AUC′ towards 0.5.
- The E5 paper reports fine-tuning e5 on MS MARCO, NQ and NLI data, so NQ is not new to the dense encoder.
- Pre-registrations and results were committed together (§2).

## 9. Conclusion

Per-query routing between BM25, dense and hybrid retrieval does not beat a tuned fixed-α hybrid. This held on all
nine held-out datasets where it was formally tested (forums, scientific literature, open-domain and multi-hop QA),
with two encoders. The main reason is that which retriever wins is decided
largely per document, not per query, so query-level signals have little to learn. The exception is some multi-hop
data, where relevant documents are complements. Acting on the per-document view, a fusion model that decides per
document beats fixed-α hybrid pooled over the fresh datasets (positive on all six, significant on two), by a small
margin at a moderate latency cost.

## Reproducibility

- Pre-registrations and hashes: `results/preregistration/`.
- Pipelines: `scripts/run_strengthening_pipeline.sh`, `scripts/run_boundary_pipeline.sh`, `scripts/run_e4r_pipeline.sh`.
- Outputs: `results/strengthening/` (exploration in `explore/`, confirmatory runs in `fresh/`, `boundary/`, `e4r/`).
- Raw downloads were deleted after the BEIR-format corpora were built; the builders re-download them if needed.
