# Open problems that need a decision

Items the code cannot settle on its own. Each one lists the options, a recommendation, and its current status.

Status on 2026-10-31: **resolved** 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13 · **partly resolved** 3 (hardening tried, not effective) · **open** 12.

## 1. NFCorpus has only 868 synthetic queries

- **Evidence:** 3,128 of NFCorpus's 3,633 passages are relevant to some test query, so the leakage rule excludes them. That leaves 434 eligible passages and a capacity of 868 queries at 50/50 style split.
- **Options:**
  - a. Accept the cap and state it in the report.
  - b. Drop NFCorpus from the synthetic set and say so.
  - c. Relax the leakage rule for NFCorpus. Not recommended: it breaks the test-isolation guarantee.
- **Recommendation:** (a). It is honest and needs no code change.
- **Status: resolved (a).** The cap stands; the sweep reports realised counts (item 11).

## 2. FiQA makes up 68% of the 30k budget

- **Evidence:** With the current equal-split-and-redistribute allocation, FiQA gets 20,404 of 30,000 queries. NFCorpus gets 868 and SciFact 8,728. The pooled scaling curve is therefore mostly FiQA.
- **Options:**
  - a. Keep the current allocation and report the per-dataset shares.
  - b. Cap every dataset at its capacity and report the shortfall, so the total is less than 30k.
  - c. Balance by sampling FiQA down to the smallest dataset's size.
- **Recommendation:** (a) for the first run, since the pooled scaling curve is already reported per dataset.
- **Status: resolved (a).**

## 3. Lexical synthetic queries are nearly solved by BM25

- **Evidence:** On a 600-query SciFact smoke run, BM25 gets nDCG@10 0.999 on lexical queries. Natural SciFact BM25 is 0.683 (`improvements.md` §1.7). Each lexical query is made from its source passage's rarest words, so the source passage is almost always the top BM25 hit.
- **Risk:** The router may learn "lexical means BM25", which will not hold on natural queries. Any gain from synthetic data could be an artifact.
- **Options:**
  - a. Make the generator harder: drop a random subset of the chosen terms, or add a distractor-free paraphrase constraint.
  - b. Keep the generator and report synthetic-only and natural-only results separately, with the mismatch stated.
  - c. Validate with a small hand-checked or LLM-judged natural-style sample before scaling.
- **Recommendation:** (c) first, then (a) if the gap is still large.
- **Status: partly resolved.** (a) is implemented as an opt-in: `ExtractiveLexicalGenerator(pool_factor=...)` / `generate_synthetic_queries.py --lexical_pool_factor` samples the terms from the `pool_factor × n` rarest tokens instead of taking the n rarest (default 1 = old behaviour). **It does not fix the problem.** A 600-query SciFact lexical smoke run gives BM25 nDCG@10 0.999 at pool factor 1 and 0.998 at pool factor 3 (dense 0.729 → 0.701; BM25 beats hybrid on 4.8% → 6.0% of queries). Any handful of a SciFact passage's rarest tokens identifies it. Making lexical queries realistically hard needs a different generator, e.g. seq2seq keyword queries (`--lexical_generator seq2seq`, GPU) or mixing in mid-IDF terms. (c) still needs a human or LLM judge. Until then, report synthetic and natural results separately (option b).

## 4. Synthetic labels use one relevant document, natural labels use many

- **Evidence:** A synthetic query's only relevant document is its source passage (`synthetic/builder.py`). Natural qrels are multi-relevant: about 38 judgments per test query on NFCorpus (12,335 over 323), 2.6 on FiQA (1,707 over 648), 1.1 on SciFact (340 over 300).
- **Risk:** nDCG@10 values are on different scales, so the advantage targets are not directly comparable across the two sources.
- **Options:**
  - a. Accept the difference and document it.
  - b. Add a second relevant document per synthetic query, for example a near-duplicate passage. This is hard to do cleanly.
- **Recommendation:** (a). Revisit only if item 3 validation shows a problem.
- **Status: resolved (a).** Documented here and in the report (§4.2).

## 5. MRR-blended training target (0.6·nDCG@10 + 0.4·MRR@10)

- **Evidence:** Specified in `improvements.md` §2.3. No code implements it. The committed cache holds nDCG only.
- **Options:**
  - a. Implement it. For natural queries, recompute MRR from the run files, which exist locally for NFCorpus, SciFact and FiQA. For synthetic queries, MRR is the reciprocal rank of the source passage and can be added in `build_route_table`.
  - b. Drop it from the plan and keep nDCG-only targets.
- **Recommendation:** (a) only if you want to test the smoothing effect. Otherwise (b).
- **Status: resolved (a), natural queries.** Experiment 2d's "blended target" router recomputes MRR@10 from the saved runs (all five datasets) and trains on 0.6·nDCG@10 + 0.4·MRR@10. It does not help: pooled −0.0016 [−0.0029, −0.0003] vs always-hybrid (`results/experiment2d/tables.md`). Synthetic MRR is not added, since synthetic scaling has not been run (item 12).

## 6. 384-d query embedding features

- **Evidence:** `improvements.md` §3.5 lists them. `plan.md` Track 2 does not. No code implements them. Only the pilot script `scripts/demo_synthetic_routing.py` uses embeddings.
- **Options:**
  - a. Add them to Track 2: encode queries with the dense model and add them to the feature matrix.
  - b. Drop them and update `improvements.md` §3.5.
- **Recommendation:** Decide with the plan owner. Adding them increases the feature count from 18 to about 400, which is a large change for a weak-signal router.
- **Status: resolved.** Experiment 2c (`scripts/run_experiment2c_embedding_router.py`) adds the e5 query embedding reduced to 32 PCA components (fitted on training folds) to the 18 features: pooled −0.0040 [−0.0064, −0.0017]. With the Experiment 2d cost-aware router: −0.0000 [−0.0009, +0.0008].

## 7. Contract 2 wording does not match the code

- **Evidence:**
  - `plan.md` Contract 2 says `rank_correlation` is Kendall tau. The code computes Spearman (`features/retrieval_signal_features.py:46`).
  - Contract 2 says the top-1 scores are normalized. The code stores raw retriever scores (`features/feature_builder.py:150-151`).
- **Options:**
  - a. Change the contract text to Spearman and raw scores.
  - b. Change the code to Kendall and normalized scores, then regenerate the features.
- **Recommendation:** (a). It is free, and the committed experiments already use Spearman and raw scores.
- **Status: resolved (a).** `plan.md` Contract 2 now says Spearman and raw scores.

## 8. The "pre-cached runs" claim in `plan.md`

- **Evidence:** `plan.md:43` says base runs and indexes are cached in the repo. `results/runs/`, `results/indexes/` and `data/raw/` are gitignored, and only the five `results/experiment2/cache/*.csv` files are tracked.
- **Options:**
  - a. Commit `results/runs/` (large, mostly `.trec` and `.parquet` files).
  - b. Add a script that regenerates runs from `data/raw` and make `plan.md` say so.
  - c. Store runs outside git, with a download link, and point `plan.md` at it.
- **Recommendation:** (b). It keeps the repo small and makes the claim true after one command.
- **Status: resolved (b).** `plan.md` now says what is committed (the per-query route tables) and gives the regeneration commands: `run_retrieval_baselines.py` (NFCorpus, SciFact, FiQA), `run_experiment1b_query_types.py` (Unix, Quora subset), then `export_run_parquet.py`.

## 9. Canonical run format: `.trec` or parquet

- **Evidence:** Contract 1 says parquet. The experiment scripts write `.trec`. `run_retrieval_baselines.py` writes parquet as well. The loaders accept both.
- **Options:**
  - a. Make parquet canonical and have experiment scripts write it too.
  - b. Make `.trec` canonical and change Contract 1.
- **Recommendation:** (a). It matches the contract and is typed.
- **Status: resolved (a).** Parquet is canonical. `scripts/export_run_parquet.py` writes Contract 1 `bm25.parquet` / `dense.parquet` from the cached `.trec` runs for every dataset (it filled in Unix and Quora); the experiment scripts keep `.trec` as their cache.

## 10. Report Sections 2 and 4.3 are not drafted

- **Evidence:** `report/report.md` has no advantage-regression methodology section and no feature-ablation or synthetic-scaling section.
- **Decision needed:** Who writes them, and when. The numbers depend on items 1–4 above.
- **Status: resolved (drafted).** Report §2.4 (advantage regression and cost-aware routing) and §4.2 (why routing loses, Experiment 2d, feature ablations, synthetic-scaling status). The synthetic-scaling paragraph must be updated once item 12 runs.

## 11. Synthetic scaling sweep semantics

- **Evidence:** `--n_synthetic` is a per-dataset cap, not a total. Realised counts can be far below the value (NFCorpus caps at 868). The sweep labels now show the cap and the realised counts.
- **Options:**
  - a. Keep the per-dataset cap and report realised counts (current behaviour after the fix).
  - b. Make the sweep a fraction of each dataset's pool, so every setting uses the same share.
- **Recommendation:** (a) for now.
- **Status: resolved (a).**

## 12. Still to run on a GPU machine

- **Not run here:** this checkout has torch 2.15 CPU-only and no saved dense indexes, so the full 30k generation (flan-t5 generation plus dense encoding of the full NFCorpus / SciFact / FiQA corpora) is impractical. Commands:
  ```
  python scripts/generate_synthetic_queries.py --n_total 30000 [--lexical_pool_factor <from item 3>]
  python scripts/run_experiment2b_advantage.py --synthetic_dir results/synthetic --n_synthetic 0 1000 3000 10000 30000
  ```
- **Decision needed:** Run it before or after items 1–4 are settled? Running now produces numbers that items 1 and 3 may invalidate.
- **Status: open.** Items 1, 2 and 4 are settled. Run it on the GPU machine once a pool factor for item 3 is chosen.

## 13. Escalation parquet referenced in the report

- **Evidence:** `report/report.md` cites `results/experiment4/{dataset}_escalation.parquet`. None are present in this checkout (`*.parquet` was ignored, and the directory holds only CSV and JSON).
- **Decision needed:** Confirm the files exist on the machine that made them. Once they are committed through the gitignore fix, the citation is correct.
- **Status: resolved.** `scripts/export_escalation_parquet.py` writes Contract 4 for all five datasets from `results/experiment4/per_query_heldout.csv`. It re-derives which queries each policy escalates and checks that this reproduces every stored nDCG column. `latency_ms` is estimated from measured component timings (NFCorpus, SciFact, FiQA; null for Unix and Quora). The random policy is an expected value, so its `escalated` is null.
