# Open problems that need a decision

Items the code cannot settle on its own. Each one lists the options and a recommendation. Nothing here has been changed yet.

## 1. NFCorpus has only 868 synthetic queries

- **Evidence:** 3,128 of NFCorpus's 3,633 passages are relevant to some test query, so the leakage rule excludes them. That leaves 434 eligible passages and a capacity of 868 queries at 50/50 style split.
- **Options:**
  - a. Accept the cap and state it in the report.
  - b. Drop NFCorpus from the synthetic set and say so.
  - c. Relax the leakage rule for NFCorpus. Not recommended: it breaks the test-isolation guarantee.
- **Recommendation:** (a). It is honest and needs no code change.

## 2. FiQA makes up 68% of the 30k budget

- **Evidence:** With the current equal-split-and-redistribute allocation, FiQA gets 20,404 of 30,000 queries. NFCorpus gets 868 and SciFact 8,728. The pooled scaling curve is therefore mostly FiQA.
- **Options:**
  - a. Keep the current allocation and report the per-dataset shares.
  - b. Cap every dataset at its capacity and report the shortfall, so the total is less than 30k.
  - c. Balance by sampling FiQA down to the smallest dataset's size.
- **Recommendation:** (a) for the first run, since the pooled scaling curve is already reported per dataset.

## 3. Lexical synthetic queries are nearly solved by BM25

- **Evidence:** On a 600-query SciFact smoke run, BM25 gets nDCG@10 0.999 on lexical queries. Natural SciFact BM25 is 0.683 (`improvements.md` §1.7). Each lexical query is made from its source passage's rarest words, so the source passage is almost always the top BM25 hit.
- **Risk:** The router may learn "lexical means BM25", which will not hold on natural queries. Any gain from synthetic data could be an artifact.
- **Options:**
  - a. Make the generator harder: drop a random subset of the chosen terms, or add a distractor-free paraphrase constraint.
  - b. Keep the generator and report synthetic-only and natural-only results separately, with the mismatch stated.
  - c. Validate with a small hand-checked or LLM-judged natural-style sample before scaling.
- **Recommendation:** (c) first, then (a) if the gap is still large.

## 4. Synthetic labels use one relevant document, natural labels use many

- **Evidence:** A synthetic query's only relevant document is its source passage (`synthetic/builder.py`). Natural qrels are multi-relevant: about 38 judgments per test query on NFCorpus (12,335 over 323), 2.6 on FiQA (1,707 over 648), 1.1 on SciFact (340 over 300).
- **Risk:** nDCG@10 values are on different scales, so the advantage targets are not directly comparable across the two sources.
- **Options:**
  - a. Accept the difference and document it.
  - b. Add a second relevant document per synthetic query, for example a near-duplicate passage. This is hard to do cleanly.
- **Recommendation:** (a). Revisit only if item 3 validation shows a problem.

## 5. MRR-blended training target (0.6·nDCG@10 + 0.4·MRR@10)

- **Evidence:** Specified in `improvements.md` §2.3. No code implements it. The committed cache holds nDCG only.
- **Options:**
  - a. Implement it. For natural queries, recompute MRR from the run files, which exist locally for NFCorpus, SciFact and FiQA. For synthetic queries, MRR is the reciprocal rank of the source passage and can be added in `build_route_table`.
  - b. Drop it from the plan and keep nDCG-only targets.
- **Recommendation:** (a) only if you want to test the smoothing effect. Otherwise (b).

## 6. 384-d query embedding features

- **Evidence:** `improvements.md` §3.5 lists them. `plan.md` Track 2 does not. No code implements them. Only the pilot script `scripts/demo_synthetic_routing.py` uses embeddings.
- **Options:**
  - a. Add them to Track 2: encode queries with the dense model and add them to the feature matrix.
  - b. Drop them and update `improvements.md` §3.5.
- **Recommendation:** Decide with the plan owner. Adding them increases the feature count from 18 to about 400, which is a large change for a weak-signal router.

## 7. Contract 2 wording does not match the code

- **Evidence:**
  - `plan.md` Contract 2 says `rank_correlation` is Kendall tau. The code computes Spearman (`features/retrieval_signal_features.py:46`).
  - Contract 2 says the top-1 scores are normalized. The code stores raw retriever scores (`features/feature_builder.py:150-151`).
- **Options:**
  - a. Change the contract text to Spearman and raw scores.
  - b. Change the code to Kendall and normalized scores, then regenerate the features.
- **Recommendation:** (a). It is free, and the committed experiments already use Spearman and raw scores.

## 8. The "pre-cached runs" claim in `plan.md`

- **Evidence:** `plan.md:43` says base runs and indexes are cached in the repo. `results/runs/`, `results/indexes/` and `data/raw/` are gitignored, and only the five `results/experiment2/cache/*.csv` files are tracked.
- **Options:**
  - a. Commit `results/runs/` (large, mostly `.trec` and `.parquet` files).
  - b. Add a script that regenerates runs from `data/raw` and make `plan.md` say so.
  - c. Store runs outside git, with a download link, and point `plan.md` at it.
- **Recommendation:** (b). It keeps the repo small and makes the claim true after one command.

## 9. Canonical run format: `.trec` or parquet

- **Evidence:** Contract 1 says parquet. The experiment scripts write `.trec`. `run_retrieval_baselines.py` writes parquet as well. The loaders accept both.
- **Options:**
  - a. Make parquet canonical and have experiment scripts write it too.
  - b. Make `.trec` canonical and change Contract 1.
- **Recommendation:** (a). It matches the contract and is typed.

## 10. Report Sections 2 and 4.3 are not drafted

- **Evidence:** `report/report.md` has no advantage-regression methodology section and no feature-ablation or synthetic-scaling section.
- **Decision needed:** Who writes them, and when. The numbers depend on items 1–4 above.

## 11. Synthetic scaling sweep semantics

- **Evidence:** `--n_synthetic` is a per-dataset cap, not a total. Realised counts can be far below the value (NFCorpus caps at 868). The sweep labels now show the cap and the realised counts.
- **Options:**
  - a. Keep the per-dataset cap and report realised counts (current behaviour after the fix).
  - b. Make the sweep a fraction of each dataset's pool, so every setting uses the same share.
- **Recommendation:** (a) for now.

## 12. Still to run on a GPU machine

- **Not run here:** torch, sentence-transformers and a GPU are not available in this checkout, so the full 30k generation has not been executed. Commands:
  ```
  python scripts/generate_synthetic_queries.py --n_total 30000
  python scripts/run_experiment2b_advantage.py --synthetic_dir results/synthetic --n_synthetic 0 1000 3000 10000 30000
  ```
- **Decision needed:** Run it before or after items 1–4 are settled? Running now produces numbers that items 1 and 3 may invalidate.

## 13. Escalation parquet referenced in the report

- **Evidence:** `report/report.md` cites `results/experiment4/{dataset}_escalation.parquet`. None are present in this checkout (`*.parquet` was ignored, and the directory holds only CSV and JSON).
- **Decision needed:** Confirm the files exist on the machine that made them. Once they are committed through the gitignore fix, the citation is correct.
