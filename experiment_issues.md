# Experiment Issues Log

Problems found while running Experiments 1, 1b and 2 (see `improvements.md` §1–3), what each one affected, and how it was handled. Ordered roughly by how much it changes the conclusions.

---

## A. Issues with the experimental design and claims

### A1. "Hybrid degrades on X% of queries" (H1) is a biased measure
* **Problem:** a query counted as "degraded" when max(BM25, Dense) beat hybrid by ≥ ε. A per-query maximum of two systems beats almost any single system just from per-query noise.
* **Evidence:**
  - Taking the better of two near-identical hybrids (α* ± 0.1) "degrades" the tuned hybrid on 12–23% of queries, close to the reported 17–27%.
  - Under the same test, BM25 is "degraded" on 19–34% of queries and Dense on 35–45%. Hybrid degrades *least*.
* **Status:** H1 verdicts withdrawn. Replaced by head-to-head win/loss counts and paired tests of fixed systems.

### A2. Oracle headroom (H2) is an upper bound, not evidence
* **Problem:** the oracle picks the best of three noisy per-query scores, so it is ≥ hybrid by construction, and its p-value only shows the headroom isn't zero.
* **Evidence:** an oracle over three near-identical hybrids already gains +0.010 / +0.019 / +0.026 nDCG@10. Headroom beyond that noise level is only about +0.016 / +0.028 / +0.022 (NFCorpus / SciFact / FiQA).
* **Status:** reported only as a bound. The decisive test became the held-out router (Experiment 2).

### A3. Query types do not separate the retrievers
* **Problem:** the plan assumed rare-term / keyword queries favour BM25 and vague / conversational queries favour dense.
* **Evidence:**
  - The pre-specified H3 feature tests were mostly null. The Jaccard-overlap hypothesis came out *reversed*: degradation is highest when the two retrievers agree.
  - Regex-defined types (identifier-like, conversational, unix-technical with flags / paths / error strings, Quora conversational) all favour hybrid, and every type's own best α equals the global α (0.6).
* **Status:** documented as a negative result (§1.7–1.8).

### A4. Tuning hybrid per dataset makes "hybrid loses" impossible by construction
* **Problem:** α ∈ {0, 1} is in the grid, so a hybrid tuned on a dataset ties the best single retriever at worst *on that dataset*. Running a "rare-word dataset" and a "conversational dataset" separately cannot show a single retriever beating hybrid.
* **Consequence:** a query-type argument needs query groups whose best α values sit at opposite ends (≈ 0 vs ≈ 1), mixed under one shared α. None of the five datasets shows such groups.

### A5. The router does not beat tuned hybrid on held-out queries (Experiment 2, pre-registered)
* **Result:**
  - Pooled GBM router − always-hybrid = −0.006 [−0.009, −0.003]: slightly but significantly worse.
  - Router vs. random routing at the same route proportions: +0.002, CI includes 0.
  - The logistic router and cross-dataset transfer are worse.
* **Meaning:** the 18 query and retrieval-signal features carry essentially no held-out routing signal. The oracle headroom (+0.034) is real but unreachable with them.

### A6. Statistical power is limited per dataset
* **Problem:** per-query nDCG@10 difference SDs are 0.08–0.35.
* **Consequence:** per dataset (300–1,000 queries), only effects of roughly ≥ 0.02–0.05 are detectable; pooled over 3,343 queries, roughly ≥ 0.01. Expected routing gains (a fraction of the 0.02–0.03 noise-corrected headroom) sit at the edge of detectability.

### A7. Latency: the router cannot be faster than hybrid
* **Problem:** retrieval-signal features need both BM25 and dense results, so every query pays for both retrievers. Any win must come from quality, and A5 shows there isn't one.
* **Possible fix:** a query-only router can skip the dense retriever. This is the efficiency setting of Arabzadeh et al. (A8).

### A8. Prior work overlaps the routing idea
* **Prior work:** Arabzadeh, Yan & Clarke, *Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection*, CIKM 2021 ([arXiv:2109.10739](https://arxiv.org/abs/2109.10739)). A per-query classifier chooses sparse / dense / hybrid, using query-only features, on MS MARCO.
* **Consequence:** per-query routing among these three routes is not novel. Remaining differentiators are confidence-based escalation to a cross-encoder under a budget, risk-coverage analysis, and held-out / transfer evaluation.

### A9. DAT is not evidence that hybrid loses on query types
* DAT (arXiv:2503.23013) argues theoretically that some queries prefer extreme α. Empirically, it shows that an LLM-chosen per-query α beats a fixed α = 0.6 (+3–4% Precision@1 on SQuAD / DRCD).
* That supports "a fixed α is suboptimal per query", given a strong (expensive) per-query signal. It does not show that hybrid loses to single retrievers on identifiable query types.

---

## B. Implementation bugs found and fixed

### B1. Min-max fusion dropped BM25's only hits (fixed; affected early Experiment 1 numbers)
* **Problem:** with 1–2 BM25 matches, min-max normalized the lone hit to 0, the same score as unretrieved documents. α = 0 "hybrid" did not reproduce BM25 on 77 of 323 NFCorpus queries. This made hybrid a partial strawman.
* **Fix:** anchor min-max at 0 when the BM25 list is truncated (BM25 scores are ≥ 0, so unretrieved docs score exactly 0). Exact fused ties are broken in favour of documents retrieved by the more-weighted retriever. Regression tests in `tests/test_hybrid.py`.
* **Effect:** NFCorpus hybrid 0.329 → 0.343; "degraded" share 31.6% → 26.6%. Numbers reported before the fix are superseded.

### B2. Numeric query IDs parsed as integers (caught before reporting)
* **Problem:** `pd.read_csv` turned SciFact / FiQA query IDs into ints, so a re-index produced NaNs and a bogus "0.0%" in a null check.
* **Fix:** read IDs with `dtype={"query_id": str}`.

### B3. Logging crash on Windows (fixed)
* **Problem:** "α", "ε" and "→" in log messages raised `UnicodeEncodeError` under the cp1252 console encoding. It was non-fatal, but log lines were lost.
* **Fix:** the logger reconfigures stdout to UTF-8; the file handler writes UTF-8.

### B4. Smaller issues
* `.gitignore`'s `data/` rule also ignored `src/router_retrieval/data/`, so `schema.py` was never committed. Changed to `/data/`.
* The pooled-CDF figure directory was not created. Fixed.
* f-strings with backslashes / nested same-type quotes required Python ≥ 3.12. Rewritten for the declared `>=3.9`.

---

## C. Environment and infrastructure issues

* **BEIR mirror TLS:** the UKP server does not send its intermediate certificate, so Python's `urllib` fails verification. `scripts/download_data.py` falls back to `curl`, which uses the OS certificate store.
* **CQADupStack size:** the BEIR zip is 5.1 GB (all 12 forums). Per-forum copies from MTEB on Hugging Face are used instead (unix: 50 MB).
* **Quora corpus size:** 523k docs. Experiments use 1,000 sampled test queries + their 1,544 relevant docs + 50,000 random distractors (seed 42). The smaller corpus inflates absolute scores (nDCG ≈ 0.93–0.97), but every system is compared on the same corpus.
* **GPU:** the venv first had CPU-only torch (~8 docs/s encoding). It was switched to the CUDA 12.8 build for the RTX 4050. `retrievers.yaml` now uses `device: "auto"`.
* **Slow network** during setup (as low as ~34 KB/s) made the initial install and model downloads take a long time.
* **Dense model change:** Experiment 1 used `msmarco-distilbert-base-v4`, which is weaker than BM25 on NFCorpus / SciFact. Experiments 1b and 2 use `intfloat/e5-small-v2`, so Experiment 1 and 2 absolute numbers are not directly comparable.

---

## D. Issues from Experiments 3 & 4 (efficiency and escalation)

### D1. The learned skip-router is no better than a one-feature rule (Experiment 3)
* **Problem:** the pre-registered success criteria all pass (32.9% of dense calls skipped, −0.005 nDCG@10, beats random skipping), but the BM25 top-1/top-2 score-ratio heuristic matches the LightGBM model at every skip rate.
* **Consequence:** report the result as "a confident BM25 head is a usable skip signal", not as a win for the learned router.

### D2. Skipping savings are dataset-dependent (Experiment 3)
* Savings range from 63% (Quora) to 5% (FiQA). Transfer to NFCorpus and FiQA does not beat random skipping.
* A deployment would need per-domain threshold calibration.

### D3. Latency not yet measured (Experiment 3)
* The pre-registration (§4.1) promised per-component timings on FiQA. They have not been run yet.
* At our corpus sizes, dense search takes milliseconds, so the wall-clock savings may be small even though 33% of dense calls are skipped.

### D4. The cross-encoder is weaker than the first stage (Experiment 4)
* **Problem:** MiniLM-L6 reranking of the hybrid top-50 *lowers* nDCG@10 on all five datasets (−0.004 to −0.041). Escalation cannot help on average when the "expensive, better" backend is not better.
* This was flagged as a risk in the pre-registration, and it is what happened.
* **Possible fix:** a stronger reranker, as a new pre-registered experiment.

### D5. Reranker gain is unpredictable from cheap features (Experiment 4)
* Spearman(predicted gain, true gain) is −0.05 to +0.07. The oracle shows exploitable queries exist (+0.029 at a 10% budget), but neither predicted gain nor router confidence finds them better than random.

### D6. "Quality retained" metric breaks when reranking hurts
* (policy − hybrid) / (always-rerank − hybrid) has a negative denominator here, which produces meaningless percentages. Only report it when always-rerank > always-hybrid.

---

## E. Why the Experiment 2 router is *worse* than hybrid, and the fix (Experiment 2d)

### E1. Class balancing made the router leave hybrid far too often (fixed in Experiment 2d)
* **Problem:** the GBM and logistic routers use `class_weight: "balanced"` (`configs/router.yaml`). Only 5–15% of queries are labelled BM25 or dense, so balancing multiplies their weight by roughly 6–15× and the router predicts them much more often than their expected value justifies.
* **Why it costs nDCG:** leaving hybrid is asymmetric. Pooled, BM25 beats hybrid on 7% of queries (mean +0.20) and loses on 28% (mean −0.31); dense wins on 12% (+0.17) and loses on 18% (−0.22). A router has to be right well over half the time it switches. Of the 190 held-out queries the Experiment 2 router sent to BM25, 16 improved and 63 got worse (net −14.8 nDCG points).
* **Fix:** Experiment 2d (`scripts/run_experiment2d_cost_aware_router.py`, pre-declared in `results/preregistration/experiment2d_cost_aware_router_20261031.md`) removes class balancing, measures label margins against hybrid (E2), and falls back to always-hybrid unless inner CV on the training fold shows a gain.
* **Effect:** pooled router − hybrid goes from −0.0073 [−0.0100, −0.0045] (Experiment 2 router refit on the same folds) to −0.0012 [−0.0025, +0.0002]. Non-inferiority at δ = 0.005 holds. Superiority (H1) is still **not supported**; no variant (blended nDCG/MRR target, + query embedding, no fallback) has a CI above 0.
* **Status:** `configs/router.yaml` is unchanged, so the pre-registered Experiment 2 and Experiment 3 results still reproduce; Experiment 2d sets its options in its own script.

### E2. Margin labels compared the top route with the runner-up, not with hybrid (option added)
* **Problem:** `MarginLabeler` labels a query "hybrid" when top1 − top2 < ε. If BM25 and dense are close to each other but both well above hybrid, the label is "hybrid", which is the worst route. It affects 1–2 queries per dataset at the tuned ε.
* **Fix:** `MarginLabeler(margin_to="default")` measures the margin against hybrid. The default (`"runner_up"`) keeps the pre-registered rule.

### E3. The labels were not the main problem
* Labels are correct per query, but nDCG@10 per query is very noisy: all three routes tie on 34–79% of queries, and hybrid is strictly best on only 3–19%. Even cost-aware routers mostly fall back to hybrid, because no feature set separates the 7–12% of queries where a single retriever wins.

### E4. Better models cannot fix it: the features lack precision (explanatory)
* **Evidence** (`scripts/analyze_route_predictability.py`, `results/route_predictability/`): break-even precision among queries whose score changes is 61% (BM25) / 57% (dense). A held-out classifier on the 18 features + query embedding reaches AUC 0.67–0.68, but precision on its most confident 1–10% is 29–35% (BM25) and 32–46% (dense), with a negative mean change at every cut-off.
* **Document-aware features** (Experiment 2e, `scripts/run_experiment2e_ce_feature_router.py`, pre-declared): cached cross-encoder scores of each route's top-3 make routing beat random routing (MRR@10 +0.0031 [+0.0014, +0.0048]) but not always-hybrid (nDCG@10 −0.0003 [−0.0017, +0.0011]), at about 40 ms per query.
* **Remaining options:** more training data (30k synthetic queries, GPU) or an LLM reading the query, both untested here.

### E5. Reranking depth matters more than query selection (Experiment 4c)
* **Pre-declared test failed:** reranking every query's top-5 with the blended backend (same cross-encoder cost as escalating 10% of queries) gives +0.0010 [−0.0013, +0.0033], 15% of the always-blended gain, no better than 4b's escalation at equal cost.
* **Secondary, post hoc:** top-10 +0.0026 [+0.0001, +0.0050]; top-20 +0.0067 [+0.0039, +0.0095], the whole always-blended gain at 40% of its cross-encoder pairs and a p95 well below 10% escalation's. Confirm on new data before claiming it.
