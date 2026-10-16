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
