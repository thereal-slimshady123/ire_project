# Project Improvements & Experimental Designs

## 1. Experiment: Demonstrating That Always Picking Hybrid Degrades Accuracy

### 1.1 Objective & Research Question
**Research Question:** Does a static convex combination of sparse and dense retrieval scores ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) cause statistically significant ranking degradation on identifiable subsets of queries compared to single-retriever routes?

**Core Claim:** While fixed-weight hybrid retrieval achieves the highest *mean* nDCG across an entire dataset, it exhibits severe **query-level negative transfer** (fusion degradation) on 15%–30% of individual queries where pure BM25 or pure Dense strictly outperforms Hybrid.

---

### 1.2 Formal Hypotheses
* **$H_1$ (Degradation Prevalence):** On at least 15% of evaluation queries, $\max(\text{nDCG@10}_{\text{BM25}}, \text{nDCG@10}_{\text{Dense}}) - \text{nDCG@10}_{\text{Hybrid}} > \epsilon$ (with margin $\epsilon = 0.02$).
* **$H_2$ (Oracle Headroom):** An oracle router dynamically selecting the best route per query achieves a statistically significant headroom ($\ge 2.0$ nDCG points, $p < 0.01$) over the best-tuned fixed-$\alpha$ hybrid baseline.
* **$H_3$ (Attributable Failure Modes):**
  * Degradation where **BM25 > Hybrid** correlates with high rare-entity density and high max-IDF (dense introduces semantic drift/false positives).
  * Degradation where **Dense > Hybrid** correlates with low keyword overlap and low BM25 top-1 score gap (BM25 introduces spurious lexical noise).
  * Severity of degradation peaks when retriever agreement (top-$k$ Jaccard overlap) is near zero.

---

### 1.3 Variables & Controls
* **Independent Variables:**
  * Retrieval Route: BM25-only, Dense-only (Contriever / DistilBERT), and Hybrid.
  * Hybrid Weight $\alpha$: Evaluated across grid $[0.0, 0.1, \dots, 1.0]$. The baseline hybrid must use the **optimal $\alpha^*$ tuned on training folds** (e.g. 5-fold CV) so that Hybrid is never a "strawman".
* **Normalization Control:**
  * BM25 scores are unbounded $[0, \infty)$ while dense cosine scores are bounded $[-1, 1]$. Both must be Min-Max normalized per query into $[0, 1]$ prior to combination:
    $$\tilde{s}(d) = \frac{s(d) - \min_{d' \in C} s(d')}{\max_{d' \in C} s(d') - \min_{d' \in C} s(d') + \delta}$$
* **Dependent Variables:**
  * Primary: $\text{nDCG@10}(r, q)$ per query $q$ for route $r$.
  * Secondary: $\text{MRR@10}(r, q)$, $\text{MAP}(r, q)$, $\text{Recall@100}(r, q)$.

---

### 1.4 Step-by-Step Experimental Protocol

#### Step 1: Execution & Run Persistence
1. For each dataset (NFCorpus, SciFact, FiQA), execute:
   * BM25 retrieval for top-100 candidates per query.
   * Dense retrieval for top-100 candidates per query.
2. For each query, normalize candidate scores and compute Hybrid run for $\alpha \in \{0.0, 0.1, \dots, 1.0\}$.
3. Select $\alpha^*$ on training folds.

#### Step 2: Per-Query Metric Computation
Using ground-truth qrels, compute per-query ranking metrics for each route:
$$\Delta_{\text{BM25}}(q) = \text{nDCG@10}(\text{BM25}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$
$$\Delta_{\text{Dense}}(q) = \text{nDCG@10}(\text{Dense}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$
$$\Delta_{\text{MaxSingle}}(q) = \max(\text{nDCG@10}(\text{BM25}, q), \text{nDCG@10}(\text{Dense}, q)) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$

#### Step 3: Query Partitioning (Win / Tie / Loss Classification)
Categorize every query into one of four mutually exclusive partitions using threshold $\epsilon = 0.02$:
1. **BM25 Dominant (Dense Noise Pollution):**
   $$\text{nDCG@10}(\text{BM25}, q) - \max(\text{nDCG@10}(\text{Dense}, q), \text{nDCG@10}(\text{Hybrid}, q)) \ge \epsilon$$
2. **Dense Dominant (BM25 Lexical Noise Pollution):**
   $$\text{nDCG@10}(\text{Dense}, q) - \max(\text{nDCG@10}(\text{BM25}, q), \text{nDCG@10}(\text{Hybrid}, q)) \ge \epsilon$$
3. **Hybrid Dominant (Synergistic Fusion):**
   $$\text{nDCG@10}(\text{Hybrid}, q) - \max(\text{nDCG@10}(\text{BM25}, q), \text{nDCG@10}(\text{Dense}, q)) \ge \epsilon$$
4. **Indifferent / Tie:**
   $$|\max(\text{scores}) - \min(\text{scores})| < \epsilon$$

#### Step 4: Statistical Testing
* Run a **paired Wilcoxon signed-rank test** and **paired bootstrap test** ($B=10{,}000$) comparing:
  * $\text{Oracle Router}$ vs. $\text{Hybrid}_{\alpha^*}$ across all queries.
  * $\text{BM25}$ vs. $\text{Hybrid}_{\alpha^*}$ strictly on the BM25-dominant partition to prove magnitude and significance of degradation.
  * $\text{Dense}$ vs. $\text{Hybrid}_{\alpha^*}$ strictly on the Dense-dominant partition.

---

### 1.5 Diagnostic Analysis: Correlating Failure Modes with Query Features
To prove *why* hybrid makes accuracy suffer, correlate per-query degradation $\Delta_{\text{MaxSingle}}(q)$ with measurable feature signals:
1. **Lexical Specificity:** Compute Max-IDF and rare-term density.
   * *Hypothesis:* Queries in the top quartile of Max-IDF have significantly higher $\Delta_{\text{BM25}}$ (dense dilutes precision).
2. **Retriever Disagreement (Jaccard Overlap @ 10):**
   $$J(q) = \frac{|\text{Top10}_{\text{BM25}}(q) \cap \text{Top10}_{\text{Dense}}(q)|}{|\text{Top10}_{\text{BM25}}(q) \cup \text{Top10}_{\text{Dense}}(q)|}$$
   * *Hypothesis:* When $J(q) = 0$, fusion is high-risk; hybrid assigns intermediate scores to discordant items, demoting true positives from both sources.
3. **Score Separation Gap:** Ratio of Top-1 score to Top-2 score for each retriever. If BM25 has a massive gap but Dense is flat, combining them ruins BM25's confident hit.

---

### 1.6 Results Reporting & Visualizations

1. **Query Win/Loss Table:**
   | Dataset | Total Queries | Hybrid Wins (%) | BM25 Wins (%) | Dense Wins (%) | Ties (%) | Total Degraded by Hybrid (%) |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
   | NFCorpus | $N$ | - | - | - | - | - |
   | SciFact | $N$ | - | - | - | - | - |
   | FiQA | $N$ | - | - | - | - | - |

2. **Oracle Headroom Table:**
   | Dataset | BM25 nDCG | Dense nDCG | Best Hybrid ($\alpha^*$) | Oracle Router | Headroom ($\Delta$) | $p$-value |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
   | NFCorpus | - | - | - | - | - | - |
   | SciFact | - | - | - | - | - | - |
   | FiQA | - | - | - | - | - | - |

3. **Plots to Generate:**
   * **Degradation Scatter Plot:** Per-query plot with x-axis $= \max(\text{nDCG}_{\text{BM25}}, \text{nDCG}_{\text{Dense}})$ and y-axis $= \text{nDCG}_{\text{Hybrid}}$. All points below the $y = x$ identity line are empirical evidence of hybrid degradation.
   * **CDF of Delta:** Cumulative distribution of $\Delta_{\text{MaxSingle}}(q)$.
   * **Feature Distribution Boxplots:** Max-IDF and Jaccard overlap grouped by winning route partition.

---

### 1.7 Implementation Notes & Results (run 2026-10-08)

**Reproduce:** `python scripts/download_data.py && python scripts/run_experiment1_hybrid_degradation.py --datasets nfcorpus scifact fiqa`
Outputs land in `results/experiment1/` (`tables.md`, per-dataset `per_query.csv`, `summary.json`, `alpha_sweep.csv`, `feature_correlations.csv`, `figures/`). BM25/dense runs and corpus embeddings are cached under `results/runs/` and `results/indexes/`, so re-running the analysis takes seconds.

**Setup:** BM25 (k1=1.2, b=0.75, Lucene stopwords + Porter stemming) and `msmarco-distilbert-base-v4` dense (cosine, FAISS flat IP), top-100 each; hybrid over the union of both lists; α* chosen per fold by 5-fold CV over test queries (hybrid scores are out-of-fold); ε = 0.02; 10,000 bootstrap resamples. Sanity check: our BM25 nDCG@10 (NFCorpus 0.322, SciFact 0.683, FiQA 0.253) is in line with published BEIR BM25 numbers.

**Deviations from §1.3–1.4 (and why):**
1. *Normalization floor.* Min-max over the candidate list (§1.3) breaks on short BM25 lists: a single BM25 hit normalizes to 0, the same score as an unretrieved document, so the hybrid drops BM25's only match. Because BM25 only returns positively-scored documents, a list shorter than k means every unretrieved document scores exactly 0, so we anchor min-max at 0 in that case (list minimum otherwise). Exact fused-score ties are broken in favour of documents retrieved by the more-weighted retriever. Without this fix the hybrid was a partial strawman (NFCorpus hybrid 0.329 → 0.343 after the fix; degradation 31.6% → 26.6%).
2. *Partitions.* §1.4's four partitions are not exhaustive (e.g. BM25 ≈ Dense, both ≥ ε above Hybrid). We use six: `bm25/dense/hybrid_dominant`, `tie`, and `near_tie_degraded` / `near_tie_safe` (no single winner, split by whether max(BM25, Dense) − Hybrid ≥ ε). "Degraded" = `bm25_dominant ∪ dense_dominant ∪ near_tie_degraded` = {q : Δ_MaxSingle(q) ≥ ε} exactly (unit-tested).
3. *Extra controls.* Every H1/H2 result is also reported with α tuned in-sample on all queries (the most favourable fixed α possible for hybrid), plus a two-route oracle {BM25, Dense} that never fuses.

**Results:**

| Dataset | Queries | BM25 | Dense | Hybrid (CV α*) | Oracle | Headroom [95% CI] | Degraded by hybrid [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NFCorpus | 323 | 0.322 | 0.268 | 0.343 | 0.369 | **+0.026** [0.020, 0.033] | **26.6%** [22.1, 31.7] |
| SciFact | 300 | 0.683 | 0.529 | 0.703 | 0.750 | **+0.047** [0.032, 0.063] | **17.3%** [13.5, 22.0] |
| FiQA | 648 | 0.253 | 0.245 | 0.305 | 0.353 | **+0.048** [0.040, 0.057] | **25.5%** [22.3, 29.0] |

* **Hybrid is the strongest fixed system on every dataset**, both on average (+0.021 / +0.020 / +0.052 nDCG@10 over the better single retriever) and head-to-head per query (it beats BM25 on 25% / 14% / 31% of queries and loses to BM25 on only 14% / 8% / 13%).
* **H1, as formulated, is not informative, and the "supported" verdict in `tables.md` should not be cited.** "Degraded" compares hybrid against the *per-query max* of two other systems, and a per-query max beats almost any single system just from per-query noise. Two checks:
  - Taking the better of two near-identical hybrids (α* ± 0.1) "degrades" the tuned hybrid on 14% / 12% / 23% of queries, close to the reported 27% / 17% / 26%.
  - Under the same test, BM25 is "degraded" on 19–34% of queries and Dense on 35–45%. Hybrid degrades *least*.
* **H2 (oracle headroom) is an upper bound, not evidence.** The oracle picks the best of three noisy per-query scores, so it is ≥ hybrid by construction and its p-value only shows the headroom isn't zero. An oracle over three near-identical hybrids (α*, α* ± 0.1) already gains +0.010 / +0.019 / +0.026. The routing headroom **beyond that noise level** is about **+0.016 / +0.028 / +0.022** nDCG@10 (NFCorpus / SciFact / FiQA). That is the most a router could gain, and it is roughly at the proposal's ~2-point checkpoint threshold. The partition-level tests in `summary.json` (`partition_tests_posthoc`) are likewise circular, since the partitions are selected on the outcome; treat them as descriptive.
* **H3 mostly not supported as stated.** Of the four pre-specified Mann-Whitney tests, only Max-IDF → larger Δ_BM25 on FiQA reached p < 0.05 (p = 0.017); none survive across datasets. **H3c is contradicted:** degradation is *highest* when BM25 and Dense agree (Jaccard@10 > 0.4: 35–45% degraded on all three datasets), not when they disagree completely (Jaccard@10 = 0: 7% / 15% / 22% degraded on NFCorpus / SciFact / FiQA, the lowest or near-lowest bin). A plausible reading is that when the two lists overlap, fusion mostly reshuffles shared documents and can demote the better retriever's top hit, whereas disjoint lists leave the stronger retriever's head intact. BH-significant feature correlations exist on every dataset (|ρ| ≤ 0.28: Jaccard, top-1 scores, score gaps/entropy, query-term coverage), but they are weak and differ between datasets. In other words, degradation is predictable only weakly from these features, which is the open question Experiment 2 / the router has to answer.

### 1.8 Experiment 1b: Query Types Chosen to Favour Each Retriever (run 2026-10-08)

**Reproduce:** `python scripts/run_experiment1b_query_types.py` → `results/experiment1b/`.

**Setup:**
- **Mixed workload:** CQADupStack-unix (1,072 StackExchange questions with error strings, flags and paths; full 47k-doc corpus) plus a Quora subsample (1,000 test queries; their 1,544 relevant docs + 50,000 random distractors; seed 42). Each query searches its own corpus.
- **Retrievers:** dense model `intfloat/e5-small-v2` (with `query:` / `passage:` prefixes); BM25 as before.
- **Always-hybrid:** one global α chosen by 5-fold CV over the pooled queries.
- **Query types,** fixed by a regex rule before any per-type result was computed (`TECH_RE` in the script): `unix_technical` (188), `unix_plain` (884), `quora_conversational` (1,000).
- **Test:** paired comparison of fixed systems within each type, so no per-query max-of-two bias.

| Query type | n | BM25 | Dense | Hybrid (global α) | BM25 − Hybrid [95% CI] | Dense − Hybrid [95% CI] | α tuned on this type alone → nDCG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| unix_technical | 188 | 0.346 | 0.427 | **0.448** | −0.102 [−0.137, −0.067] | −0.020 [−0.043, +0.001] | 0.6 → 0.456 |
| unix_plain | 884 | 0.283 | 0.345 | **0.356** | −0.074 [−0.089, −0.059] | −0.011 [−0.020, −0.001] | 0.6 → 0.359 |
| quora_conversational | 1000 | 0.933 | 0.959 | **0.968** | −0.034 [−0.043, −0.026] | −0.009 [−0.014, −0.004] | 0.6 → 0.968 |

**Hybrid wins on every type, and every type's own best α (0.6) equals the global α.** Splitting the workload by type gives a per-type α no different from the global one, so a fixed hybrid loses nothing to the mixing. This also rules out "run each query type as its own dataset" as a route to the claim: α ∈ {0, 1} is in the grid, so a hybrid tuned on a dataset can never lose to a single retriever *on that dataset* (it ties at worst). A query-type argument against always-hybrid needs query groups whose best α values sit at opposite ends, and none of the five datasets tested shows this.

**Conclusion of Experiment 1 / 1b.** We could not show a natural query type on which a tuned hybrid loses to BM25 or dense. What remains is per-query variation of about 1.6–2.8 nDCG points beyond noise (§1.7). Whether a router can *predict* it on held-out queries is the decisive question, and that is Experiment 2 (§3).

---

## 2. Generating Ground Truth Labels for the Training Dataset

Labels are generated inside each training fold of Experiment 2 (§3, step 3); they are never computed on held-out queries.

---

## 3. Experiment 2: Held-Out Routing Test (pre-registered 2026-10-08, before any router was trained)

### 3.1 Question
Can a router trained on some queries beat **always-hybrid with α tuned on the same training queries** on queries it has never seen? This answers "why not always pick hybrid?" directly. The oracle gap from §1.7 is only an upper bound on the answer.

### 3.2 Data and Retrievers (fixed)
* **Five datasets:** NFCorpus (323 test queries), SciFact (300), FiQA (648), CQADupStack-unix (1,072), and the Quora subsample from §1.8 (1,000). Total **3,343 queries**.
* **Retrievers:** BM25 as in §1.7; dense `intfloat/e5-small-v2` with E5 prefixes; top-100 each.
* **Hybrid:** min-max with the BM25 floor fix (§1.7).
* **Routes:** `bm25`, `dense`, `hybrid`.
* **Features:** the 18 query and retrieval-signal features of §1.5 (`FEATURE_COLUMNS` in `run_experiment1_hybrid_degradation.py`). They are computed once per query and use no relevance labels.

### 3.3 Protocol (within each dataset: 5-fold CV over queries, seed 42)
Everything below is fitted on the **training folds only**, then applied to the held-out fold.
1. **Hybrid α\*:** argmax of mean training nDCG@10 over the α grid. This α defines both the `hybrid` route and the always-hybrid baseline.
2. **Best single retriever:** whichever of BM25 / Dense has the higher mean training nDCG@10.
3. **Labels:** margin labeler (`labels.yaml`). The best route by nDCG@10 wins only if it beats the runner-up by ≥ ε; otherwise the label is `hybrid`. ε ∈ {0, 0.01, 0.02, 0.05, 0.10} is chosen by an inner 3-fold CV on the training queries, maximizing the routed nDCG@10 of the GBM router.
4. **Routers:**
   * **GBM (primary):** LightGBM, class-balanced, sample weight = per-query regret spread (max − min route nDCG@10, + 0.01).
   * **Logistic regression (secondary):** standardized features, class-balanced. Hyper-parameters from `configs/router.yaml`, not tuned.
5. **Held-out score:** the nDCG@10 of the route the router picks.

### 3.4 Comparisons on held-out queries
* **Primary:** GBM router vs. always-hybrid (training-fold α\*).
* **Secondary:**
  - logistic router vs. always-hybrid;
  - router vs. the best single retriever;
  - router vs. **random routing with the router's own route proportions**, scored by expected value per query. Beating it shows the router chooses *which* queries to send where, not just a good route mix.
  - oracle (upper bound only).
* **Cross-dataset transfer:** leave one dataset out; train router, α, ε and best-single on the other four; test on the held-out dataset. Compare against hybrid with α from the four source datasets, and against the harder baseline of hybrid with α tuned in-domain by CV.

### 3.5 Success Criterion (fixed in advance)
* **Primary test:** pool the held-out per-query differences nDCG@10(GBM router) − nDCG@10(always-hybrid) over all five datasets (n = 3,343). Take a paired bootstrap (10,000 resamples, seed 42).
  - **Success:** the 95% CI lower bound is > 0.
  - **Negative:** the CI includes 0. A negative result is reported as such.
* Per-dataset results, the logistic router, the random-routing control and the transfer results are reported but do not change the verdict.
* **Power:** per-query difference SDs measured in Experiments 1/1b are 0.08–0.35. Pooled over 3,343 queries, the standard error is about 0.003–0.005, so gains of roughly ≥ 0.01 should be detectable. Per dataset, roughly 0.02–0.05 is needed.
* **No changes after seeing results.** Features, models, hyper-parameters, the ε grid and this criterion will not change once held-out results exist. Any later analysis is labelled exploratory.

### 3.6 Known Limitations, Stated Up Front
* **Latency.** The router uses both retrievers' scores, so it cannot be faster than hybrid. Any win must come from quality. A query-only router variant (directly comparable to Arabzadeh et al.) is the way to claim efficiency.
* **Related work.** Per-query selection among sparse / dense / hybrid was done by Arabzadeh, Yan & Clarke, "Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection" (CIKM 2021, arXiv:2109.10739). They use query-only features on MS MARCO. This project's novelty must rest on confidence-based escalation to a cross-encoder under a budget, the risk-coverage analysis, and the held-out/transfer evaluation, not on routing itself.
* **Escalation.** The escalation analysis (router-chosen vs. random escalation at equal rate vs. always-rerank) does not depend on hybrid being suboptimal. It runs after this experiment.
### 3.7 Results (run 2026-10-08, after pre-registration commit `f078818`)

**Reproduce:** `python scripts/run_experiment2_heldout_routing.py` → `results/experiment2/` (`tables.md`, `summary.json`, `per_query_heldout.csv`).

**Primary verdict: NEGATIVE.** Pooled over all 3,343 held-out queries, nDCG@10(GBM router) − nDCG@10(always-hybrid) = **−0.0061 [−0.0091, −0.0032]**. The router is slightly but significantly *worse* than always-hybrid tuned on the same training folds.

| Dataset | n | Always-hybrid | Best single | GBM router | GBM − hybrid [95% CI] | GBM − random (same route mix) | Oracle − hybrid (bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NFCorpus | 323 | 0.359 | 0.316 | 0.355 | −0.004 [−0.009, +0.000] | −0.000 [−0.005, +0.004] | +0.029 |
| SciFact | 300 | 0.726 | 0.668 | 0.730 | +0.004 [−0.010, +0.018] | +0.014 [+0.002, +0.027] | +0.065 |
| FiQA | 648 | 0.403 | 0.375 | 0.393 | −0.010 [−0.017, −0.004] | +0.004 [−0.003, +0.010] | +0.044 |
| CQADupStack-unix | 1072 | 0.376 | 0.360 | 0.366 | −0.011 [−0.017, −0.004] | +0.000 [−0.006, +0.006] | +0.042 |
| Quora (subset) | 1000 | 0.967 | 0.959 | 0.965 | −0.002 [−0.006, +0.001] | −0.001 [−0.004, +0.003] | +0.013 |
| **Pooled** | 3343 | 0.588 | 0.565 | 0.582 | **−0.006 [−0.009, −0.003]** | +0.002 [−0.001, +0.004] | +0.034 |

**What the results say:**
* **The router does not choose well.** It sends 68–95% of queries to hybrid. Its choice of *which* queries go to BM25 or dense is no better than random assignment at the same proportions (pooled +0.002, CI includes 0; only SciFact is marginally better than random). The 18 query and retrieval-signal features carry essentially no held-out routing signal, consistent with the weak feature correlations in §1.7.
* **The headroom exists but is unreachable with these features.** The oracle is +0.034 above hybrid, yet the router captures none of it. The per-query differences between routes behave mostly like noise *with respect to these features*.
* **The logistic router is worse** (−0.025 pooled). **Transfer** (train on four datasets, test on the fifth) is worse than or tied with hybrid on every target.
* **Against the best single retriever,** the router wins (+0.017), but only because it mostly picks hybrid.

**Implication for the proposal.** Under this evaluation, "why not always pick hybrid?" has no good answer: tuned hybrid is the best cheap choice, and routing among the cheap routes does not beat it on held-out queries. The project's case has to rest on the parts that do not depend on hybrid being suboptimal:
1. **Confidence-based escalation to a cross-encoder under a budget.** Compare router-chosen escalation against random escalation at the same rate and against always-rerank.
2. **The negative result itself,** reported per the pre-registration.

Possible follow-ups (exploratory, not part of the pre-registered test): richer features such as query embeddings (the MDPI-style latent router), or a query-only router aimed at *efficiency* rather than quality (the Arabzadeh et al. setting).

---

## 4. Experiments 3 & 4: Efficiency and Escalation (pre-registered 2026-10-08, before any code for them was written)

**Reframed claim.** Always-hybrid is not beaten on quality by cheap routing (§3.7). The claim tested instead is that **always-hybrid wastes compute on easy queries (Experiment 3) and under-serves hard ones (Experiment 4).**

**Shared setup.** Same as §3.2–3.3:
* five datasets, 3,343 queries; e5-small-v2; top-100;
* 5-fold CV within each dataset (seed 42), with every parameter fitted on training folds only;
* pooled primary tests, paired bootstrap (10,000 resamples), 95% CI;
* per-dataset results and leave-one-dataset-out transfer are secondary.

### 4.1 Experiment 3: Skipping the Dense Retriever
* **Decision per query:** answer from BM25 only, or run hybrid (BM25 + dense). Dense-only costs the same as hybrid and is excluded.
* **Features:** only what exists before dense runs. Query-only: length, max/mean IDF, rare-term density, entity density, OOV fraction. BM25 result signals: top-1 score, normalized top-1/top-2 gap, top-1/top-2 ratio, score entropy, top-10 query-term coverage, number of BM25 hits (≤ 100).
* **Label:** "BM25 suffices" ⇔ nDCG@10(BM25) ≥ nDCG@10(hybrid at training α\*) − 0.02. Fixed, not tuned.
* **Model:** LightGBM classifier (`configs/router.yaml` GBM hyper-parameters, class-balanced) giving P(BM25 suffices).
* **Operating point:**
  - Inner 3-fold CV on the training folds gives out-of-fold P for training queries.
  - The skip rule is "skip dense iff P ≥ τ". τ is the lowest value (most skipping) whose out-of-fold training loss, mean nDCG@10(routed) − mean nDCG@10(hybrid), is ≥ −0.005 (half the non-inferiority margin).
  - The same τ is applied to held-out queries.
* **Comparisons:**
  - always-hybrid;
  - always-BM25;
  - **random skipping at the router's held-out skip rate** (expected value);
  - **heuristic H1:** BM25 top-1/top-2 ratio alone;
  - **heuristic H2:** max-IDF alone;
  - oracle skipping (upper bound).

  For each heuristic, the direction (higher → skip, or lower → skip) and the threshold are chosen on training folds by the same loss ≤ 0.005 rule.
* **Curves:** held-out nDCG@10 vs. dense-call rate, sweeping the skip rate 0–100% by ranking held-out queries on each method's score. Random skipping is the straight line between always-BM25 and always-hybrid.
* **Success (all three required; pooled, at the training-chosen operating point):**
  1. **Non-inferior:** the 95% CI lower bound of nDCG@10(router) − nDCG@10(always-hybrid) is ≥ −0.01.
  2. **Real savings:** dense-call savings ≥ 20% of queries.
  3. **Better than chance:** router − random skipping at the same rate has a 95% CI lower bound > 0.

  Reported but not part of the verdict: router vs. heuristics H1/H2 at matched skip rate. If a heuristic matches the router, the report will say so.
* **Latency:** per-component timings on FiQA (the largest corpus): BM25 search, dense query encoding (CPU and GPU), FAISS search, fusion, router inference. Mean and p95. Dense-call share is the hardware-independent cost measure.

### 4.2 Experiment 4: Selective Cross-Encoder Escalation
* **Base system:** always-hybrid with training-fold α\*.
* **Escalation:** rerank the top-50 of the hybrid ranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Hybrid ranks 51–100 are appended unchanged. Cross-encoder scores are precomputed once for every (query, doc) in the BM25 ∪ dense top-100 union, so no choice depends on held-out labels.
* **Gain** = nDCG@10(escalated) − nDCG@10(hybrid), per query.
* **Policies at budget B (share of queries escalated):**
  - **predicted-gain (primary):** LightGBM regressor trained on training-fold gains; features = the 18 features of §3.2 + hybrid fused-score top-1, top-1/top-2 gap and entropy.
  - **router-confidence (secondary):** the §3 GBM router; escalate the lowest max-class-probability.
  - random (expected value);
  - oracle (top-B by true gain; upper bound).

  Held-out queries are ranked by each policy's score and the top-B share is escalated, per fold.
* **Budgets:** B ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0}.
* **Reported:**
  - nDCG@10 vs. B per policy;
  - **quality retained** = (policy − hybrid) / (always-rerank − hybrid);
  - per-dataset always-rerank gain, so a weak result can be traced to "reranking doesn't help here" vs. "the policy chooses badly";
  - latency of cross-encoder reranking per escalated query.
* **Success (pooled, B = 10%):** nDCG@10(predicted-gain) − nDCG@10(random escalation) has a 95% CI lower bound > 0. All other budgets and policies are secondary.

**No changes after seeing results** (as in §3.5). Negative results are reported as such.

### 4.3 Results (run 2026-10-08; pre-registration snapshot `results/preregistration/`, SHA-256 `127ca91f…`)

**Experiment 3, skipping dense: SUCCESS on all three pre-registered criteria, with a major caveat.** Reproduce: `python scripts/run_experiment3_skip_dense.py` → `results/experiment3/`.

| (pooled, n = 3,343) | Value | Criterion |
| :--- | :---: | :---: |
| Dense calls skipped | **32.9%** | ≥ 20% ✅ |
| Router − always-hybrid (nDCG@10) | −0.0053 [−0.0073, −0.0032] | CI low ≥ −0.01 ✅ |
| Router − random skipping at the same rate | +0.0104 [+0.0080, +0.0129] | CI low > 0 ✅ |
| H1 (BM25 top-1/top-2 ratio alone) − random | +0.0120 [+0.0094, +0.0146] at 34.4% skipped | (comparison) |
| H2 (max-IDF alone) − random | +0.0021 [+0.0004, +0.0038] at 14.7% skipped | (comparison) |

* **Caveat: the LightGBM router adds nothing over a one-feature rule.** "Skip dense when BM25's top-1 score is far above its top-2" matches it at every skip rate on the pooled curve (`figures/cost_quality_curve.png`). The defensible claim is about the *signal* (a confident BM25 head means dense rarely helps), not the learned model.
* **The quality loss is small but real:** −0.005 nDCG@10, significantly below 0, inside the pre-set −0.01 margin.
* **Savings vary widely by dataset:** Quora 63%, SciFact 47%, NFCorpus 38%, unix 17%, FiQA 5%.
* **Transfer** (train on four datasets) still beats random skipping on SciFact, unix and Quora, but not on NFCorpus or FiQA.
* **Not yet done:** the pre-registered latency measurement (per-component timings on FiQA). Dense-call share is the only cost measure so far.

**Experiment 4, selective escalation: NEGATIVE.** Reproduce: `python scripts/run_experiment4_escalation.py` → `results/experiment4/`.
* **Primary (pooled, B = 10%):** predicted-gain − random = **+0.0008 [−0.0012, +0.0028]**. Router-confidence − random = +0.0013 [−0.0009, +0.0035].
* **Root cause: the MiniLM cross-encoder is worse than the e5 hybrid on every dataset.** Always-rerank − always-hybrid = −0.004 (NFCorpus), −0.041 (SciFact), −0.027 (FiQA), −0.008 (unix), −0.019 (Quora). Reranking helps 14.6% of queries and hurts 19.6%.
* **Predicted gain is uncorrelated with true gain** (Spearman −0.05 to +0.07 per dataset). The oracle (+0.029 at B = 10%) shows there *are* queries where reranking helps, but these features cannot find them.
* **"Quality retained" is undefined here,** because its denominator (always-rerank − hybrid) is negative. The column in `tables.md` should not be cited.
* **What it means:** the escalation premise needs a reranker that is stronger than the first stage. MiniLM-L6 (trained on MS MARCO) is not stronger than an e5 hybrid on these BEIR datasets. A follow-up with a stronger reranker (e.g. `BAAI/bge-reranker-base` or a MiniLM-L12 variant) would be a new, separately pre-registered experiment, not a re-run of this one.


## Regarding experiment issues : 

### Analysis & Verification: Low Router Confidence vs. Cross-Encoder Escalation

#### 1. Verification: Does this issue actually exist?
**Yes, the conceptual critique is valid, but the assumption about where reranking helps contains a major flaw.**

- **The Conceptual Flaw is Real:** A 3-class router’s confidence $\max_c P(c \mid q)$ measures certainty among the cheap routes ({BM25, Dense, Hybrid}). The router is "uncertain" whenever cheap routes perform similarly:
  1. *High-scoring ties:* When all cheap routes score high (e.g., $\text{nDCG} \approx 0.95$), the router is uncertain, but the cross-encoder has **zero headroom**.
  2. *Low-scoring ties:* When all cheap routes score 0, margin labeling ($|\max - \text{runner-up}| < \epsilon$) defaults to **`hybrid` with 0 regret**.
- **Empirical Prevalence Across 3,343 Test Queries:**
  - **25.3% of all test queries** (847 / 3,343) score exactly $\text{nDCG@10} = 0$ on hybrid retrieval (CQADupStack-unix: 46.5%, FiQA: 32.4%, NFCorpus: 28.5%, SciFact: 14.0%).
  - On these zero-score queries, the router’s average confidence is **0.767** (median 0.786). Only **14.5%** fall into the bottom 10% least-confident bucket. The router indeed treats them as confident "hybrid" predictions.
- **The Critical Catch (Candidate Starvation):**
  - The note claims: *"that [0-score queries] is exactly where reranking helps most."*
  - **In reality, 87.8% (744 / 847) of zero-score queries STILL score 0 after cross-encoder reranking.**
  - A cross-encoder is a reranker, not a retriever. If BM25 and Dense fail to retrieve any relevant document in the top-50 candidate pool (candidate recall = 0), the cross-encoder has nothing to promote. Reranking only rescued **12.2%** (103 / 847) of these queries.

---

#### 2. Evaluation of the Proposed Solution ("Predict Per-Route nDCG / Reranker Gain")
**Verdict: It does not work in practice. The project already tested this in Experiment 4, and it failed.**

The exact suggested fix—training a LightGBM regressor to predict continuous cross-encoder gain ($\text{nDCG}_{\text{CE}} - \text{nDCG}_{\text{Hybrid}}$)—was evaluated across all 5 datasets:
- **At 10% Escalation Budget ($B = 0.10$):**
  - Random Escalation: **0.5864**
  - Router Confidence Policy: **0.5877** ($+0.0013$ over random)
  - Predicted Gain Policy: **0.5872** ($+0.0008$ over random, 95% CI $[-0.0012, +0.0028]$)
  - Oracle Escalation (Ceiling): **0.6169** ($+0.0305$ over random)
- **Why Predicted Gain Failed:**
  1. **Near-Zero Correlation ($\rho \approx 0.04$):** Spearman rank correlation between predicted gain and true gain is practically zero across all datasets (CQADupStack: $+0.068$, SciFact: $+0.057$, FiQA: $+0.027$, Quora: $+0.021$, NFCorpus: $-0.045$). Cheap surface features cannot predict continuous cross-encoder score shifts.
  2. **Model Inversion:** MiniLM-L6 (trained on MS MARCO) is weaker overall on BEIR than `e5-small-v2` + BM25 hybrid (Always-CE = **0.5718** vs. Always-Hybrid = **0.5880**). The cross-encoder hurts 26.2% of queries and helps only 15.4%.
  3. **Pointwise nDCG Variance:** Continuous per-query nDCG is dominated by dataset-level query difficulty and qrel sparsity rather than route advantage.

---

#### 3. Better Solutions

1. **Two-Stage Cascade (Recall Viability Gate):**
   - **Gate 1 (Candidate Viability):** Check raw retrieval signals (e.g., Dense top cosine $< 0.65$ and BM25 top score $< 5.0$). If both are at floor level, candidate recall@50 is likely 0. **Abstain from escalation** because reranking cannot rescue zero-recall queries.
   - **Gate 2 (Ambiguity / Synergy):** For queries with viable candidates, check score entropy and top-1/top-2 separation in the candidate pool. Escalate only when viable documents exist but the first stage is uncertain about their order.
2. **Pairwise Ranking instead of Pointwise nDCG Regression:**
   - Train a pairwise ranking model (LambdaMART / RankNet) to optimize query escalation priority ($q_i \succ q_j \iff \Delta_{\text{CE}}(q_i) > \Delta_{\text{CE}}(q_j)$), avoiding noisy continuous MSE targets.
3. **Upgrade the Reranker Backend:**
   - Replace MiniLM-L6 with a modern cross-encoder (e.g., `BAAI/bge-reranker-base`) where $\text{Always-CE} > \text{Always-Hybrid}$. Escalation cannot succeed when the escalation target is worse on average than the baseline.

---

#### 4. Recommended Decision for the Codebase
**Stick with Option 2 ("Original: classifier + $\tau$") as the primary method, and report the findings as an honest experimental audit:**
- Keep the pre-registered 3-class classifier + $\tau$ as the primary design.
- Report "predicted reranker gain" as an experimental comparison showing that cheap features fail to correlate with continuous reranker gain ($\rho \approx 0.04$).
- Report **Oracle Escalation** ($+0.0305$ at $B=10\%$) as the theoretical ceiling, explaining the discovered root causes: **87.8% candidate pool starvation on zero-score queries** and the **MiniLM vs. E5-hybrid performance inversion**.
