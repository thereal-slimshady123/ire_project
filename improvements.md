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

## 2. Generating Ground Truth Labels & Overcoming the Label Noise Floor

### 2.1 The Core Bottleneck: Target Label Noise in Sparse Benchmarks
In sparse retrieval benchmarks (such as SciFact with ~1.1 relevant documents per query, or NFCorpus), binary relevance over 1–2 documents causes ranking metrics to exhibit severe bimodal step-function behavior:
* A relevant document landing at Rank 3 yields $\text{nDCG@10} \approx 0.50$.
* That same document slipping to Rank 11 yields $\text{nDCG@10} = 0.0$.

In a discrete $\arg\max$ routing formulation, whether BM25 or Dense "won" on a query is frequently an artifact of minor rank jitter on a single document rather than structural retriever superiority. Furthermore, predicting differences between noisy system scores ($\Delta = \text{Score}_A - \text{Score}_B$) compounds variance ($\mathrm{Var}(\Delta) = \mathrm{Var}(A) + \mathrm{Var}(B) - 2\,\mathrm{Cov}(A,B)$). Surface score statistics alone cannot reliably disambiguate these ties.

---

### 2.2 Dual Advantage Regression Formulation
To eliminate the failure mode of treating a $\Delta = 0.60$ blowout identically to a $\Delta = 0.01$ near-tie, we replace discrete 3-class classification with **Dual Advantage Regression**.
The router learns continuous functions predicting the margin of improvement over the tuned hybrid baseline:
$$\Delta_{\text{BM25}}(q) = \text{nDCG@10}(\text{BM25}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$
$$\Delta_{\text{Dense}}(q) = \text{nDCG@10}(\text{Dense}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$

**Routing Decision Rule with Risk Threshold $\theta$:**
$$\text{Route}(q) = \begin{cases}
\text{BM25} & \text{if } \hat{\Delta}_{\text{BM25}} > \theta_{\text{bm25}} \text{ and } \hat{\Delta}_{\text{BM25}} > \hat{\Delta}_{\text{Dense}} \\
\text{Dense} & \text{if } \hat{\Delta}_{\text{Dense}} > \theta_{\text{dense}} \text{ and } \hat{\Delta}_{\text{Dense}} > \hat{\Delta}_{\text{BM25}} \\
\textbf{Hybrid} & \textbf{otherwise (conservative safe default whenever gains are marginal or uncertain)}
\end{cases}$$
The risk margins $\theta_{\text{bm25}}, \theta_{\text{dense}}$ (default $0.02$) are calibrated on training folds to guarantee queries are only diverted when the expected gain exceeds the downside risk of misrouting.

---

### 2.3 Label Robustness & Filtering
To ensure the router trains on high-signal examples:
1. **Stability Filtering:** Drop queries whose winning route flips when the hybrid weight $\alpha^*$ is shifted by $\pm 0.1$.
2. **Metric Averaging:** Train targets against a composite utility score (e.g. $0.6 \cdot \text{nDCG@10} + 0.4 \cdot \text{MRR@10}$) to smooth rank cutoff discontinuities.
3. **Denser Evaluation Grounding:** Evaluate generalization on benchmarks with dense relevance pools (e.g. TREC-COVID or TREC Deep Learning).

---

## 3. Synthetic Dataset Generation & The Definitive Experiment

### 3.1 Motivation & Scale (Scaling from 3,300 to 30,000+ Queries)
Standard BEIR subsets offer limited query counts (SciFact: 300, NFCorpus: 323, FiQA: 648). Learning expressive representations from 3,343 total queries leads to severe overfitting or underfitting. 
Using synthetic query generation (`doc2query`, InPars, Promptagator), we scale the training corpus by 10× to 30,000+ queries in ~1 hour of GPU time, while preserving 100% natural, untouched test splits for evaluation.

---

### 3.2 Controlled Style Generation Protocol
Passages are sampled from the corpus and converted into queries using dual-style generation:
1. **Lexical / Entity Style (50%):** Prompts targeting specific entities, technical identifiers, error codes, and gene symbols (e.g., *"Kubernetes CrashLoopBackOff error 137"*).
2. **Semantic / Paraphrastic Style (50%):** Prompts asking conceptual, conversational, or exploratory questions without reusing source vocabulary (e.g., *"Why is my container repeatedly dying from RAM limits?"*).

Ground-truth label assignment is automatic: each synthetic query treats its source passage as the positive relevant document ($y = 1$).

---

### 3.3 Leakage Prevention & Anti-Contamination Protocols
* **Strict Corpus Partitioning:** Documents that serve as relevant targets for ANY natural test query are strictly excluded from synthetic query generation.
* **Generator Artifact Isolation:** Synthetic queries are strictly restricted to the training fold. Validation and final reporting occur exclusively on natural human queries.

---

### 3.4 Live Empirical Validation (Proof-of-Concept Pilot)
The end-to-end synthetic data generation, advantage labeling, and router training pipeline was implemented and validated in `scripts/demo_synthetic_routing.py` on an NVIDIA GeForce RTX 2050 GPU using `sentence-transformers/all-MiniLM-L6-v2` and LightGBM:

1. **Retriever Breakdown by Synthetic Query Style:**
   | Query Style | BM25 Score | Dense Score | Hybrid Score | $\Delta_{\text{BM25}}$ | $\Delta_{\text{Dense}}$ |
   | :--- | :---: | :---: | :---: | :---: | :---: |
   | **Lexical / Entity** | **1.0000** | 0.9754 | **1.0000** | 0.0000 | -0.0246 |
   | **Semantic / Concept** | 0.5083 | **0.8726** | 0.7187 | -0.2104 | **+0.1539** |

2. **Held-Out Test Performance:**
   * **Fixed Hybrid Score:** `0.7312`
   * **Advantage Router Score:** `0.8015` (**Gain over Hybrid: +0.0703**)
   * **Theoretical Oracle Headroom:** `0.9077` (Total available: `+0.1766`)
   * **Headroom Captured:** **39.8% of theoretical oracle headroom** on unseen test queries.
   * The router successfully routed high-semantic queries away from noisy Hybrid combinations to Dense, and maintained Hybrid where gains were ambiguous.

---

### 3.5 The Definitive Experiment Design ("The Experiment That Settles It")
1. **Scale:** Generate 30,000 synthetic training queries across NFCorpus, SciFact, and FiQA corpora.
2. **Features:** Combine 384-d dense query embeddings (`sentence-transformers`), cheap lexical content signals (top-1 text overlap), and retrieval interaction signals (Jaccard, top gap, entropy).
3. **Objective:** Fit Dual Advantage Regressors on $\Delta_{\text{BM25}}$ and $\Delta_{\text{Dense}}$ with stability filtering.
4. **Evaluation:** Pre-registered evaluation on untouched natural test splits.
   * **If positive:** Proves that advantage regression + synthetic scaling solves query routing without costly LLM calls (DAT).
   * **If negative:** Forms a definitive, publishable empirical proof of the fundamental limits of query routing in hybrid retrieval under high data scale.

