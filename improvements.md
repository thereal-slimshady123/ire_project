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
