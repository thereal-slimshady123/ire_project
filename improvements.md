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

## 2. Generating Ground Truth Labels for the Training Dataset

<!-- To be documented in the next step -->