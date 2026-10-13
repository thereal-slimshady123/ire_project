# Project Improvements & Experimental Designs

**Status of each section (see `plan.md` §0 for the binding contracts):**
- **§1 (hybrid degradation analysis)** is part of the main project. It runs on train + dev queries next to the Week-2 headroom checkpoint and motivates routing.
- **§2 (advantage regression)** and **§3 (synthetic training queries)** are **stretch / future-work threads** (plan.md Phase 7, Thread 6). They are *not* the main method. The main method is the 3-class classifier with confidence-based escalation.

---

## 1. Experiment: Demonstrating That Always Picking Hybrid Degrades Accuracy

### 1.1 Objective & Research Question
**Research Question:** Does a static convex combination of sparse and dense retrieval scores ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) degrade ranking quality on identifiable subsets of queries compared to single-retriever routes?

**Core Claim:** Fixed-weight hybrid retrieval may achieve the highest *mean* nDCG across a dataset, yet show **query-level negative transfer** (fusion degradation) on a sizable fraction of queries, where pure BM25 or pure Dense outperforms Hybrid.

---

### 1.2 Formal Hypotheses
* **$H_1$ (Degradation Prevalence):** On at least 15% of queries, $\max(\text{nDCG@10}_{\text{BM25}}, \text{nDCG@10}_{\text{Dense}}) - \text{nDCG@10}_{\text{Hybrid}} \ge \epsilon$, with $\epsilon$ equal to the labeling default ($0.05$, `configs/labels.yaml`).
* **$H_2$ (Oracle Headroom above Noise):** The oracle router's headroom over the best-tuned fixed-$\alpha$ hybrid, **minus the noise floor**, is $\ge 2.0$ nDCG points. The noise floor is the same oracle computed between $\text{Hybrid}_{\alpha^*}$ and $\text{Hybrid}_{\alpha^* \pm 0.1}$, which differ in no systematic way, so any "headroom" between them is rank jitter. No significance test of oracle vs. hybrid is run: the oracle is $\ge$ hybrid on every query by construction, so such a test always "passes".
* **$H_3$ (Attributable Failure Modes):**
  * Degradation where **BM25 > Hybrid** correlates with high rare-entity density and high max-IDF (dense introduces semantic drift/false positives).
  * Degradation where **Dense > Hybrid** correlates with low keyword overlap and low BM25 top-1 score gap (BM25 introduces spurious lexical noise).
  * Severity of degradation peaks when retriever agreement (top-$k$ Jaccard overlap) is near zero.

---

### 1.3 Variables & Controls
* **Independent Variables:**
  * Retrieval Route: BM25-only (`bm25s`, stemmed), Dense-only (TAS-B), and Hybrid.
  * Hybrid Weight $\alpha$: Evaluated across grid $[0.0, 0.1, \dots, 1.0]$. The baseline hybrid uses the **optimal $\alpha^*$ tuned on the train split**, so that Hybrid is never a "strawman".
* **Normalization Control:**
  * BM25 scores and TAS-B dot-product scores are on different, unbounded scales. Hybrid is computed over the union $C$ of both top-100 lists, with both raw scores computed exactly for every $d \in C$, and each min-max normalized per query into $[0, 1]$ before combination:
    $$\tilde{s}(d) = \frac{s(d) - \min_{d' \in C} s(d')}{\max_{d' \in C} s(d') - \min_{d' \in C} s(d') + \delta}$$
* **Dependent Variables:**
  * Primary: $\text{nDCG@10}(r, q)$ per query $q$ for route $r$.
  * Secondary: $\text{MRR@10}(r, q)$, $\text{MAP@100}(r, q)$, $\text{Recall@100}(r, q)$.

---

### 1.4 Step-by-Step Experimental Protocol

All of §1 uses **train + dev** queries, so test queries stay untouched for final reporting.

#### Step 1: Execution & Run Persistence
1. For each dataset (NFCorpus, SciFact, FiQA), execute:
   * BM25 retrieval for top-100 candidates per query.
   * Dense retrieval for top-100 candidates per query.
2. For each query, score the union exactly, normalize, and compute the Hybrid run for $\alpha \in \{0.0, 0.1, \dots, 1.0\}$.
3. Select $\alpha^*$ on the train split.

#### Step 2: Per-Query Metric Computation
Using ground-truth qrels, compute per-query ranking metrics for each route:
$$\Delta_{\text{BM25}}(q) = \text{nDCG@10}(\text{BM25}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$
$$\Delta_{\text{Dense}}(q) = \text{nDCG@10}(\text{Dense}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$
$$\Delta_{\text{MaxSingle}}(q) = \max(\text{nDCG@10}(\text{BM25}, q), \text{nDCG@10}(\text{Dense}, q)) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$

#### Step 3: Query Partitioning (Win / No-Clear-Winner)
Sort the three route scores for query $q$ as $s_{(1)} \ge s_{(2)} \ge s_{(3)}$, with $\epsilon = 0.05$. Every query falls into exactly one of four partitions:
1. **BM25 Dominant (Dense Noise Pollution):** BM25 has $s_{(1)}$ and $s_{(1)} - s_{(2)} \ge \epsilon$.
2. **Dense Dominant (BM25 Lexical Noise Pollution):** Dense has $s_{(1)}$ and $s_{(1)} - s_{(2)} \ge \epsilon$.
3. **Hybrid Dominant (Synergistic Fusion):** Hybrid has $s_{(1)}$ and $s_{(1)} - s_{(2)} \ge \epsilon$.
4. **No Clear Winner:** $s_{(1)} - s_{(2)} < \epsilon$. This includes exact ties and queries where every route scores 0.

These partitions match the margin labels exactly (partition 4 is the hybrid-fallback label). Separately, a query counts as **degraded by hybrid** (used in $H_1$) when $\Delta_{\text{MaxSingle}}(q) \ge \epsilon$, whichever single retriever is better. A degraded query can be BM25-dominant, Dense-dominant, or (when BM25 and Dense tie above hybrid) No Clear Winner.

#### Step 4: Statistical Testing
* Paired bootstrap ($B=10{,}000$) of BM25 vs. $\text{Hybrid}_{\alpha^*}$ and of Dense vs. $\text{Hybrid}_{\alpha^*}$ over **all** queries.
* Per-partition differences are reported **descriptively** (mean, CI) only. A partition is selected because one route won there, so testing that route against hybrid *within* the partition is circular.
* Oracle vs. hybrid is reported as headroom minus noise floor (see $H_2$), without a significance test.

---

### 1.5 Diagnostic Analysis: Correlating Failure Modes with Query Features
To find *why* hybrid hurts accuracy, correlate per-query degradation $\Delta_{\text{MaxSingle}}(q)$ with measurable feature signals:
1. **Lexical Specificity:** Compute Max-IDF and rare-term density.
   * *Hypothesis:* Queries in the top quartile of Max-IDF have significantly higher $\Delta_{\text{BM25}}$ (dense dilutes precision).
2. **Retriever Disagreement (Jaccard Overlap @ 10):**
   $$J(q) = \frac{|\text{Top10}_{\text{BM25}}(q) \cap \text{Top10}_{\text{Dense}}(q)|}{|\text{Top10}_{\text{BM25}}(q) \cup \text{Top10}_{\text{Dense}}(q)|}$$
   * *Hypothesis:* When $J(q) = 0$, fusion is high-risk; hybrid assigns intermediate scores to discordant items, demoting true positives from both sources.
3. **Score Separation Gap:** Top-1 minus top-2 score for each retriever. If BM25 has a large gap but Dense is flat, combining them can dilute BM25's confident hit.

---

### 1.6 Results Reporting & Visualizations

1. **Query Partition Table:**
   | Dataset | Total Queries | Hybrid Dominant (%) | BM25 Dominant (%) | Dense Dominant (%) | No Clear Winner (%) | Degraded by Hybrid (%) |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
   | NFCorpus | $N$ | - | - | - | - | - |
   | SciFact | $N$ | - | - | - | - | - |
   | FiQA | $N$ | - | - | - | - | - |

   The first four percentage columns sum to 100%. "Degraded by Hybrid" overlaps them.

2. **Oracle Headroom Table:**
   | Dataset | BM25 nDCG | Dense nDCG | Best Hybrid ($\alpha^*$) | Oracle Router | Headroom ($\Delta$) | Noise Floor | Headroom − Noise |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | NFCorpus | - | - | - | - | - | - | - |
   | SciFact | - | - | - | - | - | - | - |
   | FiQA | - | - | - | - | - | - | - |

3. **Plots to Generate:**
   * **Degradation Scatter Plot:** Per-query plot with x-axis $= \max(\text{nDCG}_{\text{BM25}}, \text{nDCG}_{\text{Dense}})$ and y-axis $= \text{nDCG}_{\text{Hybrid}}$. Points below the $y = x$ identity line are queries where hybrid loses to a single retriever.
   * **CDF of Delta:** Cumulative distribution of $\Delta_{\text{MaxSingle}}(q)$.
   * **Feature Distribution Boxplots:** Max-IDF and Jaccard overlap grouped by partition.

---

## 2. [Stretch] Ground Truth Labels & the Label Noise Floor

*Future-work thread. The main method keeps margin-based 3-class labels (plan.md §0.6).*

### 2.1 The Core Bottleneck: Target Label Noise in Sparse Benchmarks
On benchmarks with very few relevant documents per query (SciFact ~1.1, FiQA ~2.6), binary relevance makes ranking metrics behave like step functions:
* A single relevant document landing at Rank 3 yields $\text{nDCG@10} = 0.50$.
* That same document slipping to Rank 11 yields $\text{nDCG@10} = 0.0$.

NFCorpus is different: it has graded relevance and ~38 relevant documents per query, so its nDCG@10 is much smoother and this argument applies to it far less.

On the sparse datasets, in a discrete $\arg\max$ formulation, whether BM25 or Dense "won" a query is often an artifact of minor rank jitter on a single document rather than structural retriever superiority. Predicting differences between noisy system scores ($\Delta = \text{Score}_A - \text{Score}_B$) also compounds variance ($\mathrm{Var}(\Delta) = \mathrm{Var}(A) + \mathrm{Var}(B) - 2\,\mathrm{Cov}(A,B)$).

---

### 2.2 Dual Advantage Regression Formulation
To avoid treating a $\Delta = 0.60$ blowout identically to a $\Delta = 0.01$ near-tie, this thread replaces 3-class classification with **Dual Advantage Regression**: the router predicts the margin of improvement over the tuned hybrid baseline:
$$\Delta_{\text{BM25}}(q) = \text{nDCG@10}(\text{BM25}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$
$$\Delta_{\text{Dense}}(q) = \text{nDCG@10}(\text{Dense}, q) - \text{nDCG@10}(\text{Hybrid}_{\alpha^*}, q)$$

**Routing Decision Rule with Risk Threshold $\theta$:**
$$\text{Route}(q) = \begin{cases}
\text{BM25} & \text{if } \hat{\Delta}_{\text{BM25}} > \theta_{\text{bm25}} \text{ and } \hat{\Delta}_{\text{BM25}} > \hat{\Delta}_{\text{Dense}} \\
\text{Dense} & \text{if } \hat{\Delta}_{\text{Dense}} > \theta_{\text{dense}} \text{ and } \hat{\Delta}_{\text{Dense}} > \hat{\Delta}_{\text{BM25}} \\
\textbf{Hybrid} & \textbf{otherwise (conservative safe default whenever gains are marginal or uncertain)}
\end{cases}$$
The risk margins $\theta_{\text{bm25}}, \theta_{\text{dense}}$ (default $0.02$) are selected on dev, so that queries are only diverted when the predicted gain exceeds the downside risk of misrouting.

**Open issue:** this formulation produces no class probabilities, so the main method's confidence-based escalation does not apply to it directly. An escalation rule would have to be designed for it (e.g., escalate when the predicted hybrid quality is low).

---

### 2.3 Label Robustness & Filtering
To make the router train on high-signal examples (all of these apply to **training queries only**, never to dev or test):
1. **Stability Filtering:** Drop training queries whose winning route flips when the hybrid weight $\alpha^*$ is shifted by $\pm 0.1$.
2. **Metric Averaging:** Train targets against a composite utility score (e.g. $0.6 \cdot \text{nDCG@10} + 0.4 \cdot \text{MRR@10}$) to smooth rank cutoff discontinuities.
3. **Denser Evaluation Grounding:** Evaluate generalization on benchmarks with dense relevance pools (e.g. TREC-COVID or TREC Deep Learning).

---

## 3. [Stretch] Synthetic Training Queries

*Future-work thread. The main method trains on natural BEIR train queries (plan.md §0.2).*

### 3.1 Motivation & Scale
BEIR test sets are small (SciFact: 300, NFCorpus: 323, FiQA: 648; 1,271 in total). The main method already trains on the official **natural train splits** (roughly 0.8k / 2.6k / 5.5k queries; confirm exact counts after download). This thread asks whether **additional** synthetic queries (`doc2query`, InPars, Promptagator) help further, for example scaling to ~30,000 training queries. Generation time on the available RTX 2050 (4 GB) has not been measured. Test splits stay 100% natural.

---

### 3.2 Controlled Style Generation Protocol
Passages are sampled from the corpus and converted into queries using dual-style generation:
1. **Lexical / Entity Style (50%):** Prompts targeting specific entities, technical identifiers, error codes, and gene symbols (e.g., *"Kubernetes CrashLoopBackOff error 137"*).
2. **Semantic / Paraphrastic Style (50%):** Prompts asking conceptual, conversational, or exploratory questions without reusing source vocabulary (e.g., *"Why is my container repeatedly dying from RAM limits?"*).

Ground-truth label assignment is automatic: each synthetic query treats its source passage as the positive relevant document ($y = 1$).

**Known risks:**
* doc2query-style queries copy vocabulary from their source passage, which systematically favours BM25.
* Forcing a 50/50 style split imposes an artificial distribution, and the style may be trivially detectable (e.g. every semantic query is a question).
* On NFCorpus, the source passage is rarely the only relevant document, so "only the source is relevant" creates many false negatives.

Required checks: compare route-label and feature distributions of synthetic vs. natural queries, and always report natural-only vs. natural + synthetic training.

---

### 3.3 Leakage Prevention & Anti-Contamination Protocols
* **Strict Corpus Partitioning:** Documents that are relevant to ANY natural dev or test query are excluded from synthetic query generation.
* **Generator Artifact Isolation:** Synthetic queries are restricted to training. Selection uses natural dev queries and final reporting uses natural test queries only.

---

### 3.4 Toy Smoke Test (not evidence)
`scripts/demo_synthetic_routing.py` checks that the code path (BM25 / dense / hybrid → per-query scores → LightGBM advantage regressors → routing) runs end to end, using `sentence-transformers/all-MiniLM-L6-v2` and LightGBM. **Its numbers support no claim**, because:
* the 60 queries are **hand-written**, not generated: no generation model is used;
* the test split has 12 queries, and both queries of one document can land on both sides of the split;
* query style is trivially separable (every "semantic" query is a question), so the router can learn sentence form instead of anything about retrieval;
* half of the 60-document corpus are near-duplicate `_var` documents;
* hybrid uses an untuned $\alpha = 0.5$;
* with 48 training rows and LightGBM's default `min_child_samples=20`, each tree has at most two leaves;
* the router chose BM25 for zero test queries, so BM25 routing was not exercised.

Output, recorded only so the run can be checked for reproducibility:

| Query Style | BM25 Score | Dense Score | Hybrid Score | $\Delta_{\text{BM25}}$ | $\Delta_{\text{Dense}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Lexical / Entity | 1.0000 | 0.9754 | 1.0000 | 0.0000 | -0.0246 |
| Semantic / Question | 0.5083 | 0.8726 | 0.7187 | -0.2104 | +0.1539 |

Held-out (n = 12): hybrid 0.7312, router 0.8015, oracle 0.9077; routes chosen: Dense 6, Hybrid 6.

---

### 3.5 Full-Scale Design (if this thread is pursued)
1. **Scale:** Generate synthetic training queries from the NFCorpus, SciFact, and FiQA corpora (excluding dev/test-relevant documents, §3.3).
2. **Features:** The main method's hand-crafted features, optionally plus dense query embeddings (768-d for TAS-B). Embeddings overlap with the latent-representation thread (plan.md Thread 1) and must be reported as a separate ablation.
3. **Objective:** Fit Dual Advantage Regressors on $\Delta_{\text{BM25}}$ and $\Delta_{\text{Dense}}$ with stability filtering (training queries only).
4. **Evaluation:** Pre-registered comparison on the untouched natural test splits: natural-only training vs. natural + synthetic training, and advantage regression vs. the main 3-class router.
   * **If positive:** evidence that synthetic scaling and/or advantage regression improve routing without per-query LLM calls (cf. DAT).
   * **If negative:** an informative negative result about the limits of query routing in hybrid retrieval at larger training scale.
