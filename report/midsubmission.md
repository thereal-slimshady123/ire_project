# Mid-Submission: Confidence-Based Query Routing for Hybrid Retrieval

## 1. Project
We test whether cheap per-query features can route each query to BM25, dense, or tuned hybrid retrieval, and escalate low-confidence queries to a cross-encoder under a small budget. Evaluation uses held-out 5-fold predictions and paired bootstrap CIs on NFCorpus, SciFact, FiQA, CQADupStack-Unix and Quora (3,343 queries). The metric is nDCG@10.

## 2. Hypotheses
**Phase 1 (original):**

| | Hypothesis | Verdict | Evidence |
|:--|:--|:--|:--|
| H1 | Routing beats hybrid and RRF | Not supported (vs hybrid) | Fixed router (2d): nDCG@10 −0.0012 [−0.0025, +0.0002]. The difference isn't significant, so routing performs about the same as hybrid. Beats RRF by +0.0142. |
| H2 | Recall@100 matches hybrid | Supported | All routers within ±0.01. |
| H3 | Budgeted escalation keeps most of the rerank gain | Not supported | At a 10% budget it keeps 16% of the +0.0067 blended gain. |
| H4 | Latency near hybrid | Partly | Router +5–13 ms. 10% escalation is 2.5–7× hybrid. Always-rerank costs 368–406 ms. |

**Phase 2 (tentative and subject to change, added to try to explain the Phase 1 hypotheses failing).** Natural queries show no routing signal, so we ask when routing can work.
- **P2-H1:** routing needs query groups whose best retriever differs; natural sets have none.
- **P2-H2:** the 18 features separate keyword from paraphrase queries, so route labels are learnable.
- **P2-H3:** one shared α is a poor compromise on a mixed-style workload.
- **P2-H4:** router gain is lowest at a balanced style mix (due to majority of queries getting routed to fixed hybrid) and higher at either extreme.

## 3. What we have built
- BM25, dense (e5-small-v2), hybrid and RRF retrievers
- 18 query and retrieval-signal features
- GBM, logistic, advantage-regression and cost-aware routers
- Cross-encoder escalation with predicted-gain, confidence and τ-threshold rules
- Evaluation tooling: oracle headroom, measured latency, risk-coverage analysis, ablations

## 4. Results so far
**Phase 1:** always-hybrid (0.5880) is the best dependable cheap policy. The GBM router (0.5808) is not better than random routing at the same proportions (+0.0010 [−0.0016, +0.0036]).


## 5. What's Next
- Our next step is to run the pipeline on a different kind of dataset and check if it outperforms fixed hybrid on that, and if the results validate our Phase 2 hypotheses.
