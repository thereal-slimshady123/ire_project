# Final Report: Confidence-Based Query Router for Hybrid Retrieval

## Abstract

<!-- Summary of research question, methods, results across NFCorpus, SciFact, FiQA, and key takeaways -->

## 1. Introduction & Motivation

<!-- IR trade-offs: BM25 latency vs. Dense semantic matching vs. Hybrid vs. Cross-encoder reranking -->

## 2. Methodology & Architecture

<!-- Route definitions (BM25 / dense / hybrid at alpha*), 3-class router, confidence threshold tau (calibrated on dev), margin labels, class-balanced regret-weighted training -->

## 3. Experimental Setup

<!-- Datasets, BEIR train/dev/test protocol, retrievers (bm25s, TAS-B, MiniLM cross-encoder: top-50 rerank, 100 returned), baselines and controls (incl. oracle escalation), metrics (nDCG@10, MRR@10, MAP@100, Recall@100), paired bootstrap + Holm-Bonferroni -->

## 4. Results & Analysis

### 4.1 Retrieval Performance (H1 & H2)
<!-- nDCG@10, MRR@10, MAP@100, Recall@100 tables and significance testing (test queries) -->

### 4.2 Latency vs. Quality Trade-offs (H3 & H4)
<!-- Mean and p95 latency comparisons, risk-coverage curves, router escalation vs. random escalation vs. oracle-escalation ceiling -->

### 4.3 Feature Ablations
<!-- Query-only vs. Retrieval-signal vs. Combined -->

## 5. Discussion & Future Work

<!-- Week-2 checkpoint (headroom minus noise floor), hybrid degradation analysis (improvements.md §1), failure modes, generalization, conclusion -->

## References
