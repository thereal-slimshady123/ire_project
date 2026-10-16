# Queries where a tuned hybrid is outperformed

Existence demonstration. Natural test queries from five BEIR/MTEB datasets, **selected because** BM25 or dense beat hybrid by ≥ 0.02 nDCG@10. Hybrid α is tuned in-sample on each full dataset (best case for hybrid). *Robust* = the win also holds against hybrid at α* ± 0.1. Retrievers: BM25 (k1 1.2, b 0.75); dense `intfloat/e5-small-v2`.

## Mean nDCG@10 on each selected set

| Set | Queries | BM25 | Dense | Hybrid (tuned α) | Robust queries |
| :--- | :---: | :---: | :---: | :---: | :---: |
| bm25_wins (all) | 214 | 0.549 | 0.204 | 0.343 | 158 |
| bm25_wins (robust only) | 158 | 0.555 | 0.203 | 0.334 | — |
| dense_wins (all) | 384 | 0.181 | 0.585 | 0.407 | 278 |
| dense_wins (robust only) | 278 | 0.180 | 0.592 | 0.398 | — |

## Per source dataset

| Dataset | Test queries | BM25 wins (robust) | Dense wins (robust) |
| :--- | :---: | :---: | :---: |
| nfcorpus | 323 | 43 (28) | 55 (40) |
| scifact | 300 | 21 (14) | 40 (32) |
| fiqa | 648 | 60 (47) | 120 (78) |
| cqadupstack-unix | 1072 | 63 (47) | 132 (100) |
| quora | 1000 | 27 (22) | 37 (28) |

## Largest margins: bm25_wins

| Dataset | Query | Winner | Hybrid | Robust |
| :--- | :--- | :---: | :---: | :---: |
| cqadupstack-unix | Linux tools to treat files as sets and perform set operations on them | 1.00 | 0.33 | no |
| quora | What are some mind-blowing facts about sex/sexuality that sound like BS, but are actually  | 1.00 | 0.36 | no |
| fiqa | Will an ETF immediately reflect a reconstitution of underlying index | 0.63 | 0.00 | yes |
| cqadupstack-unix | The amount of time since the system was last booted | 0.63 | 0.00 | yes |
| cqadupstack-unix | Failed to sign CSR with the CA root key | 1.00 | 0.39 | yes |
| quora | Someone hacked my Instagram account using some type of software. How do I block them? | 1.00 | 0.43 | yes |
| fiqa | How does pre-market trading work? | 1.00 | 0.43 | yes |
| fiqa | Making money through CFD | 0.92 | 0.39 | yes |
| quora | What is the fastest way to lose weight and body fat? | 0.82 | 0.31 | yes |
| cqadupstack-unix | SCP Automation in production environment | 0.50 | 0.00 | yes |

## Largest margins: dense_wins

| Dataset | Query | Winner | Hybrid | Robust |
| :--- | :--- | :---: | :---: | :---: |
| scifact | The genomic aberrations found in matasteses are very similar to those found in the primary | 1.00 | 0.33 | yes |
| fiqa | Should an ADR that is being delisted be sold off? | 1.00 | 0.36 | yes |
| cqadupstack-unix | LVM to extend a partition? | 0.63 | 0.00 | yes |
| cqadupstack-unix | difference between function foo() {} and foo() {} | 1.00 | 0.39 | yes |
| scifact | Modifying the epigenome in the brain affects the normal human aging process by affecting c | 1.00 | 0.39 | yes |
| cqadupstack-unix | scp from remote host fails due to login greeting set in .bashrc | 1.00 | 0.43 | yes |
| scifact | Glycolysis is one of the primary glycometabolic pathways in cells. | 1.00 | 0.43 | yes |
| scifact | PPAR-RXRs can be activated by PPAR ligands. | 1.00 | 0.43 | yes |
| cqadupstack-unix | How to make X less sensitive when right clicking? | 1.00 | 0.43 | yes |
| cqadupstack-unix | Why put things other than /home to a separate partition? | 1.00 | 0.50 | no |
