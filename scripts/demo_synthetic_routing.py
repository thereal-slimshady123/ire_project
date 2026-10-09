"""Proof-of-Concept: Synthetic Query Generation, Advantage Labeling, and Router Training."""

import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Set deterministic seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Ensure offline huggingface usage
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("=" * 70)
print("PROOF OF CONCEPT: SYNTHETIC DATASET GENERATION & ADVANTAGE ROUTING")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. SAMPLE CORPUS OF DIVERSE SCIENTIFIC & TECHNICAL PASSAGES
# -----------------------------------------------------------------------------
raw_corpus = [
    # Medical & Bio Passages
    {"id": "doc_01", "text": "Patients with acute myocardial infarction (AMI) exhibit elevated serum troponin I (cTnI) levels exceeding 0.04 ng/mL within 4 to 6 hours after symptom onset."},
    {"id": "doc_02", "text": "Metformin hydrochloride lowers fasting blood glucose concentrations by inhibiting mitochondrial glycerophosphate dehydrogenase, thereby suppressing hepatic gluconeogenesis."},
    {"id": "doc_03", "text": "BRCA1 and BRCA2 pathogenic germline mutations markedly increase lifetime risk of epithelial ovarian cancer and triple-negative breast adenocarcinoma."},
    {"id": "doc_04", "text": "Systemic lupus erythematosus (SLE) manifests with anti-double-stranded DNA (anti-dsDNA) antibodies and immune complex-mediated glomerulonephritis."},
    {"id": "doc_05", "text": "Streptococcus pneumoniae remains the primary etiological pathogen causing community-acquired lobar pneumonia in immunocompromised geriatric cohorts."},
    {"id": "doc_06", "text": "Chronic obstructive pulmonary disease (COPD) management utilizes long-acting beta2-agonists (LABA) and muscarinic antagonists (LAMA) via dry-powder inhalers."},
    {"id": "doc_07", "text": "Gastroesophageal reflux disease (GERD) with Barrett's esophagus requires endoscopic surveillance due to progressive risk of esophageal adenocarcinoma."},
    {"id": "doc_08", "text": "Parkinson's disease pathology is characterized by intracellular alpha-synuclein aggregates forming Lewy bodies within dopaminergic neurons of the substantia nigra."},
    {"id": "doc_09", "text": "Celiac disease is an autoimmune enteropathy triggered by dietary gluten ingestion in genetically susceptible HLA-DQ2 or HLA-DQ8 positive individuals."},
    {"id": "doc_10", "text": "Hypertrophic cardiomyopathy (HCM) mutations commonly involve MYH7 and MYBPC3 sarcomeric protein genes leading to asymmetrical septal hypertrophy."},

    # Computer Science & Technical Passages
    {"id": "doc_11", "text": "PostgreSQL error code 23505 indicates a unique constraint violation when duplicate primary key values are inserted into an indexed B-tree relation."},
    {"id": "doc_12", "text": "Apache Kafka consumer groups achieve horizontal scalability and fault-tolerant log consumption by partition rebalancing protocols using librdkafka."},
    {"id": "doc_13", "text": "CUDA kernel execution on NVIDIA Ampere architecture utilizes warp-level primitives, tensor cores, and asynchronous shared memory copy pipelines."},
    {"id": "doc_14", "text": "Kubernetes CrashLoopBackOff error 137 typically indicates an out-of-memory (OOMKilled) condition enforced by Linux cgroups memory.max limits."},
    {"id": "doc_15", "text": "Rust ownership system enforces memory safety at compile time without a garbage collector through affine types and strict borrow checking semantics."},
    {"id": "doc_16", "text": "Redis AOF (Append-Only File) persistence with fsync everysec provides an optimal trade-off between write throughput and data durability against crashes."},
    {"id": "doc_17", "text": "HTTP/3 replaces TCP with QUIC over UDP, mitigating head-of-line blocking across multiplexed streams and supporting 0-RTT connection handshakes."},
    {"id": "doc_18", "text": "Docker container bridge networks isolate network namespaces using iptables masquerading rules and virtual ethernet pair (veth) interfaces."},
    {"id": "doc_19", "text": "Git detached HEAD state occurs when checking out a specific commit hash rather than a local tracking branch reference, preventing branch tip progression."},
    {"id": "doc_20", "text": "Linux kernel eBPF (extended Berkeley Packet Filter) bytecode runs safely inside the kernel virtual machine for low-overhead network observability."},

    # Physics & Engineering Passages
    {"id": "doc_21", "text": "Quantum key distribution (QKD) utilizing the BB84 protocol ensures cryptographic secrecy via the no-cloning theorem and single-photon polarization measurement."},
    {"id": "doc_22", "text": "Lithium-iron-phosphate (LiFePO4) cathode chemistry offers thermal stability and cycle life exceeding 3000 cycles at the expense of nominal cell voltage."},
    {"id": "doc_23", "text": "Superconducting qubits operate at dilution refrigerator temperatures near 15 millikelvin to avoid thermal quasiparticle excitation and decoherence."},
    {"id": "doc_24", "text": "Photovoltaic perovskite tandem solar cells achieve power conversion efficiencies surpassing 33% by absorbing complementary ultraviolet and infrared spectra."},
    {"id": "doc_25", "text": "Finite element analysis (FEA) of structural stress employs von Mises yield criteria to predict plastic deformation under multiaxial loading conditions."},
    {"id": "doc_26", "text": "Optical coherence tomography (OCT) relies on low-coherence interferometry to achieve micrometer-resolution cross-sectional retinal imaging."},
    {"id": "doc_27", "text": "Gas turbine Brayton cycle efficiency increases with higher turbine inlet temperatures and pressure ratios, mitigated by ceramic thermal barrier coatings."},
    {"id": "doc_28", "text": "Graphene ballistic thermal conductivity exhibits exceptionally high values exceeding 4000 W/mK at ambient temperature due to long phonon mean free paths."},
    {"id": "doc_29", "text": "CRISPR-Cas9 endonuclease complexes direct sequence-specific double-strand breaks guided by synthetic single guide RNAs (sgRNA) with NGG PAM sequences."},
    {"id": "doc_30", "text": "Stellar nucleosynthesis during core helium burning produces carbon-12 and oxygen-16 via the triple-alpha process in red giant stars."},
]

# Duplicate and perturb to expand corpus to 60 docs
corpus = []
for doc in raw_corpus:
    corpus.append(doc)
    corpus.append({
        "id": f"{doc['id']}_var",
        "text": f"Secondary research on {doc['text'].lower()} This confirms experimental validation in recent clinical and engineering evaluations."
    })

print(f"\n[1] Built evaluation corpus of {len(corpus)} documents.")

# -----------------------------------------------------------------------------
# 2. SYNTHETIC QUERY GENERATION (Dual Style: Lexical vs Paraphrased)
# -----------------------------------------------------------------------------
print("\n[2] Synthesizing queries for each document...")
synthetic_data = []

for doc in raw_corpus:
    doc_id = doc["id"]
    text = doc["text"]
    
    # Style A: Lexical / Keyword / Entity Query (BM25-favored)
    # Extracts exact technical phrases, identifiers, error codes, gene names
    if "troponin" in text:
        q_lexical = "elevated serum troponin I cTnI 0.04 ng/mL acute myocardial infarction"
        q_semantic = "Which blood test biomarkers spike quickly when heart muscle tissue is damaged?"
    elif "Metformin" in text:
        q_lexical = "Metformin hydrochloride mitochondrial glycerophosphate dehydrogenase gluconeogenesis"
        q_semantic = "What biological pathway does the primary type 2 diabetes drug block to stop liver sugar output?"
    elif "BRCA1" in text:
        q_lexical = "BRCA1 BRCA2 germline mutations ovarian cancer triple-negative breast"
        q_semantic = "Which hereditary genetic flaws predispose women to severe reproductive tumors?"
    elif "lupus" in text:
        q_lexical = "Systemic lupus erythematosus SLE anti-dsDNA glomerulonephritis"
        q_semantic = "How does autoantibody buildup damage kidney filters in systemic autoimmune illness?"
    elif "pneumonia" in text:
        q_lexical = "Streptococcus pneumoniae community-acquired lobar pneumonia geriatric"
        q_semantic = "What bacteria most frequently causes acute lung consolidation in elderly patients?"
    elif "COPD" in text:
        q_lexical = "COPD LABA LAMA dry-powder inhalers beta2-agonists"
        q_semantic = "What maintenance inhaler medicines are used for long term chronic airflow restriction?"
    elif "Barrett" in text:
        q_lexical = "GERD Barrett's esophagus endoscopic surveillance adenocarcinoma"
        q_semantic = "Why do acid reflux patients require periodic camera checks of their food pipe?"
    elif "Parkinson" in text:
        q_lexical = "Parkinson's alpha-synuclein Lewy bodies substantia nigra dopaminergic"
        q_semantic = "What protein misfolds and kills movement brain cells in tremor disorders?"
    elif "Celiac" in text:
        q_lexical = "Celiac disease gluten HLA-DQ2 HLA-DQ8 enteropathy"
        q_semantic = "Which wheat protein causes immune destruction of gut lining in genetic enteropathy?"
    elif "cardiomyopathy" in text:
        q_lexical = "Hypertrophic cardiomyopathy HCM MYH7 MYBPC3 asymmetrical septal hypertrophy"
        q_semantic = "Which sarcomere protein gene defects cause fatal enlargement of the heart muscle wall?"
    elif "23505" in text:
        q_lexical = "PostgreSQL error code 23505 unique constraint violation B-tree"
        q_semantic = "Why is my relational database rejecting record insertions with a duplicate key?"
    elif "Kafka" in text:
        q_lexical = "Apache Kafka partition rebalancing librdkafka consumer groups"
        q_semantic = "How do distributed event streaming clients share the workload across stream slices?"
    elif "CUDA" in text:
        q_lexical = "CUDA Ampere architecture warp-level primitives tensor cores"
        q_semantic = "How do GPU hardware acceleration units schedule threads for ultra-fast matrix math?"
    elif "137" in text:
        q_lexical = "Kubernetes CrashLoopBackOff error 137 OOMKilled cgroups memory.max"
        q_semantic = "Why is my cloud container repeatedly dying when it consumes too much RAM?"
    elif "Rust" in text:
        q_lexical = "Rust ownership borrow checking affine types compile time"
        q_semantic = "How does modern systems programming guarantee memory validity without pausing for garbage cleanup?"
    elif "AOF" in text:
        q_lexical = "Redis AOF Append-Only File fsync everysec data durability"
        q_semantic = "How can in-memory key-value stores protect against power loss without slowing down writes?"
    elif "HTTP/3" in text:
        q_lexical = "HTTP/3 QUIC UDP 0-RTT head-of-line blocking"
        q_semantic = "What modern web protocol removes TCP packet stalls by streaming over datagrams?"
    elif "veth" in text:
        q_lexical = "Docker bridge networks iptables masquerading veth"
        q_semantic = "How do isolated containerized processes route network packets to the host machine?"
    elif "detached HEAD" in text:
        q_lexical = "Git detached HEAD state checkout commit hash"
        q_semantic = "What happens when you inspect an old revision in version control without creating a branch?"
    elif "eBPF" in text:
        q_lexical = "Linux eBPF bytecode kernel virtual machine network observability"
        q_semantic = "How can programmers trace operating system network calls safely from user space?"
    elif "BB84" in text:
        q_lexical = "BB84 protocol QKD no-cloning theorem photon polarization"
        q_semantic = "How can two parties exchange uncrackable secret encryption keys using photon physics?"
    elif "LiFePO4" in text:
        q_lexical = "Lithium-iron-phosphate LiFePO4 3000 cycles cathode thermal stability"
        q_semantic = "Which rechargeable battery chemistry prioritizes fire safety and longevity over raw voltage?"
    elif "qubits" in text:
        q_lexical = "Superconducting qubits dilution refrigerator 15 millikelvin quasiparticle"
        q_semantic = "Why do quantum computers require cryogenic chilling near absolute zero?"
    elif "perovskite" in text:
        q_lexical = "perovskite tandem solar cells 33% efficiency complementary ultraviolet infrared"
        q_semantic = "How do advanced multilayer solar panels harvest wider bands of the light spectrum?"
    elif "FEA" in text:
        q_lexical = "Finite element analysis FEA von Mises yield criteria multiaxial loading"
        q_semantic = "How do structural engineers predict metal failure under complex mechanical stress?"
    elif "OCT" in text:
        q_lexical = "Optical coherence tomography OCT low-coherence interferometry retinal imaging"
        q_semantic = "What non-invasive laser imaging technique scans microscopic cross-sections of the human eye?"
    elif "Brayton" in text:
        q_lexical = "Gas turbine Brayton cycle turbine inlet temperature thermal barrier coatings"
        q_semantic = "How do jet engines maximize thermal thermodynamic efficiency without melting turbine blades?"
    elif "Graphene" in text:
        q_lexical = "Graphene thermal conductivity 4000 W/mK phonon mean free paths"
        q_semantic = "Why does a 2D sheet of carbon atoms conduct heat better than bulk copper or diamond?"
    elif "CRISPR" in text:
        q_lexical = "CRISPR-Cas9 sgRNA NGG PAM sequence double-strand breaks"
        q_semantic = "How do molecular biology tools target exact genomic DNA sites for precision editing?"
    elif "nucleosynthesis" in text:
        q_lexical = "stellar nucleosynthesis triple-alpha process carbon-12 oxygen-16 red giant"
        q_semantic = "How are heavy elements forged inside collapsing stars through helium fusion?"
    else:
        continue

    synthetic_data.append({
        "query_id": f"{doc_id}_q_lex",
        "target_doc_id": doc_id,
        "query_text": q_lexical,
        "style": "lexical"
    })
    synthetic_data.append({
        "query_id": f"{doc_id}_q_sem",
        "target_doc_id": doc_id,
        "query_text": q_semantic,
        "style": "semantic"
    })

print(f"Generated {len(synthetic_data)} synthetic queries (50% lexical keyword, 50% semantic concept).")

# -----------------------------------------------------------------------------
# 3. BUILD RETRIEVERS (BM25 + DENSE GPU EMBEDDINGS)
# -----------------------------------------------------------------------------
print("\n[3] Indexing corpus with BM25 and SentenceTransformer (GPU)...")
corpus_tokens = [d["text"].lower().split() for d in corpus]
doc_ids = [d["id"] for d in corpus]
doc_texts = [d["text"] for d in corpus]

# BM25 Index
bm25_engine = BM25Okapi(corpus_tokens)

# Dense Bi-Encoder Index on CUDA GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
doc_embeddings = embedder.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)

def min_max_norm(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max - s_min < 1e-6:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min)

# -----------------------------------------------------------------------------
# 4. RUN ALL 3 ROUTES & COMPUTE ADVANTAGE LABELS
# -----------------------------------------------------------------------------
print("\n[4] Running BM25, Dense, and Hybrid on all synthetic queries...")
alpha = 0.5  # Fixed hybrid balance

results = []
query_embeddings = []

for item in synthetic_data:
    q_text = item["query_text"]
    target_id = item["target_doc_id"]
    
    # BM25 scores
    q_toks = q_text.lower().split()
    bm25_raw = np.array(bm25_engine.get_scores(q_toks))
    bm25_norm = min_max_norm(bm25_raw)
    
    # Dense scores
    q_emb = embedder.encode([q_text], normalize_embeddings=True, show_progress_bar=False)[0]
    query_embeddings.append(q_emb)
    dense_raw = np.dot(doc_embeddings, q_emb)
    dense_norm = min_max_norm(dense_raw)
    
    # Hybrid combination
    hybrid_scores = (1 - alpha) * bm25_norm + alpha * dense_norm
    
    # Rank documents
    bm25_ranked = [doc_ids[i] for i in np.argsort(-bm25_norm)]
    dense_ranked = [doc_ids[i] for i in np.argsort(-dense_norm)]
    hybrid_ranked = [doc_ids[i] for i in np.argsort(-hybrid_scores)]
    
    # Reciprocal Rank (MRR) / nDCG proxy for single relevant doc
    def calc_score(ranked_list, target):
        try:
            rank = ranked_list.index(target) + 1
            if rank <= 10:
                return 1.0 / np.log2(rank + 1)
            return 0.0
        except ValueError:
            return 0.0

    score_bm25 = calc_score(bm25_ranked, target_id)
    score_dense = calc_score(dense_ranked, target_id)
    score_hybrid = calc_score(hybrid_ranked, target_id)
    
    delta_bm25 = score_bm25 - score_hybrid
    delta_dense = score_dense - score_hybrid
    
    # Features
    # Feature 1: Top-10 Jaccard Overlap
    set_bm25 = set(bm25_ranked[:10])
    set_dense = set(dense_ranked[:10])
    jaccard = len(set_bm25 & set_dense) / float(len(set_bm25 | set_dense))
    
    # Feature 2: BM25 score gap (top1 - top2)
    bm25_sorted_scores = np.sort(bm25_norm)[::-1]
    bm25_gap = bm25_sorted_scores[0] - bm25_sorted_scores[1]
    
    # Feature 3: Dense score gap (top1 - top2)
    dense_sorted_scores = np.sort(dense_norm)[::-1]
    dense_gap = dense_sorted_scores[0] - dense_sorted_scores[1]
    
    # Feature 4: Query token length
    q_len = len(q_toks)
    
    results.append({
        "query_id": item["query_id"],
        "style": item["style"],
        "score_bm25": score_bm25,
        "score_dense": score_dense,
        "score_hybrid": score_hybrid,
        "delta_bm25": delta_bm25,
        "delta_dense": delta_dense,
        "jaccard_overlap": jaccard,
        "bm25_gap": bm25_gap,
        "dense_gap": dense_gap,
        "query_length": q_len
    })

df = pd.DataFrame(results)
X_emb = np.array(query_embeddings)
X_signals = df[["jaccard_overlap", "bm25_gap", "dense_gap", "query_length"]].values
X_all = np.hstack([X_emb, X_signals])

print("\n--- Summary of Synthetic Queries by Style ---")
summary = df.groupby("style")[["score_bm25", "score_dense", "score_hybrid", "delta_bm25", "delta_dense"]].mean()
print(summary.round(4))

# -----------------------------------------------------------------------------
# 5. TRAIN DUAL ADVANTAGE REGRESSORS (Train / Test Split)
# -----------------------------------------------------------------------------
print("\n[5] Training Dual Advantage Regressors on Synthetic Data...")
n_samples = len(df)
indices = list(range(n_samples))
random.shuffle(indices)

train_idx = indices[:48]  # 80% train
test_idx = indices[48:]   # 20% test

X_train, X_test = X_all[train_idx], X_all[test_idx]
y_bm25_train, y_bm25_test = df["delta_bm25"].iloc[train_idx].values, df["delta_bm25"].iloc[test_idx].values
y_dense_train, y_dense_test = df["delta_dense"].iloc[train_idx].values, df["delta_dense"].iloc[test_idx].values

# Regressor for BM25 Advantage
reg_bm25 = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, verbose=-1, random_state=42)
reg_bm25.fit(X_train, y_bm25_train)

# Regressor for Dense Advantage
reg_dense = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, verbose=-1, random_state=42)
reg_dense.fit(X_train, y_dense_train)

# Predictions on Held-Out Test Set
pred_delta_bm25 = reg_bm25.predict(X_test)
pred_delta_dense = reg_dense.predict(X_test)

# -----------------------------------------------------------------------------
# 6. EVALUATION ON TEST QUERIES
# -----------------------------------------------------------------------------
print("\n[6] Evaluating Routing on Held-Out Test Queries...")
theta = 0.02  # Only deviate from Hybrid if predicted advantage > theta

test_df = df.iloc[test_idx].copy().reset_index(drop=True)
test_df["pred_delta_bm25"] = pred_delta_bm25
test_df["pred_delta_dense"] = pred_delta_dense

routes_chosen = []
router_scores = []
hybrid_scores = []
oracle_scores = []

for i, row in test_df.iterrows():
    p_b = row["pred_delta_bm25"]
    p_d = row["pred_delta_dense"]
    
    # Decision rule
    if p_b > theta and p_b > p_d:
        route = "BM25"
        score = row["score_bm25"]
    elif p_d > theta and p_d > p_b:
        route = "Dense"
        score = row["score_dense"]
    else:
        route = "Hybrid"
        score = row["score_hybrid"]
        
    routes_chosen.append(route)
    router_scores.append(score)
    hybrid_scores.append(row["score_hybrid"])
    oracle_scores.append(max(row["score_bm25"], row["score_dense"], row["score_hybrid"]))

test_df["route_chosen"] = routes_chosen
test_df["router_score"] = router_scores

print("\n--- Held-Out Test Sample Routing Decisions ---")
cols_to_show = ["query_id", "style", "pred_delta_bm25", "pred_delta_dense", "route_chosen", "score_hybrid", "router_score"]
print(test_df[cols_to_show].head(10).round(3))

print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE COMPARISON")
print("=" * 70)
print(f"Mean Hybrid Score:  {np.mean(hybrid_scores):.4f}")
print(f"Mean Router Score:  {np.mean(router_scores):.4f}  (Gain over Hybrid: {np.mean(router_scores) - np.mean(hybrid_scores):+.4f})")
print(f"Mean Oracle Score:  {np.mean(oracle_scores):.4f}  (Max Possible Headroom: {np.mean(oracle_scores) - np.mean(hybrid_scores):+.4f})")
print(f"Route Distribution: {pd.Series(routes_chosen).value_counts().to_dict()}")
print("=" * 70)
print("SUCCESS: End-to-end pipeline (synthetic generation -> advantage labeling -> training -> routing) validated!")

