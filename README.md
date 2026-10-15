# RAG_Q-A
```markdown
# Production-Grade Hybrid RAG Pipeline for QA on SQuAD 1.1

This repository implements an end-to-end, reproducible, and highly modular **Retrieval-Augmented Generation (RAG)** pipeline designed for answering questions over the Stanford Question Answering Dataset (SQuAD 1.1). 

It features a state-of-the-art hybrid search (sparse BM25 + dense vector retrieval), rank fusion, and cross-encoder reranking, backed by an efficient on-disk vector index and a local LLM generator.

---

## 🚀 Pipeline Architecture

```
                    [ SQuAD Contexts ]
                            │
                    [ Data Cleaning ]
                            │
              [ Duplicate Context Removal ]
                            │
                    [ LangChain Docs ]
                            │
               [ Chunking & Size Tuning ]
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
         ┌───────────┐             ┌───────────┐
         │   BM25    │             │ ChromaDB  │
         │  Sparse   │             │   Dense   │
         └─────┬─────┘             └─────┬─────┘
               └────────────┬────────────┘
                            ▼
               [ Reciprocal Rank Fusion ]
                            │
                   [ Candidate Top-50 ]
                            │
                 [ Cross-Encoder Rerank ]
                            │
                        [ Top-3 ]
                            │
                [ TinyLlama 1.1B Chat ]
                            │
                        [ Answer ]
```

---

## 🛠️ Stack & Technologies Used

- **Orchestration & Text Processing:** `LangChain`, `RecursiveCharacterTextSplitter`
- **Sparse Retrieval:** `rank-bm25` (BM25Okapi)
- **Dense Retrieval:** `ChromaDB` via `langchain-chroma` using `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generator:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (using PyTorch FP16/FP32)
- **Evaluation Datasets:** SQuAD 1.1 (via Hugging Face `datasets`)
- **Evaluation Metrics:** Token-level Precision/Recall/F1, Exact Match (EM), BERTScore, Retrieval Recall@K, and Mean Reciprocal Rank (MRR).

---

## 📐 Key Design Features

1. **Data Leakage Prevention:** Conceptually separates the retrieval corpus and evaluation questions. A small 100-sample training subset is used strictly for parameter tuning (selecting chunk size & overlap), ensuring final validation scores represent true generalization.
2. **Hybrid Search & Fusion:** Merges keyword matches (BM25) and semantic vector space matches (ChromaDB) using **Reciprocal Rank Fusion (RRF)** to construct a candidate set of the top 50 chunks.
3. **Cross-Encoder Reranking:** Filters down the candidate set to the most relevant top 3 contexts to avoid overwhelming the LLM context window.
4. **Reproducibility:** Constrained with seed matching (`SEED = 42`) across Python, NumPy, and PyTorch.
5. **Persistence:** Features programmatic export and reloading tests for the ChromaDB database index directory.
```
