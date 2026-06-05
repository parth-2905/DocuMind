# DocuMind 🧠

> An advanced document Q&A system with hybrid retrieval, CrossEncoder reranking, and confidence-based fallback — built for students who need grounded, cited answers from their own study material.

<img width="2554" height="1184" alt="image" src="https://github.com/user-attachments/assets/5745b03f-8250-4069-b47f-5a47727d43c7" />



🔗 **Live Demo:** https://huggingface.co/spaces/parthhhhg/DocuMind

---

## What is DocuMind?

DocuMind lets you upload any document (PDF, DOCX, PPTX, TXT, CSV) and ask questions about it. Unlike generic AI chatbots, DocuMind answers strictly from your uploaded material — with page citations — and explicitly tells you when it's drawing from general knowledge instead.

---

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, TXT, CSV with hybrid PyMuPDF + pytesseract OCR for scanned documents
- **Hybrid retrieval** — dense (FAISS + MiniLM-L6-v2) and sparse (BM25) search fused via Reciprocal Rank Fusion
- **CrossEncoder reranking** — ms-marco-MiniLM-L-6-v2 reranks top candidates for precision
- **Confidence-based fallback** — three-tier response system based on reranker confidence scores
- **Smart Notes** — Brief / Medium / Detailed note generation modes
- **Figure extraction** — embedded images and diagrams displayed alongside answers

---

## Confidence-Based Fallback

DocuMind uses CrossEncoder reranker scores to determine how to answer every query:

| Reranker Score | Source Type | Behavior |
|---|---|---|
| ≥ 3.0 | 📄 Document | Strict context-only answer with page citations |
| -1.0 to 3.0 | 📄 + 🌐 Partial | Document section + general knowledge supplement |
| < -1.0 | 🌐 Fallback | Full LLM answer with ⚠️ warning label |

<img width="2559" height="1323" alt="image" src="https://github.com/user-attachments/assets/d67efd10-6bae-469e-ac45-3dcb1b4db71f" />


<img width="2557" height="797" alt="image" src="https://github.com/user-attachments/assets/3d70d803-2422-4f6e-91df-d1e41250ec8d" />


This is what differentiates DocuMind from tools like NotebookLM — Gemini silently mixes document and general knowledge. DocuMind makes the source explicit every time.

---

## Pipeline Architecture

<img width="855" height="996" alt="image" src="https://github.com/user-attachments/assets/361fd0a4-1067-4d9c-b866-9b14aaa0bc77" />




---

## Ablation Study

Evaluated using MiniLM-L6-v2 semantic similarity scoring across 80 test cases (20 questions × 4 pipeline configurations) on an ML textbook.

| Pipeline | Faithfulness | Answer Relevancy |
|---|---|---|
| BM25 Only | 0.611 | 0.499 |
| FAISS Only | 0.671 | 0.508 |
| Hybrid RRF | 0.608 | 0.556 |
| Hybrid + Reranker | 0.654 | 0.507 |

**Key findings:**
- Hybrid RRF improved answer relevancy by **11.4%** over BM25 alone — RRF fusion surfaces more semantically relevant content
- CrossEncoder reranking improved faithfulness by **7%** over BM25 — reranked answers stay more grounded in retrieved context
- Neither single retrieval method dominates both metrics — validating the hybrid approach
- The strict grounding constraint creates a verbosity-relevancy tradeoff visible in semantic similarity scores

---

## RAG Type

DocuMind implements **Advanced Hybrid RAG**:

| RAG Type | DocuMind |
|---|---|
| Naive RAG | ❌ |
| Advanced RAG (reranking) | ✅ |
| Hybrid Retrieval (dense + sparse) | ✅ |
| Multi-format ingestion | ✅ |
| Multimodal (image OCR) | ✅ partial |
| Confidence-based fallback | ✅ |
| Agentic RAG | ❌ |

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (in-memory, stateless) |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | LLaMA 3.1 8B Instant (Groq API) |
| OCR | PyMuPDF + pytesseract |
| Backend | FastAPI |
| Frontend | React |
| Deployment | HuggingFace Spaces |

---

## Supported Formats

PDF · DOCX · PPTX · TXT · CSV

---

## Local Setup

```bash
git clone https://github.com/parth-2905/DocuMind
cd DocuMind
pip install -r requirements.txt
```

Set environment variables:
```bash
export GROQ_API_KEY=your_key_here
```

Run:
```bash
uvicorn app.main:app --reload
```

---

## Limitations

- FAISS index is stateless — document must be re-uploaded each session
- Confidence thresholds (3.0 / -1.0) are empirically tuned on tested data — may need adjustment for other domains
- LLaMA 3.1 8B via Groq free tier has token rate limits that affect response speed under load
- Partial mode triggers infrequently — most queries either clearly match or clearly miss the document


