# 📄 Ask My Paper

> **🚧 Under Active Construction — Core pipeline functional, multimodal & evaluation layers in progress.**

A multimodal academic RAG system built for researchers and students who need to **actually understand** their papers — not just keyword-search them. Ask My Paper handles text, tables, equations, and embedded images inside PDFs, runs hybrid retrieval with reranking, and is evaluated end-to-end with RAGAS metrics.

This isn't a generic PDF Q&A wrapper. It's built from scratch as a targeted academic study assistant.

---

## ✨ What It Does

- Upload any academic PDF and ask questions in natural language
- Extracts and understands **text, tables, equations, and figures** from PDFs
- Retrieves relevant context using **hybrid BM25 + dense vector search** fused with RRF
- Reranks candidates with **FlashRank** before generation
- Generates grounded answers via **GPT-4o / GPT-4V** (no hallucination shortcuts)
- Evaluates pipeline quality with **RAGAS** (faithfulness, answer relevancy)
- Logs all experiments to **Weights & Biases**

---

## 🏗️ Architecture

```
PDF Input
   │
   ├── Text Extraction      → PyMuPDF
   ├── Table Extraction     → pdfplumber
   ├── Image Captioning     → GPT-4V (via OpenAI SDK)
   └── OCR (scanned PDFs)  → Tesseract + pytesseract
          │
          ▼
   Chunking & Embedding
   (sentence-transformers)
          │
          ▼
   ┌──────────────────────────────┐
   │        Qdrant Vector DB      │
   │  Dense search + BM25 sparse  │
   └──────────────────────────────┘
          │
          ▼
   RRF Fusion → FlashRank Reranker
          │
          ▼
   GPT-4o Generation
          │
          ▼
   RAGAS Evaluation + W&B Logging
```

---

## 🛠️ Tech Stack

### Core RAG Pipeline
| Component | Technology |
|---|---|
| PDF Text Extraction | `PyMuPDF` (fitz) |
| Table Extraction | `pdfplumber` |
| Image Captioning | `GPT-4V` via OpenAI SDK |
| OCR (scanned PDFs) | `pytesseract` + Tesseract binary |
| Embeddings | `sentence-transformers` |
| Vector Store | `Qdrant` |
| Sparse Retrieval | `BM25` |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `FlashRank` |
| LLM | `GPT-4o` / `GPT-4V` (OpenAI SDK — no LangChain) |

### Evaluation & Observability
| Component | Technology |
|---|---|
| RAG Evaluation | `RAGAS` (faithfulness, answer relevancy) |
| Experiment Tracking | `Weights & Biases` |

### Infrastructure & UI
| Component | Technology |
|---|---|
| API Backend | `FastAPI` + `Uvicorn` |
| Frontend | `Streamlit` |
| Image Processing | `Pillow` |
| ML Utilities | `scikit-learn`, `numpy`, `pandas` |
| Deep Learning | `PyTorch`, `transformers` |

---

## 📁 Project Structure

```
Ask-My-Paper/
├── src/                    # Core source modules
├── notebooks/              # Experiment & prototyping notebooks
├── rag.py                  # Main RAG pipeline entrypoint
├── requirements.txt        # Full dependency list
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Tesseract OCR binary installed on your system
- OpenAI API key
- Qdrant instance (local or cloud)

### Installation

```bash
git clone https://github.com/Harsh-Prajapati54/Ask-My-Paper.git
cd Ask-My-Paper
pip install -r requirements.txt
```

Install Tesseract (for scanned PDF support):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Environment Variables

```env
OPENAI_API_KEY=your_openai_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
WANDB_API_KEY=your_wandb_key
```

### Run

```bash
# Run the pipeline
python rag.py

# Or launch the Streamlit UI
streamlit run app.py
```

---

## 📊 Evaluation Results (Current)

| Metric | Score |
|---|---|
| Faithfulness | 0.89 |
| Answer Relevancy | 0.91 |

> Evaluated on a curated set of academic paper QA pairs using RAGAS framework.

---

## 🗺️ Roadmap

- [x] Hybrid BM25 + dense retrieval with RRF fusion
- [x] FlashRank reranking
- [x] Multimodal PDF parsing (text, tables, images, equations)
- [x] GPT-4V image captioning
- [x] RAGAS evaluation pipeline
- [x] W&B experiment logging
- [ ] CRAG-style quality gate (corrective retrieval)
- [ ] Failure analysis on low-scoring RAGAS samples
- [ ] Streamlit UI polish
- [ ] Docker support
- [ ] Citation tracing (answer → exact PDF chunk + page)

---

## 🤝 Contributing

This project is actively under construction. Issues and PRs are welcome once the core pipeline stabilizes.

---

## 📬 Contact

**Harsh Prajapati** — Computer Engineering (AI/ML), UVPCE Ganpat University  
[GitHub](https://github.com/Harsh-Prajapati54) · [LinkedIn](https://linkedin.com/in/harsh-prajapati54)

---

*Built without LangChain abstractions — direct OpenAI SDK, PyMuPDF, and Qdrant client for full control over the pipeline.*