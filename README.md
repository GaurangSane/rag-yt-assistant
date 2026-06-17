<div align="center">

# 🎬 YouTube RAG Assistant

### Ask anything about any YouTube video. Get timestamped, cited answers in seconds.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Ready-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-F55036?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Development-6366F1?style=for-the-badge)]()

<br/>

> **Production-grade RAG system** that transforms any YouTube video into an interactive knowledge base.  
> Ask questions, get grounded answers with exact timestamps — no hallucinations, no guessing.

<br/>


</div>

---

## 📌 The Problem This Solves

You find a 2-hour YouTube video on a topic you're learning. You want to know what it says about one specific concept. Your options today:

- ❌ Watch the entire video hoping to find it
- ❌ Scrub through manually — slow and imprecise  
- ❌ Read auto-generated chapters — too vague

**With YouTube RAG Assistant:**

```
You    → "How does gradient descent work in this video?"

System → "As explained at [5:15], gradient descent is an optimization 
          algorithm that minimizes the loss function by iteratively 
          adjusting weights. The video further clarifies at [8:42] 
          that the learning rate controls how large each step is..."

          📍 Sources: [5:15 → 6:17], [8:42 → 9:30]
```

Exact answer. Exact timestamps. Zero hallucination.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🕐 **Timestamp Citations** | Every answer cites the exact minute:second of the video |
| 🔍 **Hybrid Search** | Combines semantic (meaning-based) + BM25 (keyword) search |
| 🎯 **Cross-Encoder Reranking** | Deep relevance scoring picks the truly best chunks |
| 🔄 **Multi-Query Retrieval** | Generates 3 search variants per question for maximum coverage |
| 💬 **Conversation Memory** | Context-aware follow-up questions work naturally |
| 🛡️ **Hallucination Guard** | Refuses to answer outside video content — never makes things up |
| ⚡ **Smart Re-ingestion** | Already-indexed videos skip processing — instant responses |
| 🏗️ **Production Architecture** | Modular codebase, logging, tests, config management |

---

## 🏗️ Architecture

The system is divided into two worlds that run at different times:

```
╔══════════════════════════════════════════════════════════════════╗
║                    WORLD 1 — INGESTION                           ║
║                  (Runs once per video)                           ║
║                                                                  ║
║   YouTube URL                                                    ║
║       │                                                          ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  1. Transcript      │  youtube-transcript-api v0.7+          ║
║   │     Fetcher         │  → raw text + timestamps per segment   ║
║   └──────────┬──────────┘                                        ║
║              │                                                   ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  2. Chunker         │  Sliding window: 60s chunks, 15s overlap║
║   │                     │  → preserves timestamps on every chunk ║
║   └──────────┬──────────┘                                        ║
║              │                                                   ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  3. Embedding Model │  all-mpnet-base-v2 (768 dimensions)    ║
║   │                     │  → semantic vector per chunk           ║
║   └──────────┬──────────┘                                        ║
║              │                                                   ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  4. Vector Database │  ChromaDB (persistent, cosine metric)  ║
║   │                     │  → indexed for millisecond search      ║
║   └─────────────────────┘                                        ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                  WORLD 2 — RETRIEVAL + GENERATION                ║
║                  (Runs on every user question)                   ║
║                                                                  ║
║   User Question                                                  ║
║       │                                                          ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  5. Query           │  Groq LLaMA3 generates 3 variants      ║
║   │     Transformer     │  + resolves pronouns from history      ║
║   └──────────┬──────────┘                                        ║
║              │                                                   ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  6. Hybrid          │  Semantic search (ChromaDB)            ║
║   │     Retriever       │  + BM25 keyword search                 ║
║   │                     │  → fused via Reciprocal Rank Fusion    ║
║   └──────────┬──────────┘                                        ║
║              │                                                   ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  7. Reranker        │  ms-marco-MiniLM cross-encoder         ║
║   │                     │  → deep pairwise scoring, top 3 kept   ║
║   └──────────┬──────────┘                                        ║
║              │                                                   ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  8. Prompt Builder  │  Structured prompt with timestamp      ║
║   │                     │  headers + hallucination guard         ║
║   └──────────┬──────────┘                                        ║
║              │                                                   ║
║       ▼                                                          ║
║   ┌─────────────────────┐                                        ║
║   │  9. LLM Generation  │  Groq (LLaMA3-8b, temp=0.3)           ║
║   │                     │  → grounded answer with citations      ║
║   └─────────────────────┘                                        ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Language** | Python 3.11 | Stable, broad ML ecosystem support |
| **Transcript** | `youtube-transcript-api` v0.7+ | Free, timestamped, no API key needed |
| **Embeddings** | `all-mpnet-base-v2` | Best free local model, 768-dim vectors |
| **Vector DB** | ChromaDB (persistent) | Zero setup, cosine similarity, local |
| **Keyword Search** | `rank-bm25` (BM25Okapi) | Exact term matching complements semantic |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Cross-encoder, free, purpose-built |
| **LLM** | Groq + LLaMA3-8b | Free tier, fastest inference (LPU hardware) |
| **Backend** | FastAPI *(coming)* | Async, auto-docs, production-grade |
| **UI** | Streamlit *(coming)* | Fast to build, clean chat interface |
| **Deployment** | Railway + HuggingFace *(coming)* | Free tier, live URL for portfolio |

---

## 📁 Project Structure

```
yt-rag-assistant/
│
├── 📂 src/                          
│   ├── config.py                    
│   │
│   ├── 📂 ingestion/                
│   │   ├── transcript.py            
│   │   ├── chunker.py               
│   │   └── embedder.py              
│   │
│   ├── 📂 storage/                  
│   │   └── vector_store.py          
│   │
│   ├── 📂 retrieval/                
│   │   ├── query_transformer.py     
│   │   ├── retriever.py             
│   │   └── reranker.py              
│   │
│   ├── 📂 generation/               
│   │   ├── prompt_builder.py        
│   │   └── generator.py             
│   │
│   └── pipeline.py                  
│
├── 📂 tests/                        
│   ├── test_transcript.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_pipeline.py
│
├── 📂 notebooks/
│   └── 01_rag_prototype.ipynb       
│
├── 📂 data/                         
├── 📂 chroma_db/                    
├── 📂 logs/                         
│
├── app.py                           
├── main.py                          
├── .env                             
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)
- Git

### 1. Clone and Setup

```bash
git clone https://github.com/GaurangSane/yt-rag-assistant.git
cd yt-rag-assistant

python3.11 -m venv venv
source venv/bin/activate        

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env

echo "GROQ_API_KEY=your_key_here" >> .env
```

### 3. Validate Setup

```bash
python -c "from src.config import validate_environment; validate_environment()"
```

### 4. Run the Pipeline

```python
from src.pipeline import rag_pipeline

result = rag_pipeline(
    youtube_url   = "https://www.youtube.com/watch?v=your_video_id",
    user_question = "What is the main concept explained in this video?"
)

print(result["answer"])

for source in result["sources"]:
    print(f"  📍 [{source['timestamp']} → {source['end_time']}]")
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v

pytest tests/test_transcript.py -v
pytest tests/test_chunker.py -v

pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔬 Advanced RAG Techniques Implemented

This project goes well beyond a basic RAG tutorial. Here is what makes it production-grade:

### 1. 🕐 Timestamp-Aware Chunking
Every chunk carries its exact video position. Answers cite `[4:32]` — not vague references but clickable, precise timestamps. No other basic RAG demo does this correctly.

### 2. 🔍 Hybrid Search (Semantic + BM25)
```
Semantic search alone:  catches meaning, misses exact technical terms
BM25 alone:             catches exact terms, misses conceptual synonyms
Both combined:          catches everything ✅
```

### 3. 🎲 Reciprocal Rank Fusion (RRF)
Six search lists (3 queries × 2 methods) merged by the formula `score = Σ 1/(rank + 60)`. Chunks appearing consistently across multiple searches score highest.

### 4. 🎯 Cross-Encoder Reranking
```
Bi-encoder (retrieval): encodes question and chunk separately → fast, shallow
Cross-encoder (reranking): reads [question + chunk] jointly → slow, deep

Two-stage pipeline: retrieve many cheaply → rerank few accurately
```

### 5. 🔄 Multi-Query Generation
One question → three differently-phrased search queries → wider retrieval net → better coverage.

### 6. 💬 Conversation Memory
Last 3 conversation turns included in context. Vague follow-ups like *"explain that more simply"* resolve correctly because prior context travels through the entire pipeline.

### 7. 🛡️ Hallucination Guard
Explicit system instruction: *if the answer is not in the provided video segments, say so — never answer from general knowledge.* Test 3 in the pipeline verifies this works on every run.

---

## 📊 Performance Benchmarks

> Tested on an 18-minute YouTube video (25 chunks after ingestion)

| Operation | Time | Notes |
|---|---|---|
| Transcript fetch | ~1.5s | Network dependent |
| Chunking (25 chunks) | ~0.01s | Pure Python, instant |
| Embedding (25 chunks) | ~5s | Local CPU, one-time cost |
| Vector DB storage | ~0.2s | ChromaDB write |
| **Total ingestion** | **~7s** | **One time per video** |
| Query transformation | ~0.8s | Groq API call |
| Hybrid retrieval | ~0.1s | ChromaDB + BM25 |
| Reranking (5→3 chunks) | ~0.3s | Local cross-encoder |
| LLM generation | ~1.2s | Groq LPU, very fast |
| **Total per question** | **~2.4s** | **Every question** |

---

## 🗺️ Roadmap

### ✅ Phase 1 — Notebook Prototype (Complete)
- [x] Step 1: YouTube transcript fetcher with timestamps
- [x] Step 2: Timestamp-based chunker with sliding window
- [x] Step 3: Embedding with `all-mpnet-base-v2`
- [x] Step 4: ChromaDB persistent vector store
- [x] Step 5: Multi-query transformer (Groq + LLaMA3)
- [x] Step 6: Hybrid retriever (Semantic + BM25 + RRF)
- [x] Step 7: Cross-encoder reranker
- [x] Step 8: Structured prompt builder with hallucination guard
- [x] Step 9: End-to-end pipeline orchestration with conversation memory

### 🔄 Phase 2 — Modular Production Codebase (In Progress)
- [x] Project scaffold and folder structure
- [x] Centralized config with `@dataclass` settings
- [x] `src/ingestion/transcript.py` — custom exceptions, logging, dataclasses
- [ ] `src/ingestion/chunker.py`
- [ ] `src/ingestion/embedder.py`
- [ ] `src/storage/vector_store.py`
- [ ] `src/retrieval/query_transformer.py`
- [ ] `src/retrieval/retriever.py`
- [ ] `src/retrieval/reranker.py`
- [ ] `src/generation/prompt_builder.py`
- [ ] `src/generation/generator.py`
- [ ] `src/pipeline.py` — clean orchestrator
- [ ] Unit tests for every module

### ⬜ Phase 3 — Streamlit Web App
- [ ] PDF/YouTube URL upload interface
- [ ] Chat UI with message history
- [ ] Timestamp source display with video links
- [ ] Session state management

### ⬜ Phase 4 — FastAPI Backend
- [ ] `POST /ingest` — accepts YouTube URL, runs ingestion pipeline
- [ ] `POST /chat` — accepts question + history, returns answer + sources
- [ ] `GET /health` — health check endpoint
- [ ] Pydantic request/response models
- [ ] Async endpoints

### ⬜ Phase 5 — Chrome Extension
- [ ] Detects YouTube video URL automatically
- [ ] Popup chat interface on any YouTube page
- [ ] Auto-ingests video on page load
- [ ] Connects to FastAPI backend

### ⬜ Phase 6 — Deployment
- [ ] FastAPI backend on Railway (free tier)
- [ ] Streamlit UI on HuggingFace Spaces (free tier)
- [ ] Chrome extension published (or GitHub + demo video)
- [ ] Live URLs in portfolio

### ⬜ Production Upgrades (Future)
- [ ] Swap ChromaDB → Pinecone for cloud-native vector storage
- [ ] Swap local reranker → Cohere Rerank API
- [ ] Swap local embeddings → OpenAI `text-embedding-3-small`
- [ ] Add Redis caching for repeated questions on same video
- [ ] Support multiple videos in one session
- [ ] Evaluation framework with Ragas metrics

---

## 🧠 What I Learned Building This

This project was built as a deep learning exercise — understanding every component before writing a single line. Key concepts internalized:

**RAG Architecture**
- The two-world separation (ingestion vs retrieval) and why it matters for performance
- Why vector DBs exist and what SQL databases fundamentally cannot do

**Retrieval Engineering**
- Bi-encoder vs cross-encoder: different architectures for different stages
- Why hybrid search (semantic + keyword) outperforms either alone
- Reciprocal Rank Fusion mathematics and why `k=60` is the standard constant

**Production Engineering**
- Centralized configuration with `@dataclass` — no magic numbers scattered in code
- Custom exceptions for clean module boundaries and decoupling
- Logging over `print()` — timestamps, severity levels, file output
- Unit testing in isolation — testing one thing without touching other modules
- Type hints as documentation and IDE tooling

**LLM Engineering**
- Prompt structure: system rules → context → question (order matters)
- Temperature selection: 0.7 for variety (query generation), 0.3 for precision (answers)
- Hallucination prevention through explicit system instructions
- Token budget awareness — prompts + context + answer must all fit

---

## 📄 API Reference

### `rag_pipeline()`

```python
from src.pipeline import rag_pipeline

result = rag_pipeline(
    youtube_url          = "https://youtube.com/watch?v=...",  
    user_question        = "Your question here",               
    conversation_history = [                                   
        {"question": "previous q", "answer": "previous a"}
    ],
    window_sec           = 60,    
    overlap_sec          = 15,    
    retrieve_top_k       = 5,     
    rerank_top_k         = 3,     
)
```

**Returns:**

```python
{
    "answer"      : "As explained at [5:15], ...",
    "sources"     : [
        {"timestamp": "5:15", "end_time": "6:17", "start_sec": 315.0},
        {"timestamp": "8:42", "end_time": "9:30", "start_sec": 522.0},
    ],
    "queries_used": [
        "gradient descent optimization neural networks",
        "how model weights update during training",
        "loss function minimization backpropagation"
    ]
}
```

---

## 🤝 Contributing

This is a learning project built in public. Feedback, suggestions, and PRs are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "feat: add your feature"`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Gaurang Sane**  
Final Year CS (Data Science) Student — University of Mumbai  
AI Developer Intern @ Enjay IT Solutions

[![GitHub](https://img.shields.io/badge/GitHub-GaurangSane-181717?style=flat&logo=github)](https://github.com/GaurangSane)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/gaurang-sane-84b5b1254/))

---

---

<div align="center">

**Built from scratch, step by step, understanding every component.**  
*Not a tutorial copy. Every design decision documented.*

⭐ Star this repo if it helped you understand RAG systems

</div>
