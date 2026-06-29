<div align="center">

# 🎬 YouTube RAG Assistant

### Chat with any YouTube video. Get AI answers with exact timestamps.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-F55036?style=for-the-badge)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-206_passing-6366F1?style=for-the-badge)]()

<br/>

> **Production-grade RAG system** — paste any YouTube URL, ask anything, get grounded answers with clickable timestamp citations. Built from scratch. Deployed. 206 tests.

<br/>

### 🌐 [Live Web App](https://huggingface.co/spaces/Gaurang19/yt-rag-assistant) &nbsp;·&nbsp; 📚 [API Docs](https://rag-yt-assistant-production.up.railway.app/docs) &nbsp;·&nbsp; 🔌 [Chrome Extension](https://github.com/GaurangSane/rag-yt-assistant/releases/tag/v1.0.0)

</div>

---

## 🎯 The Problem

You find a 2-hour YouTube video. You need one specific answer. Your options:

❌ Watch the entire video &nbsp;·&nbsp; ❌ Scrub through manually &nbsp;·&nbsp; ❌ Read vague auto-chapters

**With YouTube RAG Assistant:**

```
You    →  "How does gradient descent work in this video?"

System →  "As explained at [5:15], gradient descent minimizes the loss
           function by iteratively adjusting weights. The video clarifies
           at [8:42] that the learning rate controls step size..."

           📍 [▶ 5:15 → 6:17]  [▶ 8:42 → 9:30]  ← click to jump
```

**Exact answer. Exact timestamp. Click to verify. Zero hallucination.**

---

## 🚀 Three Ways To Use It

| | Link | Description |
|---|---|---|
| 🌐 **Web App** | [HuggingFace Spaces](https://huggingface.co/spaces/Gaurang19/yt-rag-assistant) | Paste URL → chat in browser |
| 📚 **API** | [Interactive Docs](https://rag-yt-assistant-production.up.railway.app/docs) | REST API, fully documented |
| 🔌 **Chrome Extension** | [Download Free](https://github.com/GaurangSane/rag-yt-assistant/releases/tag/v1.0.0) | Chat inside YouTube |

### Chrome Extension — Install In 2 Minutes

```
1. Download ZIP from the link above → unzip it
2. Open Chrome → go to chrome://extensions
3. Toggle Developer mode ON (top right)
4. Click "Load unpacked" → select the unzipped folder
5. Open any YouTube video → click the extension icon → start chatting
```

---

## ✨ Key Features

| Feature | What it does |
|---|---|
| 🕐 **Timestamp Citations** | Every answer cites exact `[MM:SS]` with a clickable YouTube jump link |
| 🔍 **Hybrid Search** | BM25 keyword + semantic embedding + Reciprocal Rank Fusion |
| 🎯 **Score Fusion Reranking** | Replaced CrossEncoder (26s) with score fusion (0ms) — 4.5× faster |
| 🔄 **Multi-Query Generation** | Rewrites each question into 2 search variants via LLaMA3 |
| 💬 **Conversation Memory** | Follow-up questions resolve correctly using prior context |
| 🛡️ **Hallucination Guard** | Refuses to answer anything not found in the video |
| ⚡ **Smart Re-ingestion** | Videos indexed once — second question responds instantly |
| 🌐 **Cloud-Safe Transcripts** | Supadata API primary + direct fetch fallback — no IP blocking |

---

## 🏗️ Architecture

Two worlds that run at different times:

```
╔══════════════════════════════════════════════════════════════════╗
║              WORLD 1 — INGESTION (runs once per video)           ║
║                                                                  ║
║  YouTube URL → Supadata API → Transcript + Timestamps            ║
║       ↓                                                          ║
║  Chunker (60s windows, 15s overlap) → Chunk objects              ║
║       ↓                                                          ║
║  all-MiniLM-L6-v2 (batch encode) → 384-dim vectors              ║
║       ↓                                                          ║
║  ChromaDB (persistent, cosine metric) → Indexed ✅               ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║         WORLD 2 — RETRIEVAL + GENERATION (every question)        ║
║                                                                  ║
║  User Question → LLaMA3 (Groq) → 2 search query variants        ║
║       ↓                                                          ║
║  Batch embed all queries in ONE forward pass                     ║
║       ↓                                                          ║
║  Semantic (ChromaDB) + BM25 → Reciprocal Rank Fusion             ║
║       ↓                                                          ║
║  Score Fusion Reranker → Top 3 chunks selected                   ║
║       ↓                                                          ║
║  Structured prompt + hallucination guard → Groq LLaMA3           ║
║       ↓                                                          ║
║  Grounded answer with [▶ timestamp] citations ✅                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Language** | Python 3.11 | Stable, full ML ecosystem |
| **Transcript** | Supadata API + youtube-transcript-api | Cloud-proof: no IP blocking |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim) | Fast, free, runs on CPU |
| **Vector DB** | ChromaDB (persistent volume) | Zero setup, cosine search |
| **Keyword Search** | BM25Okapi (`rank-bm25`) | Exact term matching |
| **Reranker** | Score Fusion (cloud) / CrossEncoder (local) | 0ms cloud, quality local |
| **LLM** | Groq + LLaMA3-8b-instant | Free tier, fastest inference |
| **Backend** | FastAPI + uvicorn | Async, auto-docs, rate limiting |
| **UI** | Streamlit | Fast chat interface |
| **Extension** | Chrome MV3 (Side Panel) | Persistent, stays open |
| **Deployment** | Railway (API) + HuggingFace (UI) | Free tier, persistent storage |

---

## 📁 Project Structure

```
yt-rag-assistant/
│
├── 📂 src/                           # All business logic
│   ├── config.py                     # Centralized settings (@dataclass)
│   ├── api_client.py                 # HTTP client for Streamlit → FastAPI
│   │
│   ├── 📂 ingestion/
│   │   ├── transcript.py             # Supadata + direct fetch + fallback
│   │   ├── chunker.py                # Sliding window with overlap
│   │   └── embedder.py               # Singleton, batch encode
│   │
│   ├── 📂 storage/
│   │   └── vector_store.py           # ChromaDB repository pattern
│   │
│   ├── 📂 retrieval/
│   │   ├── query_transformer.py      # Multi-query via Groq
│   │   ├── retriever.py              # Hybrid search + RRF + vector cache
│   │   └── reranker.py               # ScoreFusion (cloud) / CrossEncoder
│   │
│   ├── 📂 generation/
│   │   ├── prompt_builder.py         # Structured prompt + guard
│   │   └── generator.py              # Groq call + retry logic
│   │
│   └── pipeline.py                   # 9-step orchestrator
│
├── 📂 tests/                         # 206 tests, all passing
│   ├── test_transcript.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_query_transformer.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_prompt_builder.py
│   ├── test_generator.py
│   └── test_pipeline.py
│
├── 📂 extension/                     # Chrome Extension (MV3)
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── content.js
│   └── background.js
│
├── 📂 notebooks/
│   └── 01_rag_prototype.ipynb        # Original 9-step prototype
│
├── app.py                            # Streamlit UI
├── main.py                           # FastAPI server
├── Dockerfile                        # Pre-downloads models at build time
├── railway.json                      # Railway deployment config
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start (Local)

### Prerequisites
- Python 3.11+
- Free [Groq API key](https://console.groq.com)
- Free [Supadata API key](https://supadata.ai)

### Setup

```bash
git clone https://github.com/GaurangSane/yt-rag-assistant.git
cd yt-rag-assistant

python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env:
# GROQ_API_KEY=your_groq_key
# SUPADATA_API_KEY=your_supadata_key
```

### Run FastAPI Backend

```bash
uvicorn main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

### Run Streamlit UI

```bash
streamlit run app.py
# Opens at: http://localhost:8501
```

### Run Tests

```bash
pytest tests/ -v --tb=short -m "not integration"
# 206 tests, all green
```

---

## 📡 API Reference

### POST `/ingest`
Index a YouTube video. Idempotent — safe to call multiple times.

```bash
curl -X POST https://your-app.railway.app/ingest \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://youtube.com/watch?v=aircAruvnKk"}'
```

```json
{
  "video_id"   : "aircAruvnKk",
  "chunk_count": 25,
  "was_cached" : false,
  "message"    : "Successfully indexed 25 chunks in 8.3s."
}
```

### POST `/chat`
Ask a question about an indexed video.

```bash
curl -X POST https://your-app.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://youtube.com/watch?v=aircAruvnKk",
    "question" : "what is the main topic?",
    "history"  : []
  }'
```

```json
{
  "answer"          : "As explained at [0:45], the video covers...",
  "answer_grounded" : true,
  "sources"         : [
    {
      "rank"        : 1,
      "start_time"  : "0:45",
      "display"     : "[0:45 → 1:47]",
      "youtube_link": "https://youtube.com/watch?v=aircAruvnKk&t=45s"
    }
  ],
  "queries_used" : ["main topic neural networks", "..."],
  "latency"      : {"total_ms": 14689, "retrieval_ms": 13885}
}
```

### GET `/health` · GET `/storage` · GET `/videos`

```bash
curl https://your-app.railway.app/health
# {"status": "healthy", "pipeline_loaded": true, "warmup_complete": true}

curl https://your-app.railway.app/storage
# {"videos_indexed": 3, "total_chunks": 105, "estimated_mb": 0.61}

curl https://your-app.railway.app/videos
# {"video_ids": ["aircAruvnKk", "..."], "count": 3}
```

---

## 📊 Performance

> Measured on Railway free tier CPU — real production deployment

| Metric | Before | After | Improvement |
|---|---|---|---|
| **Server startup** | 4+ minutes | 3.2 seconds | **75× faster** |
| **Response latency** | 65.5 seconds | ~15 seconds | **4.5× faster** |
| **Reranking** | 26,000ms | 0.0ms | **Eliminated** |
| **Embedding (batch)** | 38,700ms | ~14,000ms | **2.7× faster** |
| **Timeout errors** | Yes | None | **Fixed ✅** |

Every improvement came from reading actual production logs, not guessing:

```
"Batch embedding | queries=2 | time=38700ms" → found 3 sequential calls
→ Fix: encode([q1, q2]) in one forward pass → 14,000ms

"Score fusion reranking | time=0.0ms"
→ CrossEncoder (26s) replaced with weighted score fusion

"RAGPipeline ready | init_time=3215ms"
→ Dockerfile pre-bakes models → was 4+ minutes of downloading
```

---

## 🗺️ Roadmap

### ✅ Phase 1 — Notebook Prototype
- [x] 9-step RAG pipeline end-to-end in Jupyter
- [x] Transcript → chunks → embeddings → vector store → retrieval → answer

### ✅ Phase 2 — Modular Production Codebase
- [x] 10 modules, custom exceptions, logging, type hints
- [x] 206 automated tests — every module independently testable
- [x] Centralized `@dataclass` config, singleton patterns, repository pattern

### ✅ Phase 3 — Streamlit Web App
- [x] Chat UI, session state, hallucination-aware source display
- [x] API client mode — Streamlit calls FastAPI backend

### ✅ Phase 4 — FastAPI Backend
- [x] `/ingest`, `/chat`, `/health`, `/storage`, `/videos`
- [x] Rate limiting, CORS, global error handling, async execution

### ✅ Phase 5 — Chrome Extension (MV3)
- [x] Side Panel (stays open while watching)
- [x] Auto-detect video, programmatic injection, MutationObserver
- [x] Persistent history, graceful cold-start retry

### ✅ Phase 6 — Deployment
- [x] FastAPI on Railway (persistent volume, Dockerfile pre-bake)
- [x] Streamlit on HuggingFace Spaces
- [x] Supadata for cloud-proof transcripts
- [x] UptimeRobot keep-alive, GitHub Release

### 🔜 Future
- [ ] Pinecone for cloud-native vector storage
- [ ] RAGAS evaluation metrics display
- [ ] Multi-video cross-search
- [ ] Playlist support
- [ ] Chrome Web Store listing
- [ ] Chapter-aware chunking

---

## 🔍 Advanced Techniques Used

### Hybrid Search + RRF
```
Both BM25 (keyword) + semantic run for each query variant.
All result lists merged: score = Σ 1/(rank + 60) across lists.
Chunks appearing in multiple lists rank highest — naturally surfaces consensus.
```

### Batch Query Embedding
```
Before: embed(q1) → 12s, embed(q2) → 12s = 24s sequential
After:  encode([q1, q2]) = 14s total in one forward pass
```

### Score Fusion Reranking
```python
final_score = (0.6 × normalised_semantic) +
              (0.3 × normalised_rrf) +
              (0.1 × found_by_both_bonus)
# 0ms vs 26s CrossEncoder on Railway CPU
```

### Hallucination Guard
```
System prompt rule: if not in context, say so explicitly.
Pipeline detects guard phrases → sets answer_grounded=False.
UI shows "No relevant sections" instead of misleading source buttons.
```

---

## 🐛 Real Problems Solved In Production

| Error | Root Cause | Fix |
|---|---|---|
| `"YouTube is blocking requests"` | Railway datacenter IP blocked | Supadata API primary source |
| `"Read timed out after 60s"` | Sequential embedding + CrossEncoder | Batch embed + score fusion |
| `"not all arguments converted"` | Wrong Python logger syntax | f-string format |
| Health check timeout → restart loop | Warmup blocking server start | Background thread warmup |
| OOM on Railway | all-mpnet-base-v2 (768-dim) too large | all-MiniLM-L6-v2 (384-dim) |
| 4-minute cold starts | Models downloading at runtime | Dockerfile pre-bake |

---

## 📄 Environment Variables

| Variable | Where | Description |
|---|---|---|
| `GROQ_API_KEY` | Railway + local | From [console.groq.com](https://console.groq.com) |
| `SUPADATA_API_KEY` | Railway | From [supadata.ai](https://supadata.ai) |
| `CLOUD_MODE` | Railway | `true` — enables batch embed + score fusion |
| `CHROMA_DIR` | Railway | Path to persistent volume mount |
| `FASTAPI_URL` | HuggingFace | Railway backend URL for Streamlit |

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
# make changes
pytest tests/ -v --tb=short -m "not integration"
git commit -m "feat: your feature"
git push origin feature/your-feature
# open pull request
```

---

## 👨‍💻 Author

<div align="center">

**Gaurang Sane**
B.E. Computer Science (Data Science) · CGPA 8.43
University of Mumbai · AI Developer Intern @ Enjay IT Solutions

*Actively seeking AI Engineer / GenAI Engineer roles · Mumbai · Open to Remote*

[![GitHub](https://img.shields.io/badge/GitHub-GaurangSane-181717?style=flat-square&logo=github)](https://github.com/GaurangSane)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Gaurang_Sane-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/gaurang-sane-84b5b1254/)

</div>

---

<div align="center">

**Built from scratch. Every component understood before it was written.**
*206 tests · 3 live deployments · 0 tutorial copies*

⭐ **Star this repo** if it helped you understand production RAG systems

</div>
