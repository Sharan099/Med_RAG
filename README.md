# 🏥 MedRAG — Medical AI Chatbot (Local, Free, $0/query)

> A production-grade Retrieval-Augmented Generation (RAG) chatbot for medical Q&A, built entirely on open-source models. No API keys. No cloud costs. Runs on your own machine.

**Live Demo:** [your-username-medrag.hf.space](https://huggingface.co/spaces/YOUR_USERNAME/medrag-chatbot)

---

## What Makes This Different

Most RAG tutorials use OpenAI + Pinecone and cost money per query. MedRAG uses:

| Component | Choice | Why |
|---|---|---|
| LLM | gemma2:2b via Ollama | Free, runs locally, no API key |
| Embeddings | BGE-base-en-v1.5 | State-of-art retrieval, 768 dims |
| Vector DB | ChromaDB | Persistent, no server needed |
| Retrieval | Hybrid BM25 + Semantic + RRF | Better than semantic-only |
| Reranking | Cross-encoder MiniLM | Filters noise from retrieval |
| Prompting | DSPy ChainOfThought | Self-optimizing prompts |
| Pipeline | FastAPI + Async + Queue | Production-ready, handles concurrency |
| Cost | **$0/query** | vs $0.005/query with OpenAI |

---

## Architecture

```
User Question
     │
     ▼
Query Embedding (BGE-base)
     │
     ├──► FAISS Semantic Search (top-10)
     │                                    ──► RRF Fusion
     └──► BM25 Keyword Search  (top-10)
                    │
                    ▼
          Cross-Encoder Reranker (top-5 → top-3)
                    │
                    ▼
          DSPy ChainOfThought Prompt
                    │
                    ▼
          gemma2:2b via Ollama
                    │
                    ▼
          Answer + Sources + Confidence
```

---

## Performance

| Metric | Score |
|---|---|
| Keyword Coverage | ~69.5% |
| Grounding Score | ~60.0% |
| IDK Accuracy | 80.0% |
| Avg Chunk Score | 0.851 |
| Avg Latency | ~13-16s (CPU) |
| Cost per query | **$0** |

---

## Dataset

- **Source:** MedQuAD (Medical Question Answering Dataset)
- **Size:** 16,407 Q&A pairs → 75,160 chunks → ~55,000 after deduplication
- **Domains:** 8 medical collections (diabetes, cardiology, cancer, kidney, etc.)

---

## Quick Start — Docker (Recommended)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/medrag-chatbot.git
cd medrag-chatbot

# 2. Build image (downloads models inside, ~10 min first time)
docker build -t medrag:latest .

# 3. Run
docker run -p 8001:8001 medrag:latest

# 4. Open browser
open http://localhost:8001
```

---

## Quick Start — Local Python

```bash
# 1. Install Ollama: https://ollama.com
ollama pull gemma2:2b

# 2. Clone and install
git clone https://github.com/YOUR_USERNAME/medrag-chatbot.git
cd medrag-chatbot
pip install -r requirements.txt

# 3. Build the index (run ONCE — takes ~20 mins)
python phase1_indexing.py

# 4. Start server
uvicorn backend_api:app --port 8001

# 5. Open browser
open http://localhost:8001
```

---

## Project Structure

```
medrag-chatbot/
├── backend_api.py          # FastAPI backend — async pipeline + DSPy
├── frontend.html           # Chat UI — streaming, confidence badges
├── phase1_indexing.py      # Index builder — chunking + embedding
├── evaluate_rag.py         # Evaluation — 6 metrics dashboard
├── optimize_prompts.py     # DSPy prompt optimization
├── Dockerfile              # Docker deployment
├── start.sh                # Container startup script
├── requirements.txt
└── README.md
```

---

## Key Technical Concepts

**Hybrid Retrieval + RRF:** Combines BM25 (keyword) and semantic similarity scores using Reciprocal Rank Fusion. Better than either alone — BM25 catches exact medical terms like "HIC", semantic catches conceptual matches.

**DSPy Prompting:** Instead of hand-crafted f-strings, DSPy declares input/output signatures and optimizes prompts using few-shot examples. ChainOfThought adds reasoning steps before the final answer.

**8-Domain Partitioning:** Chunks are classified into domain-specific ChromaDB collections (diabetes, cardiology, cancer, etc.). Queries auto-route to the relevant domain, reducing noise.

**4-Layer Confidence Gate:** Answer is only shown if chunk score ≥ 0.55. Below threshold → honest IDK response with source suggestions. Prevents hallucination on out-of-domain questions.

**Async Queue:** Request queue prevents multiple concurrent LLM calls from exhausting RAM. Background worker processes one at a time, HTTP responses wait via asyncio Future.

---

## Evaluation Dashboard

Running `python evaluate_rag.py` generates a full 11-panel dashboard:

- Precision@K, Recall@K, MRR
- Context Relevance, Answer Relevance, Faithfulness
- IDK Accuracy, Latency distribution
- Radar chart of all metrics

---

## Requirements

```
Python 3.11+
8GB RAM minimum (for gemma2:2b)
Ollama installed locally
```

---

## License

MIT License — free to use, modify, deploy.

---

## Author

Built by [Your Name] — Aspiring AI Engineer

- GitHub: [github.com/YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [linkedin.com/in/YOUR_PROFILE](https://linkedin.com/in/YOUR_PROFILE)
- HuggingFace: [huggingface.co/YOUR_USERNAME](https://huggingface.co/YOUR_USERNAME)
