"""
MedRAG — Production Backend v8.0
==================================
New in v8.0:
  - DSPy optimized prompting (replaces hand-crafted prompts)
  - Async pipeline (non-blocking I/O throughout)
  - Request queue (handles concurrent users without crashing)
  - Background worker (processes queue independently)
  - Async BM25 + ChromaDB search
  - Connection pooling for ChromaDB

Tech Stack Concepts:
  DSPy    = Declarative Self-improving Python — optimizes prompts automatically
  Async   = Non-blocking I/O — handles many requests without waiting
  Queue   = FIFO buffer — smooths traffic spikes, prevents overload
  Worker  = Background coroutine — drains queue independently of API

Run: uvicorn backend_api:app --port 8001 --reload-dir H:\Rag_chatbot
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import chromadb, ollama, time, json, re, csv, os, hashlib
import asyncio
import numpy  as np
import pandas as pd
from rank_bm25 import BM25Okapi
from datetime import datetime
from typing import Optional
import uuid as uuid_lib

# DSPy imports
try:
    import dspy
    USE_DSPY = True
except ImportError:
    USE_DSPY = False
    print("DSPy not installed. Run: pip install dspy-ai")
    print("Falling back to template prompts.")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH   = r"H:\Rag_chatbot\chroma_db"
COLLECTION_NAME  = "medical_qa"
BM25_CORPUS_PATH = r"H:\Rag_chatbot\chunks_bm25.csv"
EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL        = "llama3.2:1b "

TOP_K_RETRIEVE   = 10
TOP_K_RERANK     = 5
TOP_K_FINAL      = 3
MIN_SCORE        = 0.45
CONFIDENCE_GATE  = 0.55
LLM_TEMPERATURE  = 0.0
LLM_MAX_TOKENS   = 150
REPEAT_PENALTY   = 1.3
NUM_CTX          = 1024
NUM_THREAD       = 4
NUM_KEEP         = 0
BM25_WEIGHT      = 0.30
SEMANTIC_WEIGHT  = 0.70
LOG_FILE         = r"H:\Rag_chatbot\rag_eval_log.csv"

# Queue config
MAX_QUEUE_SIZE   = 50    # max pending requests before rejecting new ones
QUEUE_TIMEOUT    = 120   # seconds a request waits in queue before timeout

# Cache config
EXACT_CACHE_SIZE    = 500
SEMANTIC_CACHE_SIZE = 200
SEMANTIC_THRESHOLD  = 0.95

STOPWORDS = {
    "what","is","are","how","does","the","a","an","in","of","for",
    "to","do","can","i","me","my","it","its","was","be","and","or",
    "with","on","at","by","this","that","they","them","their","we"
}

DOMAIN_MAP = {
    "diabetes":    ["diabetes","insulin","glucose","blood sugar","diabetic"],
    "heart":       ["heart","cardiac","cholesterol","hypertension","blood pressure"],
    "respiratory": ["asthma","lung","breathing","pneumonia","bronch","inhaler"],
    "kidney":      ["kidney","renal","urine","dialysis","nephro"],
    "cancer":      ["cancer","tumor","tumour","chemotherapy","oncology"],
    "infection":   ["infection","bacteria","virus","antibiotic","fever","flu"],
    "neurology":   ["brain","neuron","alzheimer","parkinson","seizure","migraine"],
    "mental":      ["depression","anxiety","mental","psychiatric","therapy"],
}

MEDICAL_TERMS = {
    "urination":    ["urinate","urinating","urine output"],
    "glucose":      ["blood sugar","sugar level"],
    "insulin":      ["hormone from pancreas","pancreatic hormone"],
    "hypertension": ["high blood pressure","elevated blood pressure"],
    "palpitations": ["rapid heartbeat","fast heartbeat"],
    "fatigue":      ["tiredness","feeling tired","weakness"],
    "dyspnea":      ["shortness of breath","difficulty breathing"],
}

LLM_OPTIONS = {
    "temperature":    LLM_TEMPERATURE,
    "num_predict":    LLM_MAX_TOKENS,
    "repeat_penalty": REPEAT_PENALTY,
    "top_p":          0.9,
    "num_ctx":        NUM_CTX,
    "num_thread":     NUM_THREAD,
    "num_keep":       NUM_KEEP,
}

# ── DSPy SETUP ─────────────────────────────────────────────────────────────────
"""
DSPy CONCEPT:
  Instead of writing prompts like strings, DSPy lets you define
  what you WANT (input/output signatures) and learns HOW to prompt.

  Traditional: "You are a medical assistant. Answer using ONLY these facts..."
  DSPy:        Define MedicalQA(context, question -> answer) and let DSPy
               optimize the prompt through examples automatically.

  Key classes:
    dspy.Signature  = defines input/output fields
    dspy.ChainOfThought = reasoning module (adds "Let's think step by step")
    dspy.Predict    = direct prediction module
    dspy.compile()  = optimizes the prompt using few-shot examples
"""

if USE_DSPY:
    # Configure DSPy to use Ollama as the LLM backend
    # DSPy wraps Ollama and manages prompt optimization
    try:
        ollama_lm = dspy.LM(
        model       = f"ollama/{LLM_MODEL}",
        api_base    = "http://localhost:11434",
        max_tokens  = LLM_MAX_TOKENS,
        temperature = LLM_TEMPERATURE,
)
        dspy.settings.configure(lm=ollama_lm)
        print("DSPy configured with Ollama backend ✓")
    except Exception as e:
        USE_DSPY = False
        print(f"DSPy setup failed: {e} — using template prompts")

    if USE_DSPY:
        # ── DSPy Signature: defines the contract for the LLM call ──────────────
        # Input:  context (retrieved chunks), question (user query)
        # Output: answer (grounded medical response)
        # DSPy reads these docstrings and field descriptions to build the prompt
        class MedicalQASignature(dspy.Signature):
            """
            You are a careful medical assistant.
            Answer ONLY using the provided context.
            Be concise (2-3 sentences). Use exact medical terms.
            If context is insufficient, say: I don't have enough information.
            """
            context  = dspy.InputField(desc="Retrieved medical knowledge chunks numbered [1][2][3]")
            question = dspy.InputField(desc="The medical question asked by the user")
            answer   = dspy.OutputField(desc="Concise grounded answer citing [1][2][3]")

        # ── DSPy IDK Signature: for when confidence is low ─────────────────────
        class IDKSignature(dspy.Signature):
            """Explain why you cannot answer a medical question."""
            reason   = dspy.InputField(desc="Why the question cannot be answered")
            question = dspy.InputField(desc="The original question")
            response = dspy.OutputField(desc="Polite explanation with suggestion for help")

        # ── DSPy Modules: wrap signatures in reasoning strategies ───────────────
        # ChainOfThought adds "Let's think step by step" reasoning
        # This produces more accurate answers than direct Predict
        class MedicalRAGModule(dspy.Module):
            def __init__(self):
                super().__init__()
                # ChainOfThought: adds reasoning steps before final answer
                self.qa  = dspy.ChainOfThought(MedicalQASignature)
                self.idk = dspy.Predict(IDKSignature)

            def forward(self, context: str, question: str) -> dspy.Prediction:
                return self.qa(context=context, question=question)

            def explain_idk(self, reason: str, question: str) -> str:
                return self.idk(reason=reason, question=question).response

        # Instantiate the module
        dspy_module = MedicalRAGModule()
        print("DSPy MedicalRAGModule ready ✓")

# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MedRAG API", version="8.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── ASYNC QUEUE ────────────────────────────────────────────────────────────────
"""
QUEUE CONCEPT:
  A queue is a First-In-First-Out (FIFO) buffer.
  Without a queue:  100 users hit the server → 100 LLM calls at once → crash
  With a queue:     100 users hit the server → 100 requests enter queue
                    → worker processes ONE at a time → no crash

  asyncio.Queue: async-safe queue built into Python
    await queue.put(item)  = add to queue (non-blocking)
    await queue.get()      = remove from queue (waits if empty)
    queue.task_done()      = signal that item was processed

  QueueItem: wraps each request with a Future so the HTTP response
             waits for the result to come back from the worker.
"""
request_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

class QueueItem:
    """Wraps a request + a Future that resolves when the answer is ready."""
    def __init__(self, question: str, top_k: int, request_id: str):
        self.question   = question
        self.top_k      = top_k
        self.request_id = request_id
        self.future: asyncio.Future = None  # set at queue time

# ── STARTUP ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MedRAG v8.0 — DSPy + Async + Queue Pipeline")
print("=" * 60)

print("[1/5] Loading ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection    = chroma_client.get_collection(COLLECTION_NAME)
print(f"      {COLLECTION_NAME}: {collection.count():,} chunks")

domain_collections: dict = {}
for domain in list(DOMAIN_MAP.keys()) + ["general"]:
    try:
        col = chroma_client.get_collection(f"medical_{domain}")
        domain_collections[domain] = col
    except Exception:
        pass
print(f"      Domain collections: {len(domain_collections)}")

print(f"[2/5] Loading {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)
test_vec  = embedder.encode("test", normalize_embeddings=True)
db_peek   = collection.peek(limit=1)
if db_peek["embeddings"] is not None and len(db_peek["embeddings"]) > 0:
    db_dim, model_dim = len(db_peek["embeddings"][0]), len(test_vec)
    if db_dim != model_dim:
        raise ValueError(f"DIMENSION MISMATCH: DB={db_dim} Model={model_dim}")
    print(f"      Dimension: {model_dim} dims ✓")

print(f"[3/5] Loading reranker...")
try:
    reranker     = CrossEncoder(RERANKER_MODEL)
    USE_RERANKER = True
    print("      Reranker loaded ✓")
except Exception as e:
    reranker     = None
    USE_RERANKER = False
    print(f"      Reranker skipped: {e}")

print("[4/5] Building BM25 index...")
try:
    bm25_df     = pd.read_csv(BM25_CORPUS_PATH)
    bm25_texts  = bm25_df["text"].tolist()
    bm25_corpus = [t.lower().split() for t in bm25_texts]
    bm25_index  = BM25Okapi(bm25_corpus)
    USE_BM25    = True
    print(f"      BM25: {len(bm25_texts):,} docs ✓")
except Exception as e:
    bm25_index  = None
    bm25_texts  = []
    USE_BM25    = False
    print(f"      BM25 skipped: {e}")

print("[5/5] Initialising caches and log...")
_exact_cache:    dict = {}
_semantic_cache: list = []

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp","question","top_chunk_score","chunks_used",
            "answered","cache_hit","dspy_used","embed_ms",
            "retrieve_ms","rerank_ms","llm_ms","total_ms","answer_preview"
        ])

print(f"      LLM warmup...")
try:
    ollama.chat(model=LLM_MODEL,
                messages=[{"role":"user","content":"hi"}],
                options={"num_predict":1,"num_ctx":256})
    print("      LLM warm ✓")
except Exception as e:
    print(f"      Warmup failed: {e}")

print("=" * 60)
print(f"  DSPy     : {'✓ active' if USE_DSPY else '✗ fallback template'}")
print(f"  BM25     : {'✓' if USE_BM25 else '✗'}")
print(f"  Reranker : {'✓' if USE_RERANKER else '✗'}")
print(f"  Queue    : max={MAX_QUEUE_SIZE} timeout={QUEUE_TIMEOUT}s")
print("  Ready    — http://localhost:8001")
print("=" * 60)

# ── SCHEMAS ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k:    int = TOP_K_FINAL

class SourceChunk(BaseModel):
    ref:         int
    text:        str
    score:       float
    label:       str
    domain:      str = "general"
    parent_text: str = ""

class ChatResponse(BaseModel):
    question:   str
    answer:     str
    sources:    list[SourceChunk]
    answered:   bool
    confidence: str
    latency_ms: int
    cache_hit:  bool = False
    dspy_used:  bool = False
    queue_wait_ms: int = 0

# ── CACHE ──────────────────────────────────────────────────────────────────────
def _cache_key(question: str) -> str:
    normalized = re.sub(r"[^\w\s]", "", question.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()

def get_exact_cached(question: str) -> Optional[ChatResponse]:
    return _exact_cache.get(_cache_key(question))

def set_exact_cache(question: str, response: ChatResponse):
    if len(_exact_cache) >= EXACT_CACHE_SIZE:
        del _exact_cache[next(iter(_exact_cache))]
    _exact_cache[_cache_key(question)] = response

def get_semantic_cached(q_vector: list) -> Optional[ChatResponse]:
    if not _semantic_cache:
        return None
    cached_vecs = np.array([c["vec"] for c in _semantic_cache])
    scores      = cos_sim([q_vector], cached_vecs)[0]
    best_idx    = int(np.argmax(scores))
    if scores[best_idx] >= SEMANTIC_THRESHOLD:
        return _semantic_cache[best_idx]["response"]
    return None

def set_semantic_cache(question: str, q_vector: list, response: ChatResponse):
    _semantic_cache.append({"vec": q_vector, "response": response, "question": question})
    if len(_semantic_cache) > SEMANTIC_CACHE_SIZE:
        _semantic_cache.pop(0)

# ── ASYNC EMBED ────────────────────────────────────────────────────────────────
"""
ASYNC CONCEPT:
  asyncio.get_event_loop().run_in_executor() runs blocking code
  in a thread pool WITHOUT blocking the async event loop.

  Blocking (bad):    await embed_question(q)  →  freezes ALL requests
  Non-blocking:      await run_in_executor(embed_fn, q)  →  other requests
                     continue while this one embeds
"""
async def async_embed(question: str) -> list:
    loop = asyncio.get_event_loop()
    def _embed():
        return embedder.encode(
            f"Represent this sentence for searching relevant passages: {question}",
            normalize_embeddings=True
        ).tolist()
    return await loop.run_in_executor(None, _embed)

# ── ASYNC CHROMADB SEARCH ──────────────────────────────────────────────────────
async def async_chroma_search(q_vector: list, n_results: int,
                               domain: Optional[str] = None) -> dict:
    loop       = asyncio.get_event_loop()
    search_col = collection
    if domain and domain in domain_collections:
        try:
            dc = domain_collections[domain]
            if dc.count() > 0:
                search_col = dc
        except Exception:
            pass

    def _search():
        for _ in range(2):
            try:
                result = search_col.query(
                    query_embeddings=[q_vector],
                    n_results=n_results
                )
                if result["documents"][0]:
                    return result
            except Exception:
                pass
        return collection.query(query_embeddings=[q_vector], n_results=n_results)

    return await loop.run_in_executor(None, _search)

# ── ASYNC BM25 ─────────────────────────────────────────────────────────────────
async def async_bm25(question: str, top_k: int) -> list:
    if not USE_BM25:
        return []
    loop = asyncio.get_event_loop()
    def _bm25():
        tokens = question.lower().split()
        scores = bm25_index.get_scores(tokens)
        max_s  = scores.max()
        if max_s > 0:
            scores = scores / max_s
        top_i  = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_i if scores[i] > 0.05]
    return await loop.run_in_executor(None, _bm25)

# ── ASYNC RERANKER ─────────────────────────────────────────────────────────────
async def async_rerank(question: str, candidates: list, top_k: int) -> list:
    if not USE_RERANKER or not candidates:
        return candidates[:top_k]
    loop  = asyncio.get_event_loop()
    pairs = [(question, c["text"]) for c in candidates]
    def _rerank():
        scores = reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    return await loop.run_in_executor(None, _rerank)

# ── HYBRID RETRIEVAL (ASYNC) ───────────────────────────────────────────────────
def detect_domain(question: str) -> Optional[str]:
    q = question.lower()
    for domain, keywords in DOMAIN_MAP.items():
        if any(kw in q for kw in keywords):
            return domain
    return None

async def retrieve_hybrid(question: str, top_k: int, q_vector: list) -> list:
    """
    Runs semantic search AND BM25 concurrently using asyncio.gather.
    asyncio.gather = run multiple async tasks IN PARALLEL.
    Both searches run at the same time → faster than sequential.
    """
    topic_words = set(question.lower().split()) - STOPWORDS
    domain      = detect_domain(question)

    # Run both searches CONCURRENTLY (not sequentially)
    sem_results, bm25_results = await asyncio.gather(
        async_chroma_search(q_vector, top_k, domain),
        async_bm25(question, top_k)
    )

    # Build candidate dict with RRF scores
    candidates: dict = {}
    for rank, (text, dist, meta) in enumerate(zip(
        sem_results["documents"][0],
        sem_results["distances"][0],
        sem_results.get("metadatas", [[{}]*top_k])[0]
    )):
        sim  = round(1 - dist / 2, 4)
        rrf  = 1 / (rank + 60)
        candidates[text] = {
            "text":        text,
            "sem_score":   sim,
            "bm25_score":  0.0,
            "rrf_score":   rrf * SEMANTIC_WEIGHT,
            "domain":      meta.get("domain", "general"),
            "parent_text": meta.get("parent_text", ""),
        }

    # Merge BM25 results
    for rank, (idx, bm25_score) in enumerate(bm25_results):
        if idx >= len(bm25_texts):
            continue
        text = bm25_texts[idx]
        rrf  = 1 / (rank + 60)
        if text in candidates:
            candidates[text]["bm25_score"]  = bm25_score
            candidates[text]["rrf_score"]  += rrf * BM25_WEIGHT
        else:
            domain_tag = str(bm25_df.iloc[idx].get("domain","general")) \
                         if hasattr(bm25_df.iloc[idx],"get") else "general"
            candidates[text] = {
                "text": text, "sem_score": 0.0,
                "bm25_score": bm25_score,
                "rrf_score": rrf * BM25_WEIGHT,
                "domain": domain_tag, "parent_text": "",
            }

    # Filter + rank
    filtered = sorted(
        [c for c in candidates.values()
         if (c["sem_score"] >= MIN_SCORE or c["bm25_score"] >= 0.1)
         and (not topic_words or any(w in c["text"].lower() for w in topic_words))],
        key=lambda x: x["rrf_score"], reverse=True
    )
    return filtered[:top_k]

# ── DSPY GENERATION ────────────────────────────────────────────────────────────
def generate_with_dspy(question: str, chunks: list[SourceChunk]) -> tuple[str, bool]:
    """
    Generate answer using DSPy's optimized prompting.

    DSPy handles:
    - Prompt construction (no manual f-strings)
    - Chain-of-thought reasoning
    - Output parsing
    - Prompt optimization via few-shot examples

    Returns (answer, dspy_used)
    """
    if not USE_DSPY:
        return generate_fallback(question, chunks), False

    try:
        # Build numbered context for DSPy
        context_parts = []
        for chunk in chunks:
            clean = re.sub(r"(Question|Answer):\s*", "", chunk.text, flags=re.IGNORECASE)
            context_parts.append(f"[{chunk.ref}] {clean.strip()}")
        context = "\n\n".join(context_parts)

        # DSPy call — no prompt string needed, DSPy handles it
        prediction = dspy_module(context=context, question=question)
        answer     = prediction.answer if hasattr(prediction, "answer") else str(prediction)
        return answer.strip(), True

    except Exception as e:
        print(f"  DSPy generation failed: {e} — falling back")
        return generate_fallback(question, chunks), False


def generate_fallback(question: str, chunks: list[SourceChunk]) -> str:
    """Template-based fallback when DSPy is unavailable."""
    clean = []
    for chunk in chunks:
        text = re.sub(r"(Question|Answer):\s*", "", chunk.text, flags=re.IGNORECASE)
        clean.append(text.strip())
    facts   = "\n".join(f"FACT {i+1}: {t}" for i, t in enumerate(clean))
    prompt  = (
        f"<start_of_turn>user\n"
        f"You are a medical assistant. Using ONLY these facts, "
        f"answer comprehensively in 3 sentences. "
        f"Cover ALL symptoms/causes/treatments listed.\n\n"
        f"{facts}\n\nQuestion: {question}\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    try:
        r = ollama.chat(model=LLM_MODEL,
                        messages=[{"role":"user","content":prompt}],
                        options=LLM_OPTIONS)
        return r["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

# ── ASYNC GENERATE ─────────────────────────────────────────────────────────────
async def async_generate(question: str, chunks: list[SourceChunk]) -> tuple[str, bool]:
    """Run DSPy/Ollama generation in thread pool (non-blocking)."""
    loop = asyncio.get_event_loop()
    def _gen():
        return generate_with_dspy(question, chunks)
    return await loop.run_in_executor(None, _gen)

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def build_source_chunks(candidates: list) -> list[SourceChunk]:
    chunks = []
    for i, c in enumerate(candidates, start=1):
        score = c.get("rerank_score", c.get("sem_score", c.get("rrf_score", 0)))
        if "rerank_score" in c:
            score = min(1.0, max(0.0, (score + 10) / 20))
        label = "High" if score >= 0.75 else "Medium" if score >= 0.55 else "Low"
        chunks.append(SourceChunk(
            ref=i, text=c["text"], score=round(score, 4),
            label=label, domain=c.get("domain","general"),
            parent_text=c.get("parent_text","")
        ))
    return chunks

def should_answer(chunks: list[SourceChunk]) -> tuple[bool, str]:
    if not chunks:
        return False, "None"
    best = max(c.score for c in chunks)
    if best < CONFIDENCE_GATE:
        return False, "None"
    return True, ("High" if best >= 0.75 else "Medium")

def compress_context(question: str, chunks: list[SourceChunk]) -> list[SourceChunk]:
    q_words = set(question.lower().split()) - STOPWORDS
    result  = []
    for chunk in chunks:
        sents    = re.split(r'(?<=[.!?])\s+', chunk.text)
        relevant = [s for s in sents if q_words & set(s.lower().split())
                    or len(sents) <= 2]
        text = " ".join(relevant) if relevant else chunk.text
        result.append(SourceChunk(ref=chunk.ref, text=text, score=chunk.score,
                                   label=chunk.label, domain=chunk.domain,
                                   parent_text=chunk.parent_text))
    return result

def build_idk_response(question: str, best_score: float) -> str:
    if USE_DSPY:
        try:
            reason = (f"the closest match was only {int(best_score*100)}% relevant"
                      if best_score > 0 else "no relevant documents found")
            return dspy_module.explain_idk(reason=reason, question=question)
        except Exception:
            pass
    reason = (f"the closest match was only {int(best_score*100)}% relevant"
              if best_score > 0 else "no relevant documents found")
    return (f"I don't have enough information — {reason}. "
            f"Please consult a healthcare professional or visit "
            f"mayoclinic.org or medlineplus.gov.")

def postprocess_answer(answer: str, chunks: list[SourceChunk]) -> str:
    for marker in ["<start_of_turn>model","<end_of_turn>","<|assistant|>","Answer:"]:
        if marker in answer:
            answer = answer.split(marker)[-1]
            break
    noise = [
        r"<start_of_turn>.*?<end_of_turn>", r"<\|.*?\|>",
        r"respond(ing)? naturally.*?[.\n]",
        r"use only (the facts|facts).*?[.\n]",
        r"never (copy|repeat|mention).*?[.\n]",
        r"as a medical assistant.*?[,.]",
        r"question:.*?(?=\n|$)", r"\[\d+\]", r"nih:.*?(?=\n|$)",
    ]
    for p in noise:
        answer = re.sub(p, "", answer, flags=re.IGNORECASE | re.DOTALL)
    def fix_caps(w):
        if len(w) <= 1: return w
        if any(c.isupper() for c in w[1:]):
            return (w[0].upper()+w[1:].lower()) if w[0].isupper() else w.lower()
        return w
    answer = " ".join(fix_caps(w) for w in answer.split())
    for exact, synonyms in MEDICAL_TERMS.items():
        if exact not in answer.lower():
            for syn in synonyms:
                if syn in answer.lower():
                    answer = answer.replace(syn, f"{syn} ({exact})")
                    break
    sents = re.split(r'(?<=[.!?])\s+', answer.strip())
    sents = [s for s in sents if len(s.split()) > 4]
    if len(sents) > 3:
        answer = " ".join(sents[:3])
    answer = re.sub(r"\s{2,}", " ", answer)
    return answer.strip()

def postfilter_answer(answer: str, chunks: list[SourceChunk],
                      question: str) -> tuple[str, bool]:
    if len(answer.split()) < 5:
        return build_idk_response(question, 0), False
    q_clean = re.sub(r"[^\w\s]","",question).lower().strip()
    a_clean = re.sub(r"[^\w\s]","",answer).lower().strip()
    if q_clean == a_clean:
        return build_idk_response(question, 0), False
    return answer, True

def log_evaluation(question, chunks, answered, cache_hit, dspy_used,
                   embed_ms, retrieve_ms, rerank_ms, llm_ms, total_ms, answer):
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), question,
                round(max((c.score for c in chunks), default=0), 4),
                len(chunks), answered, cache_hit, dspy_used,
                embed_ms, retrieve_ms, rerank_ms, llm_ms, total_ms,
                answer[:120].replace("\n"," ")
            ])
    except Exception:
        pass

# ── CORE ASYNC PIPELINE ────────────────────────────────────────────────────────
async def run_pipeline_async(question: str, top_k: int) -> ChatResponse:
    """
    Full async RAG pipeline.
    Every blocking operation runs in a thread pool via run_in_executor.
    This means the event loop is never blocked — other requests proceed normally.
    """
    t0 = time.time()

    # Layer 1: Exact cache (instant)
    cached = get_exact_cached(question)
    if cached:
        cached.cache_hit = True
        return cached

    # Layer 2: Embed (async — non-blocking)
    t1       = time.time()
    q_vector = await async_embed(question)
    embed_ms = int((time.time()-t1)*1000)
    print(f"  Embed     → {embed_ms}ms")

    # Layer 3: Semantic cache
    sem_cached = get_semantic_cached(q_vector)
    if sem_cached:
        sem_cached.cache_hit = True
        return sem_cached

    # Layer 4: Hybrid retrieval (BM25 + semantic run CONCURRENTLY)
    t2           = time.time()
    candidates   = await retrieve_hybrid(question, TOP_K_RETRIEVE, q_vector)
    retrieve_ms  = int((time.time()-t2)*1000)
    print(f"  Retrieve  → {retrieve_ms}ms ({len(candidates)} candidates)")

    # Layer 5: Reranker (async)
    t3        = time.time()
    reranked  = await async_rerank(question, candidates, TOP_K_RERANK)
    rerank_ms = int((time.time()-t3)*1000)
    print(f"  Rerank    → {rerank_ms}ms")

    chunks     = build_source_chunks(reranked[:TOP_K_FINAL])
    chunks     = compress_context(question, chunks)
    can_answer, confidence = should_answer(chunks)
    best_score = max((c.score for c in chunks), default=0)

    if not can_answer:
        answer = build_idk_response(question, best_score)
        total  = int((time.time()-t0)*1000)
        log_evaluation(question, chunks, False, False, False,
                       embed_ms, retrieve_ms, rerank_ms, 0, total, answer)
        result = ChatResponse(
            question=question, answer=answer, sources=chunks,
            answered=False, confidence="None",
            latency_ms=total, cache_hit=False, dspy_used=False
        )
        set_exact_cache(question, result)
        return result

    # Layer 6: DSPy generation (async — runs in thread pool)
    t4              = time.time()
    raw, dspy_used  = await async_generate(question, chunks)
    llm_ms          = int((time.time()-t4)*1000)
    print(f"  LLM       → {llm_ms}ms (DSPy={'on' if dspy_used else 'off'})")

    answer          = postprocess_answer(raw, chunks)
    answer, is_good = postfilter_answer(answer, chunks, question)
    total           = int((time.time()-t0)*1000)
    print(f"  TOTAL     → {total}ms")

    log_evaluation(question, chunks, is_good, False, dspy_used,
                   embed_ms, retrieve_ms, rerank_ms, llm_ms, total, answer)

    result = ChatResponse(
        question=question, answer=answer, sources=chunks,
        answered=is_good, confidence=confidence if is_good else "None",
        latency_ms=total, cache_hit=False, dspy_used=dspy_used
    )
    set_exact_cache(question, result)
    set_semantic_cache(question, q_vector, result)
    return result

# ── BACKGROUND QUEUE WORKER ────────────────────────────────────────────────────
"""
WORKER CONCEPT:
  The worker runs in the background as an infinite async loop.
  It takes items from the queue ONE AT A TIME and processes them.
  This prevents multiple LLM calls from competing for RAM.

  asyncio.create_task() starts the worker at app startup.
  It runs concurrently with the FastAPI request handlers.

  Flow:
  User → POST /chat → item added to queue → HTTP response WAITS
  Worker → takes item from queue → runs pipeline → sets Future result
  HTTP response → Future resolved → answer returned to user
"""
async def queue_worker():
    """Background worker: drains the request queue one item at a time."""
    print("  Queue worker started ✓")
    while True:
        try:
            item: QueueItem = await request_queue.get()
            try:
                result = await run_pipeline_async(item.question, item.top_k)
                if not item.future.done():
                    item.future.set_result(result)
            except Exception as e:
                if not item.future.done():
                    item.future.set_exception(e)
            finally:
                request_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  Worker error: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Start background queue worker when FastAPI starts."""
    asyncio.create_task(queue_worker())
    print("  Background queue worker started")

# ── API ROUTES ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":          "ok",
        "chunk_count":     collection.count(),
        "embed_model":     EMBED_MODEL,
        "llm_model":       LLM_MODEL,
        "dspy_active":     USE_DSPY,
        "bm25_active":     USE_BM25,
        "reranker_active": USE_RERANKER,
        "queue_size":      request_queue.qsize(),
        "queue_capacity":  MAX_QUEUE_SIZE,
        "exact_cache":     len(_exact_cache),
        "sem_cache":       len(_semantic_cache),
        "num_ctx":         NUM_CTX,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Queued async endpoint.
    Request goes into the queue, worker processes it, response returns.
    If queue is full → 503 error (server busy).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Check exact cache before even touching the queue
    cached = get_exact_cached(req.question)
    if cached:
        cached.cache_hit = True
        return cached

    if request_queue.full():
        raise HTTPException(
            status_code=503,
            detail=f"Server busy — queue full ({MAX_QUEUE_SIZE} requests). Try again shortly."
        )

    # Create queue item with a Future that will hold the result
    loop     = asyncio.get_event_loop()
    future   = loop.create_future()
    item     = QueueItem(req.question, req.top_k, str(uuid_lib.uuid4()))
    item.future = future

    queue_enqueue_time = time.time()
    await request_queue.put(item)

    # Wait for worker to process this item (up to QUEUE_TIMEOUT seconds)
    try:
        result = await asyncio.wait_for(future, timeout=QUEUE_TIMEOUT)
        result.queue_wait_ms = int((time.time() - queue_enqueue_time) * 1000)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out in queue. Server may be overloaded."
        )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming async endpoint — answer appears word by word.
    Bypasses queue for streaming (better UX for streaming).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def generate():
        t0 = time.time()

        cached = get_exact_cached(req.question)
        if cached:
            for word in cached.answer.split():
                yield f"data: {json.dumps({'token': word+' '})}\n\n"
            yield f"data: {json.dumps({'done':True,'answered':cached.answered,'confidence':cached.confidence,'sources':[],'latency_ms':1,'cache_hit':True,'dspy_used':False})}\n\n"
            return

        q_vector   = await async_embed(req.question)
        sem_cached = get_semantic_cached(q_vector)
        if sem_cached:
            for word in sem_cached.answer.split():
                yield f"data: {json.dumps({'token': word+' '})}\n\n"
            total = int((time.time()-t0)*1000)
            yield f"data: {json.dumps({'done':True,'answered':sem_cached.answered,'confidence':sem_cached.confidence,'sources':[],'latency_ms':total,'cache_hit':True,'dspy_used':False})}\n\n"
            return

        candidates = await retrieve_hybrid(req.question, TOP_K_RETRIEVE, q_vector)
        reranked   = await async_rerank(req.question, candidates, TOP_K_RERANK)
        chunks     = build_source_chunks(reranked[:TOP_K_FINAL])
        chunks     = compress_context(req.question, chunks)
        can_answer, confidence = should_answer(chunks)
        best_score = max((c.score for c in chunks), default=0)
        sources_data = [{"ref":c.ref,"text":c.text,"score":c.score,
                          "label":c.label,"domain":c.domain} for c in chunks]

        if not can_answer:
            idk = build_idk_response(req.question, best_score)
            for word in idk.split():
                yield f"data: {json.dumps({'token': word+' '})}\n\n"
            yield f"data: {json.dumps({'done':True,'answered':False,'confidence':'None','sources':sources_data})}\n\n"
            return

        # Stream from Ollama (DSPy doesn't support streaming, use ollama directly)
        full_answer = ""
        dspy_used_stream = False
        try:
            if USE_DSPY:
                # DSPy: generate full then stream word by word
                raw, dspy_used_stream = await async_generate(req.question, chunks)
                for word in raw.split():
                    yield f"data: {json.dumps({'token': word+' '})}\n\n"
                    await asyncio.sleep(0)  # yield control to event loop
                full_answer = raw
            else:
                # Direct Ollama streaming
                clean = []
                for chunk in chunks:
                    text = re.sub(r"(Question|Answer):\s*","",chunk.text,flags=re.IGNORECASE)
                    clean.append(text.strip())
                facts  = "\n".join(f"FACT {i+1}: {t}" for i,t in enumerate(clean))
                prompt = (f"<start_of_turn>user\nYou are a medical assistant. "
                          f"Answer using ONLY the facts below in 3 sentences.\n\n"
                          f"{facts}\n\nQuestion: {req.question}\n"
                          f"<end_of_turn>\n<start_of_turn>model\n")
                stream = ollama.chat(model=LLM_MODEL,
                                     messages=[{"role":"user","content":prompt}],
                                     stream=True, options=LLM_OPTIONS)
                for chunk in stream:
                    token = chunk["message"]["content"]
                    if token:
                        full_answer += token
                        yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        answer     = postprocess_answer(full_answer, chunks)
        answer, ok = postfilter_answer(answer, chunks, req.question)
        total      = int((time.time()-t0)*1000)
        log_evaluation(req.question, chunks, ok, False, dspy_used_stream,
                       0, 0, 0, total, total, answer)

        result = ChatResponse(
            question=req.question, answer=answer, sources=chunks,
            answered=ok, confidence=confidence if ok else "None",
            latency_ms=total, cache_hit=False, dspy_used=dspy_used_stream
        )
        set_exact_cache(req.question, result)
        set_semantic_cache(req.question, q_vector, result)

        yield f"data: {json.dumps({'done':True,'answered':ok,'confidence':confidence if ok else 'None','sources':sources_data,'latency_ms':total,'cache_hit':False,'dspy_used':dspy_used_stream})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.get("/queue/stats")
async def queue_stats():
    return {
        "queue_size":     request_queue.qsize(),
        "queue_capacity": MAX_QUEUE_SIZE,
        "exact_cache":    len(_exact_cache),
        "sem_cache":      len(_semantic_cache),
    }

@app.delete("/cache/clear")
async def cache_clear():
    _exact_cache.clear(); _semantic_cache.clear()
    return {"message": "Caches cleared."}

@app.get("/eval/summary")
async def eval_summary():
    if not os.path.exists(LOG_FILE):
        return {"message": "No data yet."}
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"message": "Log is empty."}
    total      = len(rows)
    answered   = sum(1 for r in rows if r["answered"]=="True")
    cache_hits = sum(1 for r in rows if r.get("cache_hit")=="True")
    dspy_used  = sum(1 for r in rows if r.get("dspy_used")=="True")
    avg_score  = round(sum(float(r["top_chunk_score"]) for r in rows)/total, 4)
    avg_ms     = round(sum(int(r.get("total_ms",r.get("latency_ms",0))) for r in rows)/total)
    return {
        "total":          total,
        "answered":       answered,
        "cache_hit_pct":  round(cache_hits/total*100, 1),
        "dspy_pct":       round(dspy_used/total*100, 1),
        "avg_chunk_score":avg_score,
        "avg_latency_ms": avg_ms,
    }

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8001,
                reload=True, reload_dirs=[r"H:\Rag_chatbot"])