"""
MedRAG — Phase 1 Indexing v4.0
================================
Chunking strategy: Sentence-based (best for Q&A CSV data)

Why sentence-based for this dataset:
  - Each CSV row = one Question + one Answer
  - The Question must ALWAYS stay with its Answer (never split)
  - The Answer may be long — split at sentence boundaries only
  - Result: every chunk starts with the Question + 1-3 Answer sentences
  - This gives maximum retrieval precision for Q&A medical data

Other techniques kept:
  - Fast matrix deduplication (2-5 mins, not all day)
  - Hierarchical storage (parent Q+A + child chunks)
  - Domain partitioning (8 medical collections)
  - Precomputed embeddings (all offline before DB insert)
  - Checkpoint saves after every step (resume if interrupted)
  - Rich metadata tagging

Run: python phase1_indexing.py
If interrupted, run again — resumes from last completed step.
"""

import pandas  as pd
import numpy   as np
import chromadb
import uuid, re, os, time, json
import nltk
from langchain_community.document_loaders import CSVLoader
from sentence_transformers import SentenceTransformer

# Download NLTK sentence tokenizer (only once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FILE_PATH       = r"H:\Rag_chatbot\Data\medquad.csv"
CHROMA_DB_PATH  = r"H:\Rag_chatbot\chroma_db"
WORK_DIR        = r"H:\Rag_chatbot"
COLLECTION_NAME = "medical_qa"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"

# Sentence-based chunking config
SENTENCES_PER_CHUNK = 3     # how many answer sentences per chunk
SENTENCE_OVERLAP    = 1     # how many sentences overlap between chunks
MIN_CHUNK_WORDS     = 8     # skip chunks shorter than this
MAX_CHUNK_CHARS     = 600   # hard cap (safety net)

EMBED_BATCH     = 32
DEDUP_THRESHOLD = 0.92
SAVE_BATCH      = 500

# ── INTERMEDIATE CHECKPOINT PATHS ─────────────────────────────────────────────
CHUNKS_RAW_PATH    = os.path.join(WORK_DIR, "_step4_chunks_raw.json")
VECTORS_RAW_PATH   = os.path.join(WORK_DIR, "_step5_vectors_raw.npy")
CHUNKS_DEDUP_PATH  = os.path.join(WORK_DIR, "_step5b_chunks_dedup.json")
VECTORS_DEDUP_PATH = os.path.join(WORK_DIR, "_step5b_vectors_dedup.npy")
# Final output files (used by backend)
VECTORS_FINAL_PATH = os.path.join(WORK_DIR, "vectors_backup.npy")
BM25_FINAL_PATH    = os.path.join(WORK_DIR, "chunks_bm25.csv")

# ── DOMAIN MAP ─────────────────────────────────────────────────────────────────
DOMAIN_MAP = {
    "diabetes":    ["diabetes","insulin","glucose","blood sugar","diabetic","a1c"],
    "heart":       ["heart","cardiac","cholesterol","hypertension","blood pressure",
                    "artery","coronary","angina","stroke"],
    "respiratory": ["asthma","lung","breathing","pneumonia","bronch","inhaler",
                    "copd","oxygen","respiratory","airway"],
    "kidney":      ["kidney","renal","urine","dialysis","nephro","creatinine"],
    "cancer":      ["cancer","tumor","tumour","chemotherapy","oncology",
                    "malignant","radiation","biopsy"],
    "infection":   ["infection","bacteria","virus","antibiotic","fever",
                    "flu","immune","vaccine","sepsis"],
    "neurology":   ["brain","neuron","alzheimer","parkinson","seizure",
                    "migraine","nerve","dementia"],
    "mental":      ["depression","anxiety","mental","psychiatric","therapy",
                    "bipolar","schizophrenia","ptsd"],
}

def detect_domain(text: str) -> str:
    t = text.lower()
    for domain, keywords in DOMAIN_MAP.items():
        if any(kw in t for kw in keywords):
            return domain
    return "general"

# ── SENTENCE-BASED CHUNKING ────────────────────────────────────────────────────
def sentence_chunk_qa(question: str, answer: str,
                      n: int = SENTENCES_PER_CHUNK,
                      overlap: int = SENTENCE_OVERLAP) -> list[str]:
    """
    Sentence-based chunking designed specifically for Q&A CSV rows.

    Strategy:
      1. Always keep the Question intact (never split it)
      2. Split the Answer into individual sentences using NLTK
      3. Group sentences into windows of n with overlap
      4. Prepend the Question to EVERY chunk (so each chunk is self-contained)

    Why this is best for Q&A data:
      - Every chunk contains the full question → embedding captures intent
      - Answer sentences stay at natural boundaries (not mid-sentence)
      - Overlap prevents losing context at chunk boundaries
      - Short answers (1-2 sentences) stay as one chunk

    Example:
      Question: "What is diabetes?"
      Answer sentences: ["S1. Diabetes is a disease.", "S2. It affects sugar.",
                         "S3. It can damage organs.", "S4. Management helps."]
      Chunks produced (n=2, overlap=1):
        Chunk 1: "Question: What is diabetes? Answer: S1. S2."
        Chunk 2: "Question: What is diabetes? Answer: S2. S3."  ← S2 overlaps
        Chunk 3: "Question: What is diabetes? Answer: S3. S4."
    """
    # Tokenize answer into sentences using NLTK punkt
    sentences = sent_tokenize(answer)

    # Filter out very short sentence fragments
    sentences = [s.strip() for s in sentences if len(s.split()) > 2]

    # If answer is short (1-3 sentences) keep as single chunk — no splitting needed
    if len(sentences) <= n:
        chunk_text = f"Question: {question} Answer: {answer}"
        if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
            return [chunk_text]
        return []

    # Sliding window over answer sentences with overlap
    chunks = []
    step   = max(1, n - overlap)  # how many sentences to advance each time

    for start in range(0, len(sentences), step):
        end         = min(start + n, len(sentences))
        window      = sentences[start:end]
        answer_part = " ".join(window)
        chunk_text  = f"Question: {question} Answer: {answer_part}"

        # Quality filters
        if len(chunk_text.split()) < MIN_CHUNK_WORDS:
            continue
        if len(chunk_text) > MAX_CHUNK_CHARS:
            # Hard cap: truncate at last full sentence before limit
            chunk_text = chunk_text[:MAX_CHUNK_CHARS].rsplit(".", 1)[0] + "."

        chunks.append(chunk_text)

        # Stop if we've reached the last sentence
        if end == len(sentences):
            break

    return chunks

# ── FAST DEDUPLICATION ─────────────────────────────────────────────────────────
def fast_deduplicate(texts: list, vectors: np.ndarray,
                     threshold: float = DEDUP_THRESHOLD) -> list:
    """
    Fast deduplication using matrix dot product.
    For normalized vectors: dot product = cosine similarity.
    O(n × kept) with vectorized numpy — runs in 2-5 mins for 75k chunks.
    """
    n         = len(vectors)
    keep_mask = np.ones(n, dtype=bool)
    kept_vecs = []
    removed   = 0

    for i in range(n):
        if not keep_mask[i]:
            continue
        if kept_vecs:
            kept_matrix = np.array(kept_vecs)
            sims = kept_matrix @ vectors[i]
            if np.any(sims >= threshold):
                keep_mask[i] = False
                removed += 1
                continue
        kept_vecs.append(vectors[i])

        if (i + 1) % 5000 == 0:
            pct  = int((i+1) / n * 100)
            kept = int(keep_mask[:i+1].sum())
            print(f"      Dedup progress: {i+1:,}/{n:,} ({pct}%) "
                  f"kept={kept:,} removed={removed:,}")

    keep_indices = [i for i in range(n) if keep_mask[i]]
    print(f"      Dedup complete: {n:,} → {len(keep_indices):,} "
          f"({removed:,} duplicates removed, {int(removed/n*100)}% reduction)")
    return keep_indices

# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────
def save_chunks_json(chunks: list, metadata: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metadata": metadata}, f, ensure_ascii=False)
    size_mb = os.path.getsize(path) // 1024 // 1024
    print(f"      Saved {len(chunks):,} chunks ({size_mb}MB) → {path}")

def load_chunks_json(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"      Loaded {len(data['chunks']):,} chunks ← {path}")
    return data["chunks"], data["metadata"]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  MedRAG Phase 1 Indexing v4.0")
print("  Chunking: Sentence-based (Q&A CSV optimized)")
print("  Saves after every step — safe to interrupt and resume")
print("=" * 62)

# ── STEP 1-4: LOAD → CLEAN → SENTENCE CHUNK ───────────────────────────────────
if os.path.exists(CHUNKS_RAW_PATH):
    print(f"\n[1-4/6] ✓ SKIPPING — found checkpoint: {CHUNKS_RAW_PATH}")
    all_chunks, all_metadata = load_chunks_json(CHUNKS_RAW_PATH)

else:
    # STEP 1: Load CSV
    print("\n[1/6] Loading CSV...")
    df = pd.read_csv(FILE_PATH, encoding="utf-8")
    print(f"      Shape: {df.shape}")
    loader    = CSVLoader(file_path=FILE_PATH, encoding="utf-8")
    documents = loader.load()
    print(f"      Documents loaded: {len(documents)}")

    # STEP 2: Clean + Parse Q&A
    print("\n[2/6] Cleaning and parsing Q&A rows...")
    qa_pairs     = []   # list of (question, answer, metadata) tuples
    skipped      = 0

    for doc in documents:
        lines = doc.page_content.split("\n")
        data  = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip().lower()] = v.strip()

        question = data.get("question", "").replace("\n", " ").strip()
        answer   = data.get("answer",   "").replace("\n", " ").strip()

        if not question or not answer:
            skipped += 1
            continue
        if len(f"{question} {answer}".split()) < 8:
            skipped += 1
            continue

        qa_pairs.append((question, answer, doc.metadata))

    print(f"      Valid Q&A pairs: {len(qa_pairs):,} (skipped {skipped})")

    # STEP 3: (No separate embed model needed for sentence chunking)
    # NLTK handles sentence splitting — no ML model required
    print("\n[3/6] Sentence tokenizer: NLTK punkt (no model download needed)")

    # STEP 4: Sentence-based chunking
    print(f"\n[4/6] Sentence-based chunking "
          f"(n={SENTENCES_PER_CHUNK} sentences, overlap={SENTENCE_OVERLAP})...")

    all_chunks   = []
    all_metadata = []
    single_chunk = 0   # count rows that produced only 1 chunk (short answers)
    multi_chunk  = 0   # count rows that were split into multiple chunks

    for idx, (question, answer, doc_meta) in enumerate(qa_pairs):
        parent_id = str(uuid.uuid4())
        parent_text = f"Question: {question} Answer: {answer}"
        domain    = detect_domain(parent_text)

        # Apply sentence-based chunking
        chunks = sentence_chunk_qa(question, answer,
                                   n=SENTENCES_PER_CHUNK,
                                   overlap=SENTENCE_OVERLAP)

        if not chunks:
            continue

        if len(chunks) == 1:
            single_chunk += 1
        else:
            multi_chunk += 1

        for chunk_text in chunks:
            all_chunks.append(chunk_text)
            all_metadata.append({
                "type":          "medical_qa",
                "domain":        domain,
                "parent_id":     parent_id,
                "parent_text":   parent_text[:300],   # full Q+A for hierarchical retrieval
                "word_count":    str(len(chunk_text.split())),
                "source":        "medquad",
                "question_text": question[:100],      # original question for filtering
            })

        # Progress every 1000 rows
        if (idx + 1) % 1000 == 0:
            print(f"      Processed {idx+1:,}/{len(qa_pairs):,} rows "
                  f"| {len(all_chunks):,} chunks so far...")

    print(f"\n      Chunking complete:")
    print(f"      Total chunks      : {len(all_chunks):,}")
    print(f"      Single-chunk rows : {single_chunk:,} (short answers, kept intact)")
    print(f"      Multi-chunk rows  : {multi_chunk:,} (long answers, split by sentence)")
    print(f"      Avg chunks/row    : {len(all_chunks)/len(qa_pairs):.1f}")

    # Sample output to verify
    print("\n      --- Sample chunk 1 (single) ---")
    print(all_chunks[0][:200])
    print("\n      --- Sample chunk 2 ---")
    print(all_chunks[min(1, len(all_chunks)-1)][:200])
    print(f"\n      Metadata sample: {all_metadata[0]}")

    # SAVE checkpoint after step 4
    print("\n      Saving step 4 checkpoint...")
    save_chunks_json(all_chunks, all_metadata, CHUNKS_RAW_PATH)
    print("      ✓ Checkpoint saved — safe to interrupt now")

# ── STEP 5: EMBED ─────────────────────────────────────────────────────────────
if os.path.exists(VECTORS_RAW_PATH):
    print(f"\n[5/6] ✓ SKIPPING embedding — found: {VECTORS_RAW_PATH}")
    vectors_raw = np.load(VECTORS_RAW_PATH)
    print(f"      Vectors shape: {vectors_raw.shape}")
else:
    print(f"\n[5/6] Loading embedding model: {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"      Embedding {len(all_chunks):,} chunks (batch={EMBED_BATCH})...")
    t0 = time.time()

    vectors_raw = embedder.encode(
        all_chunks,
        batch_size          = EMBED_BATCH,
        show_progress_bar   = True,
        normalize_embeddings = True      # unit vectors = cosine sim = dot product
    )

    elapsed = int(time.time() - t0)
    print(f"      Done in {elapsed}s | shape: {vectors_raw.shape}")

    # SAVE checkpoint
    np.save(VECTORS_RAW_PATH, vectors_raw)
    print(f"      ✓ Vectors saved → {VECTORS_RAW_PATH}")

# ── STEP 5b: FAST DEDUPLICATION ───────────────────────────────────────────────
if os.path.exists(CHUNKS_DEDUP_PATH) and os.path.exists(VECTORS_DEDUP_PATH):
    print(f"\n[5b/6] ✓ SKIPPING dedup — found checkpoints")
    all_chunks, all_metadata = load_chunks_json(CHUNKS_DEDUP_PATH)
    vectors = np.load(VECTORS_DEDUP_PATH)
    print(f"       Deduped: {vectors.shape[0]:,} unique chunks")
else:
    print(f"\n[5b/6] Fast deduplication (threshold={DEDUP_THRESHOLD})...")
    print("       Matrix dot product method — 2-5 minutes...")
    t0 = time.time()

    keep_indices = fast_deduplicate(all_chunks, vectors_raw, DEDUP_THRESHOLD)

    all_chunks   = [all_chunks[i]   for i in keep_indices]
    all_metadata = [all_metadata[i] for i in keep_indices]
    vectors      = vectors_raw[keep_indices]

    print(f"       Completed in {int(time.time()-t0)}s | "
          f"Final: {len(all_chunks):,} unique chunks")

    # SAVE checkpoint
    np.save(VECTORS_DEDUP_PATH, vectors)
    save_chunks_json(all_chunks, all_metadata, CHUNKS_DEDUP_PATH)
    print(f"       ✓ Dedup checkpoint saved")

# ── STEP 6: SAVE TO CHROMADB ──────────────────────────────────────────────────
print(f"\n[6/6] Saving {len(all_chunks):,} chunks to ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Recreate main collection
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"      Deleted existing: {COLLECTION_NAME}")
except Exception:
    pass

collection = client.create_collection(
    name     = COLLECTION_NAME,
    metadata = {"hnsw:space": "cosine"}
)

# Recreate domain collections
domain_collections = {}
for domain in list(DOMAIN_MAP.keys()) + ["general"]:
    try:
        client.delete_collection(f"medical_{domain}")
    except Exception:
        pass
    domain_collections[domain] = client.create_collection(
        name     = f"medical_{domain}",
        metadata = {"hnsw:space": "cosine"}
    )
print(f"      Created {len(domain_collections)} domain collections")

# Batch inserts
ids            = [str(uuid.uuid4()) for _ in all_chunks]
total          = len(ids)
domain_batches = {d: {"ids":[],"docs":[],"vecs":[],"meta":[]}
                  for d in domain_collections}

print(f"      Inserting in batches of {SAVE_BATCH}...")

for start in range(0, total, SAVE_BATCH):
    end = min(start + SAVE_BATCH, total)

    # Main collection
    collection.add(
        ids       = ids[start:end],
        documents = all_chunks[start:end],
        embeddings= vectors[start:end].tolist(),
        metadatas = all_metadata[start:end]
    )

    # Domain partition buckets
    for i in range(start, end):
        d = all_metadata[i].get("domain", "general")
        if d in domain_batches:
            domain_batches[d]["ids"].append(ids[i])
            domain_batches[d]["docs"].append(all_chunks[i])
            domain_batches[d]["vecs"].append(vectors[i].tolist())
            domain_batches[d]["meta"].append(all_metadata[i])

    pct = int(end / total * 100)
    print(f"      Main: {end:,}/{total:,} ({pct}%)")

# Save domain collections
print("\n      Saving domain collections...")
for domain, batch in domain_batches.items():
    if not batch["ids"]:
        continue
    domain_collections[domain].add(
        ids       = batch["ids"],
        documents = batch["docs"],
        embeddings= batch["vecs"],
        metadatas = batch["meta"]
    )
    print(f"      {domain:15s}: {len(batch['ids']):,} chunks")

# ── SAVE FINAL OUTPUT FILES ────────────────────────────────────────────────────
np.save(VECTORS_FINAL_PATH, vectors)
print(f"\n      Saved → {VECTORS_FINAL_PATH} "
      f"({os.path.getsize(VECTORS_FINAL_PATH)//1024//1024}MB)")

bm25_df = pd.DataFrame({
    "text":   all_chunks,
    "domain": [m["domain"]   for m in all_metadata],
    "id":     ids
})
bm25_df.to_csv(BM25_FINAL_PATH, index=False)
print(f"      Saved → {BM25_FINAL_PATH} "
      f"({os.path.getsize(BM25_FINAL_PATH)//1024//1024}MB)")

# ── CLEAN UP CHECKPOINTS ──────────────────────────────────────────────────────
print("\n      Cleaning up checkpoint files...")
for f in [CHUNKS_RAW_PATH, VECTORS_RAW_PATH,
          CHUNKS_DEDUP_PATH, VECTORS_DEDUP_PATH]:
    try:
        os.remove(f)
        print(f"      Deleted: {os.path.basename(f)}")
    except Exception:
        pass

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  INDEXING COMPLETE")
print("=" * 62)
print(f"  Main collection  : {collection.count():,} chunks")
for domain, col in domain_collections.items():
    cnt = col.count()
    if cnt > 0:
        print(f"  {domain:16s}: {cnt:,} chunks")
print(f"\n  Chunking style   : Sentence-based (Q&A optimized)")
print(f"  Sentences/chunk  : {SENTENCES_PER_CHUNK} (overlap={SENTENCE_OVERLAP})")
print(f"  Embedding model  : {EMBED_MODEL}")
print(f"  Dedup threshold  : {DEDUP_THRESHOLD}")
print(f"  DB path          : {CHROMA_DB_PATH}")
print(f"\n  Final files:")
print(f"  → vectors_backup.npy")
print(f"  → chunks_bm25.csv")
print(f"\n  Next step:")
print(f"  uvicorn backend_api:app --port 8001 --reload-dir H:\\Rag_chatbot")
print("=" * 62)