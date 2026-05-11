"""
MedRAG — Step 6 Only: Save to ChromaDB
========================================
Run this when steps 1-5b already completed.
Loads from: _step5b_chunks_dedup.json + _step5b_vectors_dedup.npy
Saves to:   chroma_db/ + vectors_backup.npy + chunks_bm25.csv

Run: python step6_save_chromadb.py
"""

import pandas  as pd
import numpy   as np
import chromadb
import uuid, os, json, time

# ── CONFIG — must match your phase1_indexing.py ────────────────────────────────
CHROMA_DB_PATH  = r"H:\Rag_chatbot\chroma_db"
WORK_DIR        = r"H:\Rag_chatbot"
COLLECTION_NAME = "medical_qa"
SAVE_BATCH      = 500

VECTORS_FINAL_PATH = os.path.join(WORK_DIR, "vectors_backup.npy")
BM25_FINAL_PATH    = os.path.join(WORK_DIR, "chunks_bm25.csv")
CHUNKS_DEDUP_PATH  = os.path.join(WORK_DIR, "_step5b_chunks_dedup.json")
VECTORS_DEDUP_PATH = os.path.join(WORK_DIR, "_step5b_vectors_dedup.npy")

SENTENCES_PER_CHUNK = 3
SENTENCE_OVERLAP    = 1
EMBED_MODEL         = "BAAI/bge-base-en-v1.5"
DEDUP_THRESHOLD     = 0.92

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

# ── STEP 1: VERIFY CHECKPOINT FILES EXIST ─────────────────────────────────────
print("=" * 62)
print("  MedRAG Step 6 — Save to ChromaDB")
print("=" * 62)

if not os.path.exists(CHUNKS_DEDUP_PATH):
    print(f"\n✗ File not found: {CHUNKS_DEDUP_PATH}")
    print("  Steps 1-5b must have completed first.")
    print("  Check H:\\Rag_chatbot\\ for _step5b_chunks_dedup.json")
    exit(1)

if not os.path.exists(VECTORS_DEDUP_PATH):
    print(f"\n✗ File not found: {VECTORS_DEDUP_PATH}")
    print("  Steps 1-5b must have completed first.")
    exit(1)

print("\n  ✓ Checkpoint files found. Loading...")

# ── STEP 2: LOAD CHECKPOINTS ──────────────────────────────────────────────────
print("\n[1/3] Loading deduplicated chunks and vectors...")

with open(CHUNKS_DEDUP_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

all_chunks   = data["chunks"]
all_metadata = data["metadata"]
vectors      = np.load(VECTORS_DEDUP_PATH)

print(f"      Chunks   : {len(all_chunks):,}")
print(f"      Vectors  : {vectors.shape}")
print(f"      Sample   : {all_chunks[0][:100]}...")

# ── STEP 3: SAVE TO CHROMADB ──────────────────────────────────────────────────
print(f"\n[2/3] Saving {len(all_chunks):,} chunks to ChromaDB...")

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
t0 = time.time()

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
# Save domain collections — split into sub-batches to avoid ChromaDB limit
print("\n      Saving domain collections...")
DOMAIN_BATCH = 2000   # safe size well under ChromaDB's 5461 limit

for domain, batch in domain_batches.items():
    if not batch["ids"]:
        continue
    d_total = len(batch["ids"])
    for s in range(0, d_total, DOMAIN_BATCH):
        e = min(s + DOMAIN_BATCH, d_total)
        domain_collections[domain].add(
            ids       = batch["ids"][s:e],
            documents = batch["docs"][s:e],
            embeddings= batch["vecs"][s:e],
            metadatas = batch["meta"][s:e]
        )
    print(f"      {domain:15s}: {d_total:,} chunks")

# ── STEP 4: SAVE FINAL OUTPUT FILES ───────────────────────────────────────────
print(f"\n[3/3] Saving final output files...")

np.save(VECTORS_FINAL_PATH, vectors)
print(f"      ✓ Saved → {VECTORS_FINAL_PATH} "
      f"({os.path.getsize(VECTORS_FINAL_PATH)//1024//1024}MB)")

bm25_df = pd.DataFrame({
    "text":   all_chunks,
    "domain": [m["domain"] for m in all_metadata],
    "id":     ids
})
bm25_df.to_csv(BM25_FINAL_PATH, index=False)
print(f"      ✓ Saved → {BM25_FINAL_PATH} "
      f"({os.path.getsize(BM25_FINAL_PATH)//1024//1024}MB)")

# ── CLEAN UP CHECKPOINTS ──────────────────────────────────────────────────────
print("\n      Cleaning up checkpoint files...")
for f in [r"H:\Rag_chatbot\_step4_chunks_raw.json",
          r"H:\Rag_chatbot\_step5_vectors_raw.npy",
          CHUNKS_DEDUP_PATH,
          VECTORS_DEDUP_PATH]:
    try:
        os.remove(f)
        print(f"      Deleted: {os.path.basename(f)}")
    except Exception:
        pass

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 6 COMPLETE — ChromaDB Ready")
print("=" * 62)
print(f"  Main collection  : {collection.count():,} chunks")
for domain, col in domain_collections.items():
    cnt = col.count()
    if cnt > 0:
        print(f"  {domain:16s}: {cnt:,} chunks")
print(f"\n  Chunking style   : Sentence-based (Q&A optimized)")
print(f"  Embedding model  : {EMBED_MODEL}")
print(f"  Dedup threshold  : {DEDUP_THRESHOLD}")
print(f"  DB path          : {CHROMA_DB_PATH}")
print(f"\n  Output files:")
print(f"  ✓ vectors_backup.npy")
print(f"  ✓ chunks_bm25.csv")
print(f"  ✓ chroma_db/")
print(f"\n  Start backend:")
print(f"  uvicorn backend_api:app --port 8001 --reload-dir H:\\Rag_chatbot")
print("=" * 62)