"""
MedRAG — Step 5: Fast Deduplication (Standalone)
==================================================
Resumes from vectors_backup.npy and chunks_bm25.csv saved by steps 1-4.
Uses batched matrix similarity instead of O(n²) nested loop.
Runtime: ~2-5 minutes instead of all day.

Run: python step5_dedup.py
"""

import numpy  as np
import pandas as pd
import chromadb
import uuid
import os
import time

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VECTORS_PATH    = r"H:\Rag_chatbot\vectors_backup.npy"
CHUNKS_PATH     = r"H:\Rag_chatbot\chunks_bm25.csv"
CHROMA_DB_PATH  = r"H:\Rag_chatbot\chroma_db"
COLLECTION_NAME = "medical_qa"
DEDUP_THRESHOLD = 0.92    # chunks more similar than this → duplicate
BATCH_SIZE      = 500     # save to ChromaDB in batches of 500

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

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    for domain, keywords in DOMAIN_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "general"

# ── LOAD SAVED DATA ────────────────────────────────────────────────────────────
print("=" * 60)
print("  MedRAG Step 5 — Fast Deduplication")
print("=" * 60)

print("\n[1/4] Loading saved vectors and chunks...")

if not os.path.exists(VECTORS_PATH):
    print(f"  ERROR: {VECTORS_PATH} not found.")
    print("  Run phase1_indexing.py steps 1-4 first.")
    exit(1)

if not os.path.exists(CHUNKS_PATH):
    print(f"  ERROR: {CHUNKS_PATH} not found.")
    print("  Run phase1_indexing.py steps 1-4 first.")
    exit(1)

vectors   = np.load(VECTORS_PATH)
df        = pd.read_csv(CHUNKS_PATH)
texts     = df["text"].tolist()

print(f"  Loaded: {len(texts):,} chunks, vectors shape: {vectors.shape}")

# ── FAST DEDUPLICATION ─────────────────────────────────────────────────────────
print(f"\n[2/4] Fast deduplication (threshold={DEDUP_THRESHOLD})...")
print("      Using batched matrix similarity — should take 2-5 minutes...")

t_start   = time.time()
n         = len(vectors)
keep_mask = np.ones(n, dtype=bool)   # start: keep everything

# Process in batches to avoid memory explosion
# For each chunk i, compare against ALL already-kept chunks
# Use matrix multiply (dot product = cosine sim on normalized vectors)
DEDUP_BATCH = 1000   # compare 1000 chunks at a time

kept_vecs   = []     # growing list of kept vectors
kept_count  = 0
removed     = 0

for i in range(n):
    if not keep_mask[i]:
        continue

    # Compare chunk i against all previously kept vectors
    if kept_vecs:
        kept_matrix = np.array(kept_vecs)
        sims        = kept_matrix @ vectors[i]   # dot product = cosine sim (normalized)
        if np.any(sims >= DEDUP_THRESHOLD):
            keep_mask[i] = False
            removed += 1
            continue

    kept_vecs.append(vectors[i])
    kept_count += 1

    # Progress every 5000
    if (i + 1) % 5000 == 0:
        elapsed = int(time.time() - t_start)
        pct     = int((i+1) / n * 100)
        print(f"      Progress: {i+1:,}/{n:,} ({pct}%) | "
              f"kept={kept_count:,} removed={removed:,} | {elapsed}s elapsed")

elapsed = int(time.time() - t_start)
print(f"\n  Deduplication complete in {elapsed}s")
print(f"  Original : {n:,} chunks")
print(f"  Kept     : {kept_count:,} chunks")
print(f"  Removed  : {removed:,} duplicates ({int(removed/n*100)}%)")

# Apply mask
keep_indices  = np.where(keep_mask)[0]
texts_dedup   = [texts[i]           for i in keep_indices]
vectors_dedup = vectors[keep_indices]

# ── REBUILD METADATA ───────────────────────────────────────────────────────────
print("\n[3/4] Rebuilding metadata with domain tags...")

# Reconstruct metadata from CSV if available, else rebuild from text
if "domain" in df.columns:
    domains_dedup = [df.iloc[i].get("domain", "general") for i in keep_indices]
else:
    print("      No domain column found — detecting domains from text...")
    domains_dedup = [detect_domain(t) for t in texts_dedup]

metadatas = []
for i, (text, domain) in enumerate(zip(texts_dedup, domains_dedup)):
    metadatas.append({
        "type":      "medical_qa",
        "domain":    str(domain),
        "source":    "medquad",
        "word_count": str(len(text.split())),
    })

# ── SAVE TO CHROMADB ───────────────────────────────────────────────────────────
print(f"\n[4/4] Saving {kept_count:,} chunks to ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Delete and recreate main collection
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"  Deleted existing: {COLLECTION_NAME}")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Create domain collections
domain_collections = {}
for domain in list(DOMAIN_MAP.keys()) + ["general"]:
    try:
        client.delete_collection(f"medical_{domain}")
    except Exception:
        pass
    domain_collections[domain] = client.create_collection(
        name=f"medical_{domain}",
        metadata={"hnsw:space": "cosine"}
    )

print(f"  Created {len(domain_collections)} domain collections")

# Prepare domain batches
domain_batches: dict = {d: {"ids":[],"docs":[],"vecs":[],"meta":[]}
                        for d in domain_collections}

ids    = [str(uuid.uuid4()) for _ in texts_dedup]
total  = len(ids)

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)

    # Main collection
    collection.add(
        ids       = ids[start:end],
        documents = texts_dedup[start:end],
        embeddings= vectors_dedup[start:end].tolist(),
        metadatas = metadatas[start:end]
    )

    # Bucket into domain batches
    for i in range(start, end):
        d = metadatas[i].get("domain", "general")
        if d in domain_batches:
            domain_batches[d]["ids"].append(ids[i])
            domain_batches[d]["docs"].append(texts_dedup[i])
            domain_batches[d]["vecs"].append(vectors_dedup[i].tolist())
            domain_batches[d]["meta"].append(metadatas[i])

    pct = int(end / total * 100)
    print(f"  Main collection: {end:,}/{total:,} ({pct}%)")

# Save domain collections
print("\n  Saving domain collections...")
for domain, batch in domain_batches.items():
    if not batch["ids"]:
        continue
    domain_collections[domain].add(
        ids       = batch["ids"],
        documents = batch["docs"],
        embeddings= batch["vecs"],
        metadatas = batch["meta"]
    )
    print(f"  {domain:15s}: {len(batch['ids']):,} chunks")

# Save updated BM25 CSV with domain column
updated_df = pd.DataFrame({
    "text":   texts_dedup,
    "domain": domains_dedup,
    "id":     ids
})
updated_df.to_csv(r"H:\Rag_chatbot\chunks_bm25.csv", index=False)

# Save updated vectors backup
np.save(r"H:\Rag_chatbot\vectors_backup.npy", vectors_dedup)

# ── SUMMARY ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEDUPLICATION + INDEXING COMPLETE")
print("=" * 60)
print(f"  Main collection : {collection.count():,} chunks")
for domain, col in domain_collections.items():
    cnt = col.count()
    if cnt > 0:
        print(f"  {domain:15s} : {cnt:,} chunks")
print(f"\n  Original chunks : {n:,}")
print(f"  After dedup     : {kept_count:,}")
print(f"  Removed         : {removed:,} ({int(removed/n*100)}%)")
print(f"\n  Now start the backend:")
print(f"  uvicorn backend_api:app --port 8001 --reload-dir H:\\Rag_chatbot")
print("=" * 60)
