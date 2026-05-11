"""
MedRAG — Evaluation Suite v4.0
================================
Metrics implemented:
  1. Recall@K        — did we retrieve ALL the chunks we needed?
  2. Precision@K     — were retrieved chunks actually relevant?
  3. MRR             — was the best chunk ranked first?
  4. Faithfulness    — does answer stay within retrieved context?
  5. Context Relevance — how similar are chunks to the question?
  6. Answer Relevance  — does the answer actually address the question?
  + Keyword Coverage   — do expected medical terms appear?
  + Grounding Score    — fraction of answer words grounded in context
  + IDK Accuracy       — correct out-of-scope refusals
  + Latency            — response time per question

Run AFTER backend is running on port 8001:
  python evaluate_rag.py
"""

import requests
import csv
import os
import re
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8001"
OUTPUT_DIR = r"H:\Rag_chatbot\eval_output"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # used for context/answer relevance
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TEST CASES ─────────────────────────────────────────────────────────────────
# keywords   = stems expected in answer (for coverage + Precision@K)
# rel_chunks = keywords that MUST appear in a retrieved chunk to count as relevant
#              (used for Recall@K and Precision@K)
# ground_truth = ideal one-sentence answer (used for Answer Relevance + Faithfulness)
TEST_CASES = [
    {
        "q":           "What are the symptoms of diabetes?",
        "keywords":    ["thirst", "urinat", "blood"],
        "rel_chunks":  ["diabetes", "symptom", "glucose"],
        "ground_truth":"Diabetes symptoms include increased thirst, frequent urination, fatigue, and high blood glucose levels.",
        "idk":         False, "category": "Diabetes"
    },
    {
        "q":           "How does insulin work in the body?",
        "keywords":    ["pancrea", "gluco", "insuli"],
        "rel_chunks":  ["insulin", "glucose", "pancreas"],
        "ground_truth":"Insulin is a hormone produced by the pancreas that helps glucose enter cells to be used as energy.",
        "idk":         False, "category": "Diabetes"
    },
    {
        "q":           "What are treatments for type 2 diabetes?",
        "keywords":    ["diet", "blood", "medic"],
        "rel_chunks":  ["diabetes", "treatment", "insulin"],
        "ground_truth":"Type 2 diabetes is managed through diet, exercise, blood sugar monitoring, and medications including metformin.",
        "idk":         False, "category": "Diabetes"
    },
    {
        "q":           "What are symptoms of high blood pressure?",
        "keywords":    ["press", "blood", "heart"],
        "rel_chunks":  ["hypertension", "blood pressure", "symptom"],
        "ground_truth":"High blood pressure often has no symptoms but can cause headaches, shortness of breath, and chest pain.",
        "idk":         False, "category": "Hypertension"
    },
    {
        "q":           "What causes hypertension?",
        "keywords":    ["blood", "press", "heart"],
        "rel_chunks":  ["hypertension", "blood pressure", "cause"],
        "ground_truth":"Hypertension is caused by factors including obesity, salt intake, stress, genetics, and lack of exercise.",
        "idk":         False, "category": "Hypertension"
    },
    {
        "q":           "How is asthma treated?",
        "keywords":    ["inhal", "airwa", "lung"],
        "rel_chunks":  ["asthma", "inhaler", "treatment"],
        "ground_truth":"Asthma is treated with inhalers including bronchodilators and corticosteroids to open airways and reduce inflammation.",
        "idk":         False, "category": "Asthma"
    },
    {
        "q":           "What causes asthma attacks?",
        "keywords":    ["trigg", "airwa", "breat"],
        "rel_chunks":  ["asthma", "trigger", "airway"],
        "ground_truth":"Asthma attacks are triggered by allergens, exercise, cold air, smoke, and respiratory infections.",
        "idk":         False, "category": "Asthma"
    },
    {
        "q":           "What is cholesterol?",
        "keywords":    ["fat", "blood", "liver"],
        "rel_chunks":  ["cholesterol", "blood", "fat"],
        "ground_truth":"Cholesterol is a fat-like substance in the blood produced by the liver, necessary for cell function but harmful in excess.",
        "idk":         False, "category": "Cardiology"
    },
    {
        "q":           "What are the symptoms of a heart attack?",
        "keywords":    ["chest", "pain", "heart"],
        "rel_chunks":  ["heart attack", "chest", "pain"],
        "ground_truth":"Heart attack symptoms include chest pain or pressure, shortness of breath, sweating, and pain radiating to the arm.",
        "idk":         False, "category": "Cardiology"
    },
    {
        "q":           "What is kidney disease?",
        "keywords":    ["kidne", "urin", "blood"],
        "rel_chunks":  ["kidney", "renal", "disease"],
        "ground_truth":"Kidney disease is a condition where the kidneys lose their ability to filter waste and excess fluid from the blood.",
        "idk":         False, "category": "Kidney"
    },
    {
        "q":           "How is cancer treated?",
        "keywords":    ["surge", "chemo", "tumor"],
        "rel_chunks":  ["cancer", "treatment", "chemotherapy"],
        "ground_truth":"Cancer treatment options include surgery to remove tumors, chemotherapy, radiation therapy, and targeted therapies.",
        "idk":         False, "category": "Cancer"
    },
    {
        "q":           "What are symptoms of pneumonia?",
        "keywords":    ["cough", "fever", "lung"],
        "rel_chunks":  ["pneumonia", "lung", "symptom"],
        "ground_truth":"Pneumonia symptoms include cough, fever, chills, shortness of breath, and chest pain.",
        "idk":         False, "category": "Respiratory"
    },
    # ── OUT-OF-SCOPE ──────────────────────────────────────────────────────────
    {"q":"What is the weather forecast for tomorrow?",
     "keywords":[], "rel_chunks":[], "ground_truth":"",
     "idk":True, "category":"Out-of-scope"},
    {"q":"What is the best smartphone to buy in 2024?",
     "keywords":[], "rel_chunks":[], "ground_truth":"",
     "idk":True, "category":"Out-of-scope"},
    {"q":"Who won the FIFA World Cup in 2022?",
     "keywords":[], "rel_chunks":[], "ground_truth":"",
     "idk":True, "category":"Out-of-scope"},
]

K = 3   # top-K chunks retrieved per question

# ── LOAD EMBEDDING MODEL (for semantic metrics) ────────────────────────────────
print("Loading embedding model for semantic metrics...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Model ready.\n")

# ── METRIC FUNCTIONS ──────────────────────────────────────────────────────────

def precision_at_k(sources: list, rel_keywords: list, k: int = K) -> float:
    """
    Precision@K = Relevant chunks retrieved / K
    A chunk is 'relevant' if it contains at least one relevance keyword.
    """
    if not rel_keywords or not sources:
        return 0.0
    top_k   = sources[:k]
    relevant = sum(
        1 for src in top_k
        if any(kw.lower() in (src.get("text","") if isinstance(src,dict)
               else getattr(src,"text","")).lower()
               for kw in rel_keywords)
    )
    return round(relevant / min(k, len(top_k)), 3)


def recall_at_k(sources: list, rel_keywords: list, k: int = K) -> float:
    """
    Recall@K = Relevant chunks retrieved / Total relevant chunks possible.
    We treat each rel_keyword as a distinct 'relevant fact needed'.
    Recall = fraction of required keywords found in ANY of the top-K chunks.
    """
    if not rel_keywords or not sources:
        return 0.0
    all_text = " ".join(
        (src.get("text","") if isinstance(src,dict) else getattr(src,"text","")).lower()
        for src in sources[:k]
    )
    found = sum(1 for kw in rel_keywords if kw.lower() in all_text)
    return round(found / len(rel_keywords), 3)


def mean_reciprocal_rank(sources: list, rel_keywords: list) -> float:
    """
    MRR = 1 / rank_of_first_relevant_chunk
    MRR=1.0  → first chunk is relevant (best)
    MRR=0.5  → second chunk is first relevant one
    MRR=0.33 → third chunk is first relevant one
    MRR=0.0  → no relevant chunk found
    """
    if not rel_keywords or not sources:
        return 0.0
    for rank, src in enumerate(sources, start=1):
        text = (src.get("text","") if isinstance(src,dict)
                else getattr(src,"text","")).lower()
        if any(kw.lower() in text for kw in rel_keywords):
            return round(1.0 / rank, 3)
    return 0.0


def context_relevance(question: str, sources: list) -> float:
    """
    Context Relevance = average cosine similarity between question
    embedding and each retrieved chunk embedding.
    Measures: did we retrieve chunks that are semantically close to the question?
    High score = chunks are on-topic.
    """
    if not sources:
        return 0.0
    q_vec = embedder.encode(
        f"Represent this sentence for searching relevant passages: {question}",
        normalize_embeddings=True
    )
    chunk_texts = [
        (src.get("text","") if isinstance(src,dict) else getattr(src,"text",""))
        for src in sources
    ]
    if not chunk_texts:
        return 0.0
    chunk_vecs = embedder.encode(chunk_texts, normalize_embeddings=True)
    sims       = cosine_similarity([q_vec], chunk_vecs)[0]
    return round(float(sims.mean()), 3)


def answer_relevance(question: str, answer: str) -> float:
    """
    Answer Relevance = cosine similarity between question embedding
    and answer embedding.
    Measures: does the answer actually address the question asked?
    High score = answer is on-topic for the question.
    """
    if not answer or not question:
        return 0.0
    q_vec = embedder.encode(question, normalize_embeddings=True)
    a_vec = embedder.encode(answer,   normalize_embeddings=True)
    sim   = float(cosine_similarity([q_vec], [a_vec])[0][0])
    return round(sim, 3)


def faithfulness(answer: str, sources: list) -> float:
    """
    Faithfulness = fraction of answer content words that appear
    in the retrieved context (stem-based matching).
    High = answer stays within retrieved facts (low hallucination).
    Low  = answer contains words not in any retrieved chunk (hallucination risk).
    """
    if not sources or not answer:
        return 0.0

    GRAM_WORDS = {
        "with","that","this","from","also","when","have","been","will",
        "some","more","than","they","their","help","make","body","your",
        "lead","caus","incl","such","most","many","each","both","into",
        "over","after","before","about","there","these","those","often",
        "which","while","where","would","could","should","other","very",
        "well","main","type","form","used","made","take","give","keep",
        "high","leve","cell","tiss","orga","syst"
    }

    context_words = set()
    context_stems = set()
    for src in sources:
        text = (src.get("text","") if isinstance(src,dict)
                else getattr(src,"text",""))
        for word in re.sub(r"[^\w\s]","",text).lower().split():
            if len(word) > 3:
                context_words.add(word)
                context_stems.add(word[:5])

    answer_content = [
        w for w in re.sub(r"[^\w\s]","",answer).lower().split()
        if len(w) > 4 and w[:5] not in GRAM_WORDS
    ]
    if not answer_content:
        return 0.75

    grounded = sum(
        1 for w in answer_content
        if w in context_words or w[:5] in context_stems
    )
    return round(grounded / len(answer_content), 3)


def keyword_coverage(answer: str, keywords: list) -> float:
    """Stem-based keyword match — are expected medical terms in the answer?"""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw in answer_lower)
    return round(hits / len(keywords), 3)


def status_label(val, good, ok):
    return "✓ Good" if val >= good else "~ Acceptable" if val >= ok else "✗ Needs improvement"

# ── CHECK API ──────────────────────────────────────────────────────────────────
print("=" * 62)
print("  MedRAG Evaluation Suite v4.0")
print(f"  API: {API_BASE}  |  Questions: {len(TEST_CASES)}")
print(f"  Metrics: Precision@{K}, Recall@{K}, MRR, Faithfulness,")
print(f"           Context Relevance, Answer Relevance + Coverage")
print("=" * 62)

try:
    health = requests.get(f"{API_BASE}/health", timeout=5).json()
    print(f"\n  ✓ API online | {health.get('chunk_count',0):,} chunks | "
          f"LLM: {health.get('llm_model')} | "
          f"num_ctx: {health.get('num_ctx','?')}")
except Exception:
    print("  ✗ Cannot reach API. Start backend first:")
    print("    uvicorn backend_api:app --port 8001 --reload-dir H:\\Rag_chatbot")
    exit(1)

# ── RUN QUESTIONS ──────────────────────────────────────────────────────────────
print("\nRunning evaluation...\n")

results = []
for i, case in enumerate(TEST_CASES, 1):
    print(f"  [{i:02d}/{len(TEST_CASES)}] {case['q'][:55]}...")
    try:
        r    = requests.post(f"{API_BASE}/chat",
                             json={"question": case["q"], "top_k": K},
                             timeout=120)
        data = r.json() if r.ok else {}
        if not r.ok:
            print(f"       !! API ERROR {r.status_code}: {r.text[:150]}")
    except Exception as e:
        print(f"       !! REQUEST FAILED: {e}")
        data = {}

    answer   = data.get("answer",   "")
    sources  = data.get("sources",  [])
    answered = data.get("answered", False)
    conf     = data.get("confidence","None")
    latency  = data.get("latency_ms", 0)
    top_score= max((s.get("score",0) if isinstance(s,dict) else getattr(s,"score",0)
                    for s in sources), default=0)

    idk_correct = (case["idk"] == (not answered))

    if case["idk"]:
        # Out-of-scope: only measure IDK accuracy
        results.append({
            "question":         case["q"],
            "category":         case["category"],
            "expected_idk":     True,
            "answered":         answered,
            "idk_correct":      idk_correct,
            "confidence":       conf,
            "top_score":        round(top_score, 4),
            "precision_k":      None,
            "recall_k":         None,
            "mrr":              None,
            "context_rel":      None,
            "answer_rel":       None,
            "faithfulness":     None,
            "keyword_cov":      None,
            "latency_ms":       latency,
            "answer":           answer[:200],
        })
        ok_str = "✓" if idk_correct else "✗"
        print(f"       {ok_str} IDK-expected | latency={latency}ms")
        continue

    # In-scope: compute all metrics
    prec  = precision_at_k(sources, case["rel_chunks"], K)
    rec   = recall_at_k(sources, case["rel_chunks"], K)
    mrr   = mean_reciprocal_rank(sources, case["rel_chunks"])
    ctx_r = context_relevance(case["q"], sources)
    ans_r = answer_relevance(case["q"], answer)
    faith = faithfulness(answer, sources)
    cov   = keyword_coverage(answer, case["keywords"])
    idc   = (case["idk"] == (not answered))

    results.append({
        "question":         case["q"],
        "category":         case["category"],
        "expected_idk":     False,
        "answered":         answered,
        "idk_correct":      idc,
        "confidence":       conf,
        "top_score":        round(top_score, 4),
        "precision_k":      prec,
        "recall_k":         rec,
        "mrr":              mrr,
        "context_rel":      ctx_r,
        "answer_rel":       ans_r,
        "faithfulness":     faith,
        "keyword_cov":      cov,
        "latency_ms":       latency,
        "answer":           answer[:200],
    })

    ok_str = "✓" if idc else "✗"
    print(f"       {ok_str} {conf:6s} | "
          f"P@{K}={prec:.0%} R@{K}={rec:.0%} MRR={mrr:.2f} | "
          f"Ctx={ctx_r:.2f} Ans={ans_r:.2f} Faith={faith:.2f} | "
          f"{latency}ms")

print("\nAll questions done.\n")

# ── CACHE TEST ────────────────────────────────────────────────────────────────
print("Testing cache with repeat questions...")
cache_hits = 0
for case in TEST_CASES[:3]:
    d   = requests.post(f"{API_BASE}/chat",
                        json={"question":case["q"],"top_k":K},
                        timeout=120).json()
    hit = d.get("cache_hit", False)
    ms  = d.get("latency_ms", 0)
    print(f"  {'[CACHE HIT]' if hit else '[MISS]':12s} {ms:5d}ms | {case['q'][:45]}")
    if hit:
        cache_hits += 1
print(f"\n  Cache: {cache_hits}/3 hits "
      f"{'✓ working' if cache_hits > 0 else '✗ not working'}\n")

# ── COMPUTE SUMMARY STATS ──────────────────────────────────────────────────────
df       = pd.DataFrame(results)
in_scope = df[df["expected_idk"] == False]
oos      = df[df["expected_idk"] == True]

total_qs        = len(df)
answered_count  = int(df["answered"].sum())
idk_correct_pct = round(df["idk_correct"].mean() * 100, 1)
oos_correct     = int(oos["idk_correct"].sum())

# Core retrieval metrics (in-scope only)
avg_precision   = round(in_scope["precision_k"].dropna().mean() * 100, 1)
avg_recall      = round(in_scope["recall_k"].dropna().mean() * 100, 1)
avg_mrr         = round(in_scope["mrr"].dropna().mean(), 3)
avg_ctx_rel     = round(in_scope["context_rel"].dropna().mean() * 100, 1)
avg_ans_rel     = round(in_scope["answer_rel"].dropna().mean() * 100, 1)
avg_faithfulness= round(in_scope["faithfulness"].dropna().mean() * 100, 1)
avg_coverage    = round(in_scope["keyword_cov"].dropna().mean() * 100, 1)
avg_score       = round(in_scope["top_score"].mean(), 3)
avg_latency     = round(df["latency_ms"].mean())

# ── SAVE CSV ───────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "eval_results.csv")
df.to_csv(csv_path, index=False)
print(f"CSV saved → {csv_path}")

# ── DASHBOARD IMAGE ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#0a0f0d","axes.facecolor":"#111815",
    "axes.edgecolor":"#2a3a32","axes.labelcolor":"#e8f0eb",
    "xtick.color":"#7a9186","ytick.color":"#7a9186",
    "text.color":"#e8f0eb","grid.color":"#1e2e24",
    "grid.linewidth":0.5,"font.family":"monospace","font.size":9,
})
GREEN="#4ac88c"; AMBER="#f59e0b"; RED="#e05555"; MUTED="#7a9186"; CARD="#161e1a"

fig = plt.figure(figsize=(18, 12))
fig.suptitle("MedRAG — Evaluation Dashboard v4.0",
             fontsize=16, fontweight="bold", color=GREEN, y=0.98)
gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.42,
                      left=0.06, right=0.97, top=0.92, bottom=0.06)

# ── Panel 1: Metric Scorecard ─────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off")
ax0.set_title("All Metrics Summary", color=GREEN, fontsize=10, pad=8)
all_metrics = [
    ("Precision@K",       avg_precision,    "%", 70, 50),
    ("Recall@K",          avg_recall,       "%", 70, 50),
    ("MRR",               avg_mrr*100,      "%", 70, 50),
    ("Context Relevance", avg_ctx_rel,      "%", 70, 55),
    ("Answer Relevance",  avg_ans_rel,      "%", 70, 55),
    ("Faithfulness",      avg_faithfulness, "%", 70, 55),
]
for idx, (name, val, unit, good, ok) in enumerate(all_metrics):
    y     = 0.88 - idx * 0.16
    color = GREEN if val >= good else AMBER if val >= ok else RED
    ax0.text(0.04, y,    name,              fontsize=7.5, color=MUTED)
    ax0.text(0.96, y-0.01, f"{val:.1f}{unit}", fontsize=12,
             color=color, ha="right", fontweight="bold")
    ax0.add_patch(mpatches.FancyBboxPatch(
        (0.03, y-0.05), 0.94, 0.004,
        boxstyle="round,pad=0.01", color=color, alpha=0.3
    ))

# ── Panel 2: Precision@K and Recall@K per question ───────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
x     = np.arange(len(in_scope))
width = 0.38
p_vals = in_scope["precision_k"].fillna(0).values
r_vals = in_scope["recall_k"].fillna(0).values
ax1.bar(x - width/2, p_vals, width, color=GREEN, alpha=0.85, label=f"Precision@{K}")
ax1.bar(x + width/2, r_vals, width, color=AMBER, alpha=0.85, label=f"Recall@{K}")
ax1.axhline(0.7, color=MUTED, linestyle="--", lw=0.8, alpha=0.5)
ax1.set_title(f"Precision@{K} vs Recall@{K}", color=GREEN, fontsize=10, pad=8)
ax1.set_xlabel("Question #", fontsize=8); ax1.set_ylabel("Score", fontsize=8)
ax1.set_ylim(0, 1.1); ax1.set_xticks(x); ax1.set_xticklabels(x+1, fontsize=7)
ax1.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED); ax1.grid(axis="y")

# ── Panel 3: MRR per question ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
mrr_vals = in_scope["mrr"].fillna(0).values
bc = [GREEN if v >= 0.7 else AMBER if v >= 0.5 else RED for v in mrr_vals]
ax2.bar(range(len(mrr_vals)), mrr_vals, color=bc, width=0.7, edgecolor="none")
ax2.axhline(1.0, color=GREEN, linestyle="--", lw=0.8, alpha=0.5, label="MRR=1.0 (ideal)")
ax2.axhline(0.5, color=AMBER, linestyle="--", lw=0.8, alpha=0.5, label="MRR=0.5 (rank 2)")
ax2.set_title("MRR — Ranking Quality", color=GREEN, fontsize=10, pad=8)
ax2.set_xlabel("Question #", fontsize=8); ax2.set_ylabel("MRR Score", fontsize=8)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED); ax2.grid(axis="y")

# ── Panel 4: Context Relevance vs Answer Relevance ────────────────────────────
ax3 = fig.add_subplot(gs[0, 3])
ctx_vals = in_scope["context_rel"].fillna(0).values * 100
ans_vals = in_scope["answer_rel"].fillna(0).values  * 100
x4       = np.arange(len(in_scope))
ax3.bar(x4 - 0.2, ctx_vals, 0.38, color="#7c3aed", alpha=0.85, label="Context Relevance")
ax3.bar(x4 + 0.2, ans_vals, 0.38, color="#0ea5e9", alpha=0.85, label="Answer Relevance")
ax3.axhline(70, color=MUTED, linestyle="--", lw=0.8, alpha=0.5)
ax3.set_title("Relevance Scores", color=GREEN, fontsize=10, pad=8)
ax3.set_xlabel("Question #", fontsize=8); ax3.set_ylabel("Score %", fontsize=8)
ax3.set_ylim(0, 115); ax3.set_xticks(x4); ax3.set_xticklabels(x4+1, fontsize=7)
ax3.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED); ax3.grid(axis="y")

# ── Panel 5: Faithfulness per question ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
faith_vals = in_scope["faithfulness"].fillna(0).values * 100
cats_short = [c[:12] for c in in_scope["category"].values]
fc = [GREEN if v >= 70 else AMBER if v >= 50 else RED for v in faith_vals]
hb = ax4.barh(cats_short, faith_vals, color=fc, edgecolor="none", height=0.6)
ax4.axvline(70, color=GREEN, linestyle="--", lw=0.8, alpha=0.6, label="Target 70%")
for bar, val in zip(hb, faith_vals):
    ax4.text(val+0.5, bar.get_y()+bar.get_height()/2,
             f"{val:.0f}%", va="center", fontsize=8, color=MUTED)
ax4.set_title("Faithfulness (Grounding) per Category", color=GREEN, fontsize=10, pad=8)
ax4.set_xlabel("% Answer Words in Context", fontsize=8)
ax4.set_xlim(0, 115); ax4.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED)
ax4.grid(axis="x")

# ── Panel 6: Chunk score per question ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
scores = in_scope["top_score"].values
sc = [GREEN if s >= 0.75 else AMBER if s >= 0.55 else RED for s in scores]
ax5.bar(range(len(scores)), scores, color=sc, width=0.7, edgecolor="none")
ax5.axhline(0.75, color=GREEN, linestyle="--", lw=0.8, alpha=0.6, label="High (0.75)")
ax5.axhline(0.55, color=AMBER, linestyle="--", lw=0.8, alpha=0.6, label="Gate (0.55)")
ax5.set_title("Top Chunk Score per Q", color=GREEN, fontsize=10, pad=8)
ax5.set_xlabel("Question #", fontsize=8); ax5.set_ylabel("Score", fontsize=8)
ax5.set_ylim(0, 1.1)
ax5.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED); ax5.grid(axis="y")

# ── Panel 7: Latency ──────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 3])
lats = df["latency_ms"].values / 1000
lc   = [GREEN if l < 10 else AMBER if l < 20 else RED for l in lats]
ax6.bar(range(len(lats)), lats, color=lc, width=0.7, edgecolor="none")
ax6.axhline(10, color=GREEN, linestyle="--", lw=0.8, alpha=0.6, label="Good <10s")
ax6.axhline(20, color=AMBER, linestyle="--", lw=0.8, alpha=0.6, label="Slow >20s")
ax6.set_title("Latency per Question", color=GREEN, fontsize=10, pad=8)
ax6.set_xlabel("Question #", fontsize=8); ax6.set_ylabel("Seconds", fontsize=8)
ax6.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED); ax6.grid(axis="y")

# ── Panel 8: Radar chart of all metrics ───────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0], polar=True)
radar_labels  = ["P@K", "R@K", "MRR", "Ctx\nRel", "Ans\nRel", "Faith"]
radar_values  = [avg_precision/100, avg_recall/100, avg_mrr,
                 avg_ctx_rel/100,   avg_ans_rel/100, avg_faithfulness/100]
angles  = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
values  = radar_values + [radar_values[0]]
angles += [angles[0]]
ax7.plot(angles, values, color=GREEN, linewidth=2)
ax7.fill(angles, values, color=GREEN, alpha=0.2)
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(radar_labels, fontsize=8, color="#e8f0eb")
ax7.set_ylim(0, 1)
ax7.set_yticks([0.25, 0.5, 0.75, 1.0])
ax7.set_yticklabels(["25%","50%","75%","100%"], fontsize=6, color=MUTED)
ax7.tick_params(colors=MUTED)
ax7.spines["polar"].set_color("#2a3a32")
ax7.set_facecolor("#111815")
ax7.set_title("Metric Radar", color=GREEN, fontsize=10, pad=20)

# ── Panel 9: IDK accuracy ─────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
ax8.bar(["Correct IDK","Missed"],
        [oos_correct, len(oos)-oos_correct],
        color=[GREEN, RED], width=0.5, edgecolor="none")
for idx2, v in enumerate([oos_correct, len(oos)-oos_correct]):
    ax8.text(idx2, v+0.05, str(v), ha="center",
             fontsize=12, fontweight="bold", color=[GREEN,RED][idx2])
ax8.set_title("Out-of-Scope Detection\n(IDK Accuracy)", color=GREEN, fontsize=10, pad=8)
ax8.set_ylim(0, max(oos_correct, len(oos)-oos_correct)*1.6+1)
ax8.grid(axis="y")

# ── Panel 10: Confidence distribution ────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
cc  = df["confidence"].value_counts()
pc  = {"High":GREEN,"Medium":AMBER,"None":RED,"Low":MUTED}
ws, ts, ats = ax9.pie(
    cc.values, labels=cc.index,
    colors=[pc.get(k,MUTED) for k in cc.index],
    autopct="%1.0f%%", startangle=90,
    wedgeprops={"edgecolor":"#0a0f0d","linewidth":2}
)
for t in ts+ats: t.set_color("#e8f0eb"); t.set_fontsize(8)
ax9.set_title("Confidence Distribution", color=GREEN, fontsize=10, pad=8)

# ── Panel 11: Summary text ────────────────────────────────────────────────────
ax10 = fig.add_subplot(gs[2, 3]); ax10.axis("off")
ax10.set_title("Summary", color=GREEN, fontsize=10, pad=8)
lines = [
    f"Precision@{K}:     {avg_precision:.1f}%",
    f"Recall@{K}:        {avg_recall:.1f}%",
    f"MRR:          {avg_mrr:.3f}",
    f"Ctx Relevance:{avg_ctx_rel:.1f}%",
    f"Ans Relevance:{avg_ans_rel:.1f}%",
    f"Faithfulness: {avg_faithfulness:.1f}%",
    f"IDK Accuracy: {idk_correct_pct:.1f}%",
    f"Chunk Score:  {avg_score:.3f}",
    f"Avg Latency:  {avg_latency}ms",
    "",
    f"Model: {health.get('llm_model','-')}",
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
]
for j, line in enumerate(lines):
    ax10.text(0.03, 0.97 - j*0.082, line,
              fontsize=8, color=GREEN if j < 9 else MUTED,
              transform=ax10.transAxes, fontfamily="monospace")

img_path = os.path.join(OUTPUT_DIR, "eval_report.png")
plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor="#0a0f0d")
plt.close()
print(f"Dashboard saved → {img_path}")

# ── TEXT SUMMARY ──────────────────────────────────────────────────────────────
txt_path = os.path.join(OUTPUT_DIR, "eval_summary.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 62 + "\n")
    f.write("  MedRAG Evaluation Summary v4.0\n")
    f.write(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 62 + "\n\n")
    f.write("METRIC RESULTS\n" + "-" * 42 + "\n")
    f.write(f"  Precision@{K}       : {avg_precision:.1f}%   (relevant chunks retrieved / K)\n")
    f.write(f"  Recall@{K}          : {avg_recall:.1f}%   (needed facts found in top-K)\n")
    f.write(f"  MRR               : {avg_mrr:.3f}  (1/rank of first relevant chunk)\n")
    f.write(f"  Context Relevance : {avg_ctx_rel:.1f}%   (chunk similarity to question)\n")
    f.write(f"  Answer Relevance  : {avg_ans_rel:.1f}%   (answer similarity to question)\n")
    f.write(f"  Faithfulness      : {avg_faithfulness:.1f}%   (answer grounded in context)\n")
    f.write(f"  Keyword Coverage  : {avg_coverage:.1f}%   (expected terms in answer)\n")
    f.write(f"  IDK Accuracy      : {idk_correct_pct:.1f}%   (correct OOS refusals)\n")
    f.write(f"  Avg Chunk Score   : {avg_score:.3f}  (vector similarity)\n")
    f.write(f"  Avg Latency       : {avg_latency}ms\n\n")
    f.write("TARGETS\n" + "-" * 42 + "\n")
    f.write(f"  Precision@K       >= 70% = Good\n")
    f.write(f"  Recall@K          >= 70% = Good\n")
    f.write(f"  MRR               >= 0.70 = Good (best chunk at rank 1-2)\n")
    f.write(f"  Context Relevance >= 70% = Chunks on-topic\n")
    f.write(f"  Answer Relevance  >= 70% = Answer on-topic\n")
    f.write(f"  Faithfulness      >= 70% = Low hallucination\n")
    f.write(f"  IDK Accuracy      >= 80% = Good OOS detection\n\n")
    f.write("PER-QUESTION RESULTS\n" + "-" * 42 + "\n")
    for r in results:
        ok  = "✓" if r["idk_correct"] else "✗"
        if r["expected_idk"]:
            f.write(f"  {ok} [IDK ] {r['question'][:50]}\n")
        else:
            p = f"{r['precision_k']:.0%}" if r['precision_k'] is not None else "-"
            rc= f"{r['recall_k']:.0%}"    if r['recall_k']   is not None else "-"
            m = f"{r['mrr']:.2f}"         if r['mrr']        is not None else "-"
            fth=f"{r['faithfulness']:.0%}"if r['faithfulness']is not None else "-"
            f.write(f"  {ok} [{r['confidence']:6s}] "
                    f"P@K={p} R@K={rc} MRR={m} Faith={fth} "
                    f"{r['latency_ms']:5d}ms | {r['question'][:40]}\n")

print(f"Summary saved  → {txt_path}")

# ── FINAL CONSOLE ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  EVALUATION COMPLETE")
print("=" * 62)
print(f"  Precision@{K}       : {avg_precision:.1f}%   {status_label(avg_precision,70,50)}")
print(f"  Recall@{K}          : {avg_recall:.1f}%   {status_label(avg_recall,70,50)}")
print(f"  MRR               : {avg_mrr:.3f}  {status_label(avg_mrr*100,70,50)}")
print(f"  Context Relevance : {avg_ctx_rel:.1f}%   {status_label(avg_ctx_rel,70,55)}")
print(f"  Answer Relevance  : {avg_ans_rel:.1f}%   {status_label(avg_ans_rel,70,55)}")
print(f"  Faithfulness      : {avg_faithfulness:.1f}%   {status_label(avg_faithfulness,70,55)}")
print(f"  IDK Accuracy      : {idk_correct_pct:.1f}%   {status_label(idk_correct_pct,80,60)}")
print(f"  Avg Chunk Score   : {avg_score:.3f}  {status_label(avg_score*100,65,55)}")
print(f"  Avg Latency       : {avg_latency}ms  "
      f"{'✓ Good' if avg_latency<10000 else '~ Acceptable' if avg_latency<20000 else '✗ Slow'}")
print("=" * 62)
print(f"\n  Files saved to: {OUTPUT_DIR}")
print(f"  → eval_report.png   (11-panel dashboard)")
print(f"  → eval_results.csv  (full data table)")
print(f"  → eval_summary.txt  (text report)")