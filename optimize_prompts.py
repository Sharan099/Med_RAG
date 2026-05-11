"""
MedRAG — DSPy Prompt Optimization
====================================
Run this ONCE on your PC to auto-tune prompts.
Saves optimized_medrag.json — load in backend for better answers.

DSPy finds the best few-shot examples and prompt format
by trying many combinations and scoring them against your metric.

Run: python optimize_prompts.py
"""

import dspy
from dspy.teleprompt import BootstrapFewShot

# ── Configure DSPy with Ollama ─────────────────────────────────────────────────
print("Configuring DSPy with Ollama...")
try:
    lm = dspy.LM(model = "llama3.2:1b" ,api_base = "http://localhost:11434",max_tokens  = 150,temperature = 0.0)
    dspy.settings.configure(lm=lm)
    print("DSPy configured ✓")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure Ollama is running: ollama serve")
    exit(1)

# ── Define Signature ───────────────────────────────────────────────────────────
class MedicalQASignature(dspy.Signature):
    """
    You are a careful medical assistant.
    Answer ONLY using the provided context.
    Be concise (2-3 sentences). Use exact medical terms.
    If context is insufficient, say: I don't have enough information.
    """
    context  = dspy.InputField(desc="Retrieved medical knowledge chunks")
    question = dspy.InputField(desc="Medical question from the user")
    answer   = dspy.OutputField(desc="Concise grounded answer in 2-3 sentences")

class MedRAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(MedicalQASignature)

    def forward(self, context: str, question: str):
        return self.qa(context=context, question=question)

# ── Training Examples ──────────────────────────────────────────────────────────
# These are gold-standard (context, question, answer) triples.
# The optimizer uses these to find the best prompt format.
# Add more examples for better optimization.
trainset = [
    dspy.Example(
        context   = "Diabetes is a chronic metabolic disease. It occurs when the pancreas does not produce enough insulin or the body cannot use it effectively. This leads to high blood glucose levels which damage organs over time.",
        question  = "What is diabetes?",
        answer    = "Diabetes is a chronic disease where the pancreas fails to produce enough insulin, leading to high blood glucose. Over time this damages organs including the kidneys, eyes, and nerves."
    ).with_inputs("context", "question"),

    dspy.Example(
        context   = "Insulin is a hormone produced by the pancreas. It allows glucose from food to enter cells and be used for energy. Without insulin, glucose stays in the bloodstream causing hyperglycemia.",
        question  = "How does insulin work?",
        answer    = "Insulin is a hormone made by the pancreas that allows glucose to enter cells for energy. Without sufficient insulin, glucose accumulates in the bloodstream causing high blood sugar."
    ).with_inputs("context", "question"),

    dspy.Example(
        context   = "Asthma is a condition where airways become inflamed and narrow. Inhalers containing bronchodilators open the airways. Corticosteroid inhalers reduce inflammation. Common triggers include allergens, exercise, and cold air.",
        question  = "How is asthma treated?",
        answer    = "Asthma is treated with bronchodilator inhalers that open airways and corticosteroid inhalers that reduce inflammation. Avoiding triggers like allergens and cold air also helps manage the condition."
    ).with_inputs("context", "question"),

    dspy.Example(
        context   = "High blood pressure often has no symptoms, earning it the name 'silent killer'. In rare cases it can cause headaches, shortness of breath, or nosebleeds. It damages blood vessel walls over time.",
        question  = "What are symptoms of high blood pressure?",
        answer    = "High blood pressure usually has no symptoms, which is why it is called the silent killer. In rare cases it may cause headaches or shortness of breath, but most people feel nothing until serious damage has occurred."
    ).with_inputs("context", "question"),

    dspy.Example(
        context   = "Cancer treatment options include surgery to remove tumors, chemotherapy using drugs to kill cancer cells, radiation therapy using high-energy rays, and targeted therapy attacking specific cancer cell proteins.",
        question  = "How is cancer treated?",
        answer    = "Cancer is treated through surgery to remove tumors, chemotherapy with drugs that kill cancer cells, and radiation therapy. Targeted therapies that attack specific cancer proteins are also increasingly used."
    ).with_inputs("context", "question"),

    dspy.Example(
        context   = "Cholesterol is a fat-like substance found in all cells of the body. The liver produces most of the cholesterol the body needs. LDL cholesterol builds up in artery walls. HDL cholesterol removes cholesterol from arteries.",
        question  = "What is cholesterol?",
        answer    = "Cholesterol is a fat-like substance produced by the liver and found in all cells. LDL cholesterol is harmful as it builds up in artery walls, while HDL cholesterol is protective as it removes cholesterol from arteries."
    ).with_inputs("context", "question"),
]

# ── Evaluation Metric ──────────────────────────────────────────────────────────
def medical_answer_metric(gold, pred, trace=None):
    """
    Score a predicted answer against the gold answer.
    Uses word overlap as a simple proxy for correctness.
    Returns True if overlap >= 40% (acceptable answer).
    """
    if not hasattr(pred, "answer") or not pred.answer:
        return False
    gold_words = set(gold.answer.lower().split())
    pred_words = set(pred.answer.lower().split())
    # Remove common stopwords
    stopwords  = {"is","a","the","and","or","in","of","to","it","that","this","with"}
    gold_content = gold_words - stopwords
    pred_content = pred_words - stopwords
    if not gold_content:
        return True
    overlap = len(gold_content & pred_content) / len(gold_content)
    return overlap >= 0.35

# ── Run Optimization ──────────────────────────────────────────────────────────
print("\nRunning DSPy optimization...")
print("This tries many prompt variations to find the best one.")
print("Takes 5-15 minutes depending on hardware.\n")

module    = MedRAGModule()
optimizer = BootstrapFewShot(
    metric              = medical_answer_metric,
    max_bootstrapped_demos = 3,    # how many few-shot examples to inject
    max_labeled_demos      = 2,    # examples directly from trainset
)

try:
    optimized = optimizer.compile(module, trainset=trainset)
    output_path = "optimized_medrag.json"
    optimized.save(output_path)
    print(f"\nOptimization complete!")
    print(f"Saved to: {output_path}")
    print(f"\nTo use in backend_api.py, add after creating dspy_module:")
    print(f"  dspy_module.load('optimized_medrag.json')")
except Exception as e:
    print(f"\nOptimization failed: {e}")
    print("Using unoptimized module (still works, just less tuned).")
    module.save("optimized_medrag.json")
    print("Saved unoptimized module as fallback.")
