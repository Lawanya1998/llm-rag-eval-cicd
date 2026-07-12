"""
rag.py — the RAG pipeline we will evaluate (Groq version).

Flow:  question -> retrieve relevant doc chunks -> ask the model to answer
       USING ONLY those chunks -> return answer + sources + latency + cost.

Retrieval uses TF-IDF (lightweight, no model download, fully reproducible).
The Retriever class has a clean interface, so you can swap in vector embeddings
later without touching the rest of the pipeline.
"""

import os
import glob
import time
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Model IDs on Groq change over time — confirm the current list at
# https://console.groq.com/docs/models  and set MODEL in your environment.
MODEL = os.environ.get("MODEL", "llama-3.3-70b-versatile")

# Groq's free tier makes cost ~0, but we still track tokens so the metric exists.
# Update these per-million-token prices if you move to a paid model.
INPUT_PRICE_PER_MTOK = float(os.environ.get("INPUT_PRICE", "0.0"))
OUTPUT_PRICE_PER_MTOK = float(os.environ.get("OUTPUT_PRICE", "0.0"))

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are NimbusPay's support assistant.
Answer the user's question USING ONLY the provided context passages
Be concise. Do not invent facts, numbers, or policies."""


# ---------- Retrieval ----------

def _chunk(text, size=500, overlap=100):
    """Split a document into overlapping character chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


class Retriever:
    def __init__(self, kb_dir="knowledge_base"):
        self.chunks = []
        self.sources = []
        for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
            doc = os.path.basename(path)
            with open(path, encoding="utf-8") as f:
                for c in _chunk(f.read()):
                    self.chunks.append(c)
                    self.sources.append(doc)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query, k=3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], self.sources[i]) for i in top]


# ---------- Generation ----------

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def answer(question, k=3):
    """Run the full RAG pipeline for one question.

    Returns a dict the eval layer can score:
        answer, sources, contexts, latency_s, cost_usd, tokens
    """
    retriever = _get_retriever()
    retrieved = retriever.search(question, k=k)
    contexts = [c for c, _ in retrieved]
    sources = [s for _, s in retrieved]

    context_block = "\n\n".join(
        f"[Source: {s}]\n{c}" for c, s in retrieved
    )
    user_msg = f"Context passages:\n{context_block}\n\nQuestion: {question}"

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    latency = time.perf_counter() - t0

    text = resp.choices[0].message.content.strip()
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    cost = (in_tok / 1e6) * INPUT_PRICE_PER_MTOK + (out_tok / 1e6) * OUTPUT_PRICE_PER_MTOK

    return {
        "answer": text,
        "sources": sources,
        "contexts": contexts,
        "latency_s": round(latency, 3),
        "cost_usd": round(cost, 6),
        "tokens": {"input": in_tok, "output": out_tok},
    }


if __name__ == "__main__":
    for q in [
        "How long do I have to request a refund?",
        "What does it cost to send money to a bank account?",
        "Can I open two personal accounts?",
        "What is NimbusPay's stock price?",  # not in the KB -> should refuse
    ]:
        r = answer(q)
        print(f"\nQ: {q}")
        print(f"A: {r['answer']}")
        print(f"   sources={r['sources']} latency={r['latency_s']}s cost=${r['cost_usd']}")