"""
eval.py — run the RAG pipeline over the golden set and score it.

Metrics produced:
  - faithfulness      : does the answer stay true to retrieved context? (LLM judge)
  - answer_relevancy  : does the answer address the question? (LLM judge)
  - hallucination_rate: on UNANSWERABLE questions, did it wrongly invent an answer?
  - latency p50 / p95 : from measured request times
  - cost per query    : from measured token usage

Writes:
  - results.json      : full run summary (used by the CI gate)
  - history.jsonl     : one summary line appended per run (used by the dashboard)
"""

import os
import json
import time
import statistics
from datetime import datetime, timezone

from groq import Groq
from rag import answer

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

REFUSAL = "i don't have that information"


# ---------- LLM-as-judge ----------

JUDGE_PROMPT = """You are a strict evaluator of a RAG system's answer.
You are given the CONTEXT the system retrieved, the QUESTION, and the ANSWER.

Score two things from 1 to 5:
- faithfulness: 5 = every claim in the answer is supported by the context;
  1 = the answer contains claims not found in the context (hallucination).
- relevancy: 5 = the answer directly addresses the question;
  1 = the answer is off-topic or does not address it.

Respond with ONLY a JSON object, no other text:
{"faithfulness": <1-5>, "relevancy": <1-5>}"""


def judge(question, context, ans):
    user = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {ans}"
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=60,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # be forgiving if the model wraps JSON in text
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1])
    return int(data["faithfulness"]), int(data["relevancy"])


# ---------- Runner ----------

def load_golden(path="golden.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def run():
    golden = load_golden()
    per_case = []
    latencies, costs = [], []
    faith_scores, rel_scores = [], []
    hallucinated = 0
    unanswerable = 0

    for i, case in enumerate(golden, 1):
        q = case["question"]
        result = answer(q)
        ans = result["answer"]
        refused = REFUSAL in ans.lower()

        latencies.append(result["latency_s"])
        costs.append(result["cost_usd"])

        if case["answerable"]:
            # judge faithfulness + relevancy only on answerable questions
            context = "\n\n".join(result["contexts"])
            f_score, r_score = judge(q, context, ans)
            faith_scores.append(f_score)
            rel_scores.append(r_score)
            hallo = False
        else:
            unanswerable += 1
            # a good system refuses; if it did NOT refuse, it hallucinated
            hallo = not refused
            if hallo:
                hallucinated += 1
            f_score = r_score = None

        per_case.append({
            "question": q,
            "answer": ans,
            "answerable": case["answerable"],
            "faithfulness": f_score,
            "relevancy": r_score,
            "hallucinated": hallo,
            "latency_s": result["latency_s"],
            "cost_usd": result["cost_usd"],
        })
        print(f"[{i}/{len(golden)}] {'REFUSED' if refused else 'answered'} :: {q[:55]}")

    def avg(xs):
        return round(statistics.mean(xs), 3) if xs else None

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": os.environ.get("MODEL", "llama-3.3-70b-versatile"),
        "n_cases": len(golden),
        "faithfulness_avg": avg(faith_scores),
        "relevancy_avg": avg(rel_scores),
        "hallucination_rate": round(hallucinated / unanswerable, 3) if unanswerable else 0.0,
        "latency_p50": round(percentile(latencies, 50), 3),
        "latency_p95": round(percentile(latencies, 95), 3),
        "avg_cost_per_query": round(statistics.mean(costs), 6) if costs else 0.0,
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": per_case}, f, indent=2)
    with open("history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    print("\n===== EVAL SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    print("========================")
    print("Wrote results.json and appended to history.jsonl")
    return summary


if __name__ == "__main__":
    run()