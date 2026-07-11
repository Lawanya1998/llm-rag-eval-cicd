"""
gate.py — the CI/CD quality gate.

Reads results.json (this run) and history.jsonl (past runs) and decides
whether the change is allowed to merge. Exits 0 = PASS, 1 = FAIL.
A non-zero exit is what makes a GitHub Actions check go red and blocks a merge.

Two kinds of check:
  1. FIXED thresholds  - absolute quality bars that must always hold.
  2. BASELINE regression - faithfulness must not drop much vs the previous run.
"""

import os
import sys
import json

# ---- Fixed thresholds (override via env in CI if you like) ----
MAX_HALLUCINATION_RATE = float(os.environ.get("MAX_HALLUCINATION_RATE", "0.05"))  # >5% fails
MIN_FAITHFULNESS       = float(os.environ.get("MIN_FAITHFULNESS", "4.0"))
MIN_RELEVANCY          = float(os.environ.get("MIN_RELEVANCY", "4.0"))
MAX_LATENCY_P95        = float(os.environ.get("MAX_LATENCY_P95", "5.0"))          # seconds

# ---- Baseline regression allowance ----
MAX_FAITHFULNESS_DROP  = float(os.environ.get("MAX_FAITHFULNESS_DROP", "0.5"))


def load_current():
    with open("results.json", encoding="utf-8") as f:
        return json.load(f)["summary"]


def load_baseline():
    """Previous run = the second-to-last line of history.jsonl, if present."""
    if not os.path.exists("history.jsonl"):
        return None
    with open("history.jsonl", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[-2] if len(lines) >= 2 else None


def main():
    cur = load_current()
    base = load_baseline()
    failures = []

    # ---- fixed threshold checks ----
    if cur["hallucination_rate"] > MAX_HALLUCINATION_RATE:
        failures.append(
            f"hallucination_rate {cur['hallucination_rate']:.3f} > {MAX_HALLUCINATION_RATE}"
        )
    if cur["faithfulness_avg"] is not None and cur["faithfulness_avg"] < MIN_FAITHFULNESS:
        failures.append(
            f"faithfulness_avg {cur['faithfulness_avg']} < {MIN_FAITHFULNESS}"
        )
    if cur["relevancy_avg"] is not None and cur["relevancy_avg"] < MIN_RELEVANCY:
        failures.append(
            f"relevancy_avg {cur['relevancy_avg']} < {MIN_RELEVANCY}"
        )
    if cur["latency_p95"] > MAX_LATENCY_P95:
        failures.append(
            f"latency_p95 {cur['latency_p95']}s > {MAX_LATENCY_P95}s"
        )

    # ---- baseline regression check ----
    if base and base.get("faithfulness_avg") and cur.get("faithfulness_avg"):
        drop = base["faithfulness_avg"] - cur["faithfulness_avg"]
        if drop > MAX_FAITHFULNESS_DROP:
            failures.append(
                f"faithfulness regressed by {drop:.2f} vs baseline "
                f"({base['faithfulness_avg']} -> {cur['faithfulness_avg']})"
            )

    # ---- report ----
    print("=" * 40)
    print("QUALITY GATE")
    print("=" * 40)
    print(f"  hallucination_rate : {cur['hallucination_rate']}  (max {MAX_HALLUCINATION_RATE})")
    print(f"  faithfulness_avg   : {cur['faithfulness_avg']}  (min {MIN_FAITHFULNESS})")
    print(f"  relevancy_avg      : {cur['relevancy_avg']}  (min {MIN_RELEVANCY})")
    print(f"  latency_p95        : {cur['latency_p95']}s  (max {MAX_LATENCY_P95}s)")
    if base:
        print(f"  baseline faithful. : {base.get('faithfulness_avg')}")
    print("-" * 40)

    if failures:
        print("RESULT: FAIL - merge blocked")
        for msg in failures:
            print(f"   x {msg}")
        print("=" * 40)
        sys.exit(1)

    print("RESULT: PASS - all checks green")
    print("=" * 40)
    sys.exit(0)


if __name__ == "__main__":
    main()
