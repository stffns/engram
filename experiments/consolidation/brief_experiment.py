"""Brief-based consolidation experiment.

Tests whether LLM-generated temporal briefs (injected directly, not
retrieved) improve answer accuracy vs raw retrieval.

The hypothesis: consolidation's value is not in producing better
retrieval targets, but in producing reasoning artifacts that the LLM
can consume directly. A temporal brief like "cache: Redis -> TTL tuning
-> reverted to Caffeine (current)" gives the LLM structured context
that raw retrieved chunks don't.

Configs:
  1. retrieval-only   -- K=5 retrieved chunks as context
  2. briefs-only      -- LLM-generated topic briefs as context (no retrieval)
  3. briefs+retrieval -- briefs prepended, then K=5 retrieved chunks
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import vstash

_ENCODER = tiktoken.get_encoding("cl100k_base")
DEFAULT_MODEL = "gemini-2.0-flash"


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class Result:
    config: str
    question: str
    expected: list[str]
    predicted: str
    correct: bool
    context_tokens: int
    brief_tokens: int


def load_scenario(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.calls = 0

    def generate(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt,
        )
        self.calls += 1
        return (resp.text or "").strip()


def generate_briefs(events: list[dict], client: GeminiClient) -> str:
    """Ask LLM to produce temporal briefs from raw events."""
    event_texts = "\n".join(f"- {e['text']}" for e in events)
    prompt = (
        "You are analyzing a stream of engineering notes from a software team.\n"
        "Many of these are noise (standups, tickets, planning). Some are significant\n"
        "architectural decisions that evolved over time.\n\n"
        "Your job: identify the significant decision topics and for each one,\n"
        "produce a TEMPORAL BRIEF showing how the decision evolved.\n\n"
        "Format each brief as:\n"
        "## [Topic Name]\n"
        "- [date/version 1]: [what was decided and why]\n"
        "- [date/version 2]: [what changed and why]\n"
        "- [date/version 3]: [what changed and why]\n"
        "- **Current state:** [what is in place RIGHT NOW]\n\n"
        "Only include topics where you see clear decision evolution.\n"
        "Ignore noise (standups, tickets, planning sessions).\n\n"
        f"## Event stream\n{event_texts}"
    )
    return client.generate(prompt)


def answer_question(context: str, question: str, client: GeminiClient) -> str:
    prompt = (
        "Based on the context below, answer the question concisely and factually.\n"
        "If the context contains decision evolution, answer with the CURRENT state.\n\n"
        f"## Context\n{context}\n\n"
        f"## Question\n{question}\n\n"
        "## Answer"
    )
    return client.generate(prompt)


def check_answer(predicted: str, expect_contains: list[str]) -> bool:
    lower = predicted.lower()
    return any(kw.lower() in lower for kw in expect_contains)


def run_retrieval_only(
    events: list[dict],
    queries: list[dict],
    client: GeminiClient,
    *,
    top_k: int = 5,
) -> list[Result]:
    """Config 1: retrieve K chunks, answer from them."""
    results = []
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "retrieval.db"
        m = vstash.Memory(project="brief_exp", db=db, collection="default")
        for e in events:
            m.remember(e["text"], title=e["id"], collection="default")

        for q in queries:
            hits = m.search(q["question"], top_k=top_k, collection="default")
            chunks = [h.text for h in hits]
            context = "\n---\n".join(chunks)
            ctx_tokens = _count_tokens(context)

            predicted = answer_question(context, q["question"], client)
            correct = check_answer(predicted, q["expect_contains"])

            results.append(Result(
                config="retrieval-only",
                question=q["question"],
                expected=q["expect_contains"],
                predicted=predicted,
                correct=correct,
                context_tokens=ctx_tokens,
                brief_tokens=0,
            ))
        m.close()
    return results


def run_briefs_only(
    briefs: str,
    queries: list[dict],
    client: GeminiClient,
) -> list[Result]:
    """Config 2: use only LLM-generated briefs as context."""
    results = []
    brief_tokens = _count_tokens(briefs)

    for q in queries:
        predicted = answer_question(briefs, q["question"], client)
        correct = check_answer(predicted, q["expect_contains"])

        results.append(Result(
            config="briefs-only",
            question=q["question"],
            expected=q["expect_contains"],
            predicted=predicted,
            correct=correct,
            context_tokens=brief_tokens,
            brief_tokens=brief_tokens,
        ))
    return results


def run_briefs_plus_retrieval(
    briefs: str,
    events: list[dict],
    queries: list[dict],
    client: GeminiClient,
    *,
    top_k: int = 5,
) -> list[Result]:
    """Config 3: briefs prepended + K retrieved chunks."""
    results = []
    brief_tokens = _count_tokens(briefs)

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "hybrid.db"
        m = vstash.Memory(project="brief_exp", db=db, collection="default")
        for e in events:
            m.remember(e["text"], title=e["id"], collection="default")

        for q in queries:
            hits = m.search(q["question"], top_k=top_k, collection="default")
            chunks = [h.text for h in hits]
            retrieval_ctx = "\n---\n".join(chunks)
            full_context = f"## Decision Briefs\n{briefs}\n\n## Retrieved Details\n{retrieval_ctx}"
            ctx_tokens = _count_tokens(full_context)

            predicted = answer_question(full_context, q["question"], client)
            correct = check_answer(predicted, q["expect_contains"])

            results.append(Result(
                config="briefs+retrieval",
                question=q["question"],
                expected=q["expect_contains"],
                predicted=predicted,
                correct=correct,
                context_tokens=ctx_tokens,
                brief_tokens=brief_tokens,
            ))
        m.close()
    return results


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=Path,
                   default=Path(__file__).parent.parent / "loop_quality" / "scenarios" / "knowledge_update_hard.json")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    scenario = load_scenario(args.scenario)
    events = scenario["events"]
    queries = scenario["queries"]

    print(f"Scenario: {scenario['name']}")
    print(f"Events: {len(events)} | Queries: {len(queries)}")

    client = GeminiClient()

    # Step 1: Generate briefs
    print("\nGenerating temporal briefs...")
    t0 = time.perf_counter()
    briefs = generate_briefs(events, client)
    brief_time = time.perf_counter() - t0
    brief_tokens = _count_tokens(briefs)
    print(f"Briefs generated in {brief_time:.1f}s ({brief_tokens} tokens)")
    print(f"\n{briefs}\n")

    # Step 2: Run all configs
    print("=" * 70)
    all_results = []

    print("Running retrieval-only...")
    all_results.extend(run_retrieval_only(events, queries, client, top_k=args.top_k))

    print("Running briefs-only...")
    all_results.extend(run_briefs_only(briefs, queries, client))

    print("Running briefs+retrieval...")
    all_results.extend(run_briefs_plus_retrieval(
        briefs, events, queries, client, top_k=args.top_k))

    # Step 3: Report
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    configs = ["retrieval-only", "briefs-only", "briefs+retrieval"]
    for config in configs:
        cr = [r for r in all_results if r.config == config]
        correct = sum(1 for r in cr if r.correct)
        total = len(cr)
        avg_tokens = sum(r.context_tokens for r in cr) / total if total else 0
        print(f"\n--- {config} ---")
        print(f"  Accuracy: {correct}/{total} ({correct/total:.0%})")
        print(f"  Avg context tokens: {avg_tokens:.0f}")
        for r in cr:
            mark = "PASS" if r.correct else "FAIL"
            print(f"  [{mark}] {r.question}")
            print(f"         Expected: {r.expected} | Got: {r.predicted[:100]}")

    print(f"\nTotal Gemini calls: {client.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
