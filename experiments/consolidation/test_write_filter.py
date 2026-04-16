"""Test nanoGPT write filter on knowledge_update_50topics.

Compares retrieval accuracy with and without the nanoGPT noise filter.
If the filter correctly removes noise, the store is cleaner and
retrieval should improve.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import tiktoken

from merken import AlwaysWrite, Memory

NANOGPT_DIR = Path(__file__).parent.parent.parent.parent / "nanoGPT"
SCENARIO = Path(__file__).parent.parent / "loop_quality" / "scenarios" / "knowledge_update_50topics.json"
DEFAULT_MODEL = "gemini-2.0-flash"

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def load_scenario():
    with open(SCENARIO) as f:
        return json.load(f)


def make_judge():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    def answer(context: str, question: str) -> str:
        prompt = (
            "Based on the context below, answer concisely.\n"
            "If multiple versions exist, answer with the MOST RECENT.\n\n"
            f"## Context\n{context}\n\n"
            f"## Question\n{question}\n\n## Answer"
        )
        r = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
        return (r.text or "").strip()

    return answer


def check_answer(predicted: str, expect_contains: list[str]) -> bool:
    lower = predicted.lower()
    return any(kw.lower() in lower for kw in expect_contains)


def run_config(config_name, events, queries, write_decider, top_k=5):
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / f"{config_name}.db"
        m = Memory(project="filter_test", db=db, write_decider=write_decider)

        written = 0
        skipped = 0
        t0 = time.perf_counter()
        for e in events:
            r = m.remember(e["text"], title=e["id"])
            if r.written:
                written += 1
            else:
                skipped += 1
        ingest_time = time.perf_counter() - t0

        # Count by type
        signal_written = sum(1 for e in events if e["topic"] != "noise"
                           and any(r.written for r in [m.remember.__func__] if False))

        judge = make_judge()
        correct = 0
        total = len(queries)

        for q in queries:
            hits = m.recall(q["question"], top_k=top_k)
            context = "\n---\n".join(h.text for h in hits)
            predicted = judge(context, q["question"])
            if check_answer(predicted, q["expect_contains"]):
                correct += 1
            else:
                print(f"  FAIL: {q['question'][:50]}... -> {predicted[:60]}")

        m.close()

    return {
        "config": config_name,
        "written": written,
        "skipped": skipped,
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
        "ingest_time": ingest_time,
    }


def main():
    scenario = load_scenario()
    events = scenario["events"]
    queries = scenario["queries"]

    signal_count = sum(1 for e in events if e["topic"] != "noise")
    noise_count = sum(1 for e in events if e["topic"] == "noise")
    print(f"Scenario: {scenario['name']}")
    print(f"Events: {len(events)} ({signal_count} signal, {noise_count} noise)")
    print(f"Queries: {len(queries)}")

    # Config 1: AlwaysWrite (baseline)
    print("\n=== AlwaysWrite (baseline) ===")
    r1 = run_config("always-write", events, queries, AlwaysWrite())
    print(f"Written: {r1['written']}, Skipped: {r1['skipped']}")
    print(f"Accuracy: {r1['correct']}/{r1['total']} ({r1['accuracy']:.0%})")

    # Config 2: nanoGPT classifier
    ckpt = NANOGPT_DIR / "out-merken" / "ckpt.pt"
    meta = NANOGPT_DIR / "data" / "merken" / "meta.pkl"

    if not ckpt.exists():
        print(f"\nCheckpoint not found: {ckpt}")
        return

    from merken.classifiers.nanogpt import NanoGPTWriteDecider

    print("\n=== NanoGPT Write Filter ===")
    decider = NanoGPTWriteDecider(ckpt, meta, confidence_threshold=0.5)
    r2 = run_config("nanogpt-filter", events, queries, decider)
    print(f"Written: {r2['written']}, Skipped: {r2['skipped']}")
    print(f"Accuracy: {r2['correct']}/{r2['total']} ({r2['accuracy']:.0%})")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Config':<20} {'Written':>8} {'Skipped':>8} {'Accuracy':>10}")
    print(f"{'-'*60}")
    print(f"{'always-write':<20} {r1['written']:>8} {r1['skipped']:>8} {r1['accuracy']:>9.0%}")
    print(f"{'nanogpt-filter':<20} {r2['written']:>8} {r2['skipped']:>8} {r2['accuracy']:>9.0%}")
    print(f"{'='*60}")

    # Noise filtering analysis
    print(f"\nIdeal: write {signal_count} signal, skip {noise_count} noise")
    print(f"nanoGPT: wrote {r2['written']}, skipped {r2['skipped']}")
    if r2['written'] < len(events):
        reduction = (1 - r2['written'] / len(events)) * 100
        print(f"Store reduction: {reduction:.0f}%")


if __name__ == "__main__":
    main()
