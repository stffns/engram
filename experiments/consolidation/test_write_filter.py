"""Test nanoGPT write filter on knowledge_update_50topics.

Compares retrieval accuracy with and without the nanoGPT noise filter.
If the filter correctly removes noise, the store is cleaner and
retrieval should improve.

IMPORTANT: torch (nanoGPT) and fastembed/ONNX (vstash) cannot safely
coexist in a single Python process on macOS -- after a nanoGPT config
runs, teardown leaks semaphores and the next config segfaults
(notes/nanogpt-training-log.md, Mistake #6). Run one config per
invocation:

    PYTHONPATH=... python -m experiments.consolidation.test_write_filter --config always
    PYTHONPATH=... python -m experiments.consolidation.test_write_filter --config char
    PYTHONPATH=... python -m experiments.consolidation.test_write_filter --config bpe

Each invocation prints a one-line `RESULT:` row; aggregate them manually
(or via `run_write_filter.sh`).
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import tiktoken

from merken import AlwaysWrite, Memory

NANOGPT_DIR = Path(__file__).parent.parent.parent.parent / "nanoGPT"
SCENARIO_DIR = Path(__file__).parent.parent / "loop_quality" / "scenarios"
DEFAULT_SCENARIO = SCENARIO_DIR / "knowledge_update_50topics.json"
DEFAULT_MODEL = "gemini-2.0-flash"

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def load_scenario(path: Path = DEFAULT_SCENARIO):
    with open(path) as f:
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


class _PrecomputedDecider:
    """Look up write decisions by event id from a JSON precomputed offline.

    Exists because torch (nanoGPT) and fastembed (vstash) corrupt each
    other's state in one Python process on macOS -- see
    notes/nanogpt-training-log.md Mistake #10. Run `precompute_nanogpt.py`
    first (torch-only process), then the E2E benchmark reads the
    decisions through this decider (torch-free).
    """

    def __init__(self, name: str, decisions_path: Path) -> None:
        data = json.loads(Path(decisions_path).read_text())
        self.name = name
        self._meta = data["meta"]
        self._by_title: dict[str, dict] = data["decisions"]

    def decide(self, event, ctx):
        from merken.policies import Decision
        d = self._by_title.get(event.title)
        if d is None:
            # No precomputed decision -> default to writing. Keeps recall
            # pessimistic (we won't silently drop something we never classified).
            return Decision(
                write=True, reason="no-precomputed-decision",
                confidence=0.0, policy=self.name,
            )
        return Decision(
            write=bool(d["write"]), reason=d["reason"],
            confidence=float(d["confidence"]), policy=self.name,
        )


def _decider_for(config: str, decisions_file: Path | None = None):
    if config == "always":
        return "always-write", AlwaysWrite()

    if config == "precomputed":
        if decisions_file is None:
            raise SystemExit("--decisions-file required for --config precomputed")
        name = f"precomputed-{decisions_file.stem}"
        return name, _PrecomputedDecider(name, decisions_file)

    raise ValueError(f"unknown config: {config}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=["always", "precomputed"],
        required=True,
        help="Which single config to run in this process.",
    )
    parser.add_argument(
        "--decisions-file",
        type=Path,
        default=None,
        help="Precomputed decisions JSON (required when --config=precomputed).",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="Scenario JSON path (accepts a bare filename under loop_quality/scenarios).",
    )
    args = parser.parse_args()
    scenario_path = args.scenario
    if not scenario_path.exists() and not scenario_path.is_absolute():
        scenario_path = SCENARIO_DIR / scenario_path

    scenario = load_scenario(scenario_path)
    events = scenario["events"]
    queries = scenario["queries"]

    signal_count = sum(1 for e in events if e["topic"] != "noise")
    noise_count = sum(1 for e in events if e["topic"] == "noise")
    print(f"Scenario: {scenario['name']}")
    print(f"Events: {len(events)} ({signal_count} signal, {noise_count} noise)")
    print(f"Queries: {len(queries)}")

    name, decider = _decider_for(args.config, args.decisions_file)
    print(f"\n=== {name} ===")
    r = run_config(name, events, queries, decider)
    print(f"Written: {r['written']}, Skipped: {r['skipped']}")
    print(f"Accuracy: {r['correct']}/{r['total']} ({r['accuracy']:.0%})")
    reduction = (1 - r["written"] / len(events)) * 100 if len(events) else 0
    # Stable one-line output for bash aggregation.
    print(
        f"RESULT: config={name} written={r['written']} skipped={r['skipped']} "
        f"accuracy={r['accuracy']:.4f} correct={r['correct']} total={r['total']} "
        f"reduction={reduction:.1f}"
    )


if __name__ == "__main__":
    main()
