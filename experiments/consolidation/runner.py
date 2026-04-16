"""Consolidation value experiment.

Tests whether merken's consolidation + forget primitives improve answer
accuracy on knowledge-update scenarios where facts evolve over time.

Uses the knowledge_update.json scenario: 4 topics with v1->v2->v3
progressions, 8 noise events, 4 queries expecting the current (v3) state.

Configs tested:
  1. vstash-raw          -- plain vstash, no merken
  2. merken-recall       -- merken with AlwaysWrite, recall only
  3. merken-temporal     -- merken with temporal reranking (weight=0.3)
  4. merken-consol       -- merken + consolidation after ingestion
  5. merken-full         -- merken + consolidation + forget-consolidated

For each config x query:
  - Retrieve K docs
  - Generate answer with Gemini
  - Judge: does answer contain the expected v3 keywords?

This is a focused experiment: 20 events, 4 queries, 5 configs = 40 API calls.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
import vstash

from merken import AlwaysWrite, ForgetConsolidated, Memory, NeverForget

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TOP_K = 5

_ENCODER = tiktoken.get_encoding("cl100k_base")
_SCENARIO = Path(__file__).parent.parent / "loop_quality" / "scenarios" / "knowledge_update.json"


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class Event:
    id: str
    text: str
    topic: str


@dataclass
class Query:
    question: str
    expect_topic: str
    expect_contains: list[str]


@dataclass
class Scenario:
    name: str
    events: list[Event]
    queries: list[Query]


def load_scenario(path: Path = _SCENARIO) -> Scenario:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return Scenario(
        name=raw["name"],
        events=[Event(**e) for e in raw["events"]],
        queries=[Query(**q) for q in raw["queries"]],
    )


# ------------------------------------------------------------------ adapters


class Adapter:
    name: str

    def ingest(self, text: str, *, title: str) -> bool:
        raise NotImplementedError

    def recall(self, query: str, *, top_k: int) -> list[str]:
        raise NotImplementedError

    def consolidate(self) -> int:
        return 0

    def forget(self) -> int:
        return 0

    def doc_count(self) -> int:
        return 0

    def close(self) -> None:
        pass


class VstashAdapter(Adapter):
    name = "vstash-raw"

    def __init__(self, db: Path) -> None:
        self._m = vstash.Memory(project="consol_exp", db=db, collection="default")
        self._count = 0

    def ingest(self, text: str, *, title: str) -> bool:
        self._m.remember(text, title=title, collection="default")
        self._count += 1
        return True

    def recall(self, query: str, *, top_k: int) -> list[str]:
        hits = self._m.search(query, top_k=top_k, collection="default")
        return [h.text for h in hits]

    def doc_count(self) -> int:
        return self._count

    def close(self) -> None:
        self._m.close()


class MerkenAdapter(Adapter):

    def __init__(
        self,
        name: str,
        db: Path,
        *,
        do_consolidate: bool = False,
        do_forget: bool = False,
        temporal_weight: float = 0.0,
        synthesize_fn=None,
        consolidate_method: str = "embedding_v1",
        use_brief_search: bool = False,
    ) -> None:
        self.name = name
        self._do_consolidate = do_consolidate
        self._do_forget = do_forget
        self._synthesize_fn = synthesize_fn
        self._consolidate_method = consolidate_method
        self._use_brief_search = use_brief_search
        forget_decider = ForgetConsolidated() if do_forget else NeverForget()
        self._m = Memory(
            project="consol_exp",
            db=db,
            write_decider=AlwaysWrite(),
            forget_decider=forget_decider,
            temporal_weight=temporal_weight,
        )
        self._count = 0

    def ingest(self, text: str, *, title: str) -> bool:
        result = self._m.remember(text, title=title)
        if result.written:
            self._count += 1
        return result.written

    def recall(self, query: str, *, top_k: int) -> list[str]:
        if self._use_brief_search:
            episodic_hits, brief_texts = self._m.recall_with_briefs(
                query, top_k=top_k, brief_k=3,
            )
            # Briefs first, then episodic
            return brief_texts + [h.text for h in episodic_hits]
        hits = self._m.recall(query, top_k=top_k)
        return [h.text for h in hits]

    def consolidate(self) -> int:
        if not self._do_consolidate:
            return 0
        result = self._m.consolidate(
            method=self._consolidate_method,
            embedding_threshold=0.70,
            embedding_linkage="complete",
            synthesize_fn=self._synthesize_fn,
        )
        return len(result.facts)

    def forget(self) -> int:
        if not self._do_forget:
            return 0
        result = self._m.forget(force=True)
        return len(result.tombstoned)

    def doc_count(self) -> int:
        return self._count

    def close(self) -> None:
        self._m.close()


def _make_configs(synthesize_fn=None) -> dict[str, callable]:
    configs = {
        "vstash-raw": lambda db: VstashAdapter(db),
        "merken-recall": lambda db: MerkenAdapter("merken-recall", db),
        "merken-temporal": lambda db: MerkenAdapter(
            "merken-temporal", db, temporal_weight=0.3,
        ),
        "merken-consol": lambda db: MerkenAdapter(
            "merken-consol", db, do_consolidate=True,
        ),
        "merken-full": lambda db: MerkenAdapter(
            "merken-full", db, do_consolidate=True, do_forget=True,
        ),
    }
    if synthesize_fn is not None:
        configs["merken-llm-consol"] = lambda db: MerkenAdapter(
            "merken-llm-consol", db, do_consolidate=True,
            synthesize_fn=synthesize_fn,
        )
        configs["merken-brief"] = lambda db: MerkenAdapter(
            "merken-brief", db, do_consolidate=True,
            synthesize_fn=synthesize_fn,
            consolidate_method="brief_v1",
        )
        configs["merken-brief-search"] = lambda db: MerkenAdapter(
            "merken-brief-search", db, do_consolidate=True,
            synthesize_fn=synthesize_fn,
            consolidate_method="brief_v1",
            use_brief_search=True,
        )
    return configs


# ------------------------------------------------------------------ judge


def _build_answerer_prompt(context_chunks: list[str], question: str) -> str:
    ctx = "\n---\n".join(context_chunks)
    return (
        "You are answering questions about technical decisions in a software project.\n"
        "Based ONLY on the context below, answer the question concisely.\n"
        "If the context contains multiple versions of a decision, answer with the MOST RECENT one.\n"
        "If the context does not contain enough information, say 'I don't know'.\n\n"
        f"## Context\n{ctx}\n\n"
        f"## Question\n{question}\n\n"
        "## Answer (concise, factual)"
    )


class Judge:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._calls = 0

    def generate_answer(self, context_chunks: list[str], question: str) -> str:
        prompt = _build_answerer_prompt(context_chunks, question)
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt,
        )
        self._calls += 1
        return (resp.text or "").strip()

    @property
    def total_calls(self) -> int:
        return self._calls


# ------------------------------------------------------------------ eval


@dataclass
class QAResult:
    config: str
    question: str
    expected: list[str]
    predicted: str
    correct: bool
    retrieved_chunks: list[str]
    context_tokens: int


@dataclass
class ConfigResult:
    config: str
    docs_stored: int
    facts_created: int
    docs_forgotten: int
    ingest_time_s: float
    eval_time_s: float
    qa_results: list[QAResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(1 for q in self.qa_results if q.correct) / len(self.qa_results)


def _check_answer(predicted: str, expect_contains: list[str]) -> bool:
    lower = predicted.lower()
    return any(kw.lower() in lower for kw in expect_contains)


def run_config(
    config_name: str,
    scenario: Scenario,
    judge: Judge,
    *,
    top_k: int,
    db_dir: Path,
    configs: dict | None = None,
) -> ConfigResult:
    db = db_dir / f"{config_name}.db"
    configs = configs or _make_configs()
    adapter = configs[config_name](db)

    try:
        t0 = time.perf_counter()
        for event in scenario.events:
            adapter.ingest(event.text, title=event.id)

        facts = adapter.consolidate()
        forgotten = adapter.forget()
        ingest_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        qa_results = []

        for query in scenario.queries:
            chunks = adapter.recall(query.question, top_k=top_k)
            context_tokens = sum(_count_tokens(c) for c in chunks)
            predicted = judge.generate_answer(chunks, query.question)
            correct = _check_answer(predicted, query.expect_contains)

            qa_results.append(QAResult(
                config=config_name,
                question=query.question,
                expected=query.expect_contains,
                predicted=predicted,
                correct=correct,
                retrieved_chunks=chunks,
                context_tokens=context_tokens,
            ))

        eval_time = time.perf_counter() - t1

        return ConfigResult(
            config=config_name,
            docs_stored=adapter.doc_count(),
            facts_created=facts,
            docs_forgotten=forgotten,
            ingest_time_s=ingest_time,
            eval_time_s=eval_time,
            qa_results=qa_results,
        )
    finally:
        adapter.close()


def format_report(results: list[ConfigResult]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Consolidation Value Experiment -- knowledge_update scenario")
    lines.append("=" * 80)

    for cr in results:
        lines.append(f"\n--- {cr.config} ---")
        lines.append(f"  Accuracy: {cr.accuracy:.0%} ({sum(1 for q in cr.qa_results if q.correct)}/{len(cr.qa_results)})")
        lines.append(f"  Docs: {cr.docs_stored} | Facts: {cr.facts_created} | Forgotten: {cr.docs_forgotten}")
        for qa in cr.qa_results:
            mark = "PASS" if qa.correct else "FAIL"
            lines.append(f"  [{mark}] Q: {qa.question}")
            lines.append(f"         Expected: {qa.expected}")
            lines.append(f"         Got: {qa.predicted[:120]}")

    lines.append("\n" + "=" * 80)
    lines.append("Summary")
    lines.append("=" * 80)
    header = f"{'Config':<20} {'Accuracy':>10} {'Docs':>6} {'Facts':>6} {'Forgot':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for cr in results:
        lines.append(
            f"{cr.config:<20} {cr.accuracy:>9.0%} {cr.docs_stored:>6} "
            f"{cr.facts_created:>6} {cr.docs_forgotten:>7}"
        )

    return "\n".join(lines)


def _make_gemini_synthesizer(model: str = DEFAULT_MODEL):
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    def synthesize(texts: list[str]) -> str:
        prompt = (
            "Synthesize these related notes into a single concise fact "
            "capturing the CURRENT state. Include key context for why "
            "the current decision was made. One paragraph max.\n\n"
            + "\n".join(f"- {t}" for t in texts)
        )
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()

    return synthesize


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=Path, default=None,
                   help="Path to scenario JSON (default: knowledge_update.json)")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--llm-consol", action="store_true",
                   help="Include merken-llm-consol config")
    p.add_argument("--config", action="append", default=None,
                   help="Specific configs to run (repeat for multiple)")
    args = p.parse_args()

    scenario_path = args.scenario or _SCENARIO
    scenario = load_scenario(scenario_path)
    print(f"Scenario: {scenario.name}")
    print(f"Events: {len(scenario.events)} | Queries: {len(scenario.queries)}")
    print()

    judge = Judge()
    synthesize_fn = _make_gemini_synthesizer() if args.llm_consol else None
    all_configs = _make_configs(synthesize_fn)

    if args.config:
        config_names = args.config
    else:
        config_names = list(all_configs.keys())

    results = []
    with tempfile.TemporaryDirectory(prefix="merken_consol_") as td:
        db_dir = Path(td)
        for config_name in config_names:
            if config_name not in all_configs:
                print(f"Unknown config: {config_name}, skipping")
                continue
            print(f"Running {config_name}...")
            cr = run_config(
                config_name, scenario, judge,
                top_k=args.top_k, db_dir=db_dir,
                configs=all_configs,
            )
            results.append(cr)

    report = format_report(results)
    print(f"\n{report}")
    print(f"\nTotal Gemini API calls: {judge.total_calls}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
