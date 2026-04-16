"""LoCoMo end-to-end runner -- answer accuracy, not just retrieval.

Compares 4 configurations on the original LoCoMo QA task:
  1. vstash-raw      -- plain vstash, no merken overhead
  2. merken-recall   -- merken with heuristic write filter, recall only
  3. merken-consol   -- merken + consolidation after ingestion
  4. merken-full     -- merken + consolidation + forget-consolidated

Pipeline per conversation x config:
  1. Ingest dialogue turns session-by-session (temporal order)
  2. Optionally consolidate / forget
  3. For each QA pair: retrieve K docs, generate answer (Gemini), judge vs gold (Gemini)

Metrics: accuracy per category, tokens stored, tokens fed to answerer,
latency, and audit event counts.

Category mapping (from LoCoMo paper):
  1 = single_hop, 2 = temporal, 3 = multi_hop, 4 = open_domain, 5 = adversarial
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
import vstash

from merken import (
    AlwaysWrite,
    ForgetConsolidated,
    Memory,
    NeverForget,
    PeriodicConsolidator,
)

# ------------------------------------------------------------------ constants

CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}

DEFAULT_CATEGORIES = [2, 5]  # temporal + adversarial (most diagnostic)
DEFAULT_TOP_K = 10
DEFAULT_MODEL = "gemini-2.0-flash"

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ------------------------------------------------------------------ data loading


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    index: int
    date_time: str
    turns: list[Turn]


@dataclass
class QAPair:
    question: str
    answer: str
    evidence: list[str]
    category: int
    category_name: str


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa_pairs: list[QAPair]


def load_locomo(path: Path) -> list[Conversation]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    conversations = []
    for item in raw:
        conv_data = item["conversation"]
        speaker_a = conv_data["speaker_a"]
        speaker_b = conv_data["speaker_b"]

        sessions = []
        i = 1
        while f"session_{i}" in conv_data:
            dt = conv_data.get(f"session_{i}_date_time", "")
            turns = [
                Turn(speaker=t["speaker"], dia_id=t["dia_id"], text=t["text"])
                for t in conv_data[f"session_{i}"]
            ]
            sessions.append(Session(index=i, date_time=dt, turns=turns))
            i += 1

        qa_pairs = []
        for q in item["qa"]:
            cat = q["category"]
            answer = q.get("answer") or q.get("adversarial_answer", "")
            if not answer:
                continue
            qa_pairs.append(QAPair(
                question=q["question"],
                answer=str(answer),
                evidence=q.get("evidence", []),
                category=cat,
                category_name=CATEGORY_NAMES.get(cat, f"cat_{cat}"),
            ))

        conversations.append(Conversation(
            sample_id=item["sample_id"],
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            qa_pairs=qa_pairs,
        ))

    return conversations


# ------------------------------------------------------------------ adapters


class Adapter:
    name: str

    def ingest_session(self, text: str, *, title: str) -> bool:
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

    def __init__(self, project: str, db: Path) -> None:
        self._m = vstash.Memory(project=project, db=db, collection="default")
        self._count = 0

    def ingest_session(self, text: str, *, title: str) -> bool:
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
        project: str,
        db: Path,
        *,
        do_consolidate: bool = False,
        do_forget: bool = False,
    ) -> None:
        self.name = name
        self._do_consolidate = do_consolidate
        self._do_forget = do_forget
        forget_decider = ForgetConsolidated() if do_forget else NeverForget()
        self._m = Memory(
            project=project,
            db=db,
            write_decider=AlwaysWrite(),
            forget_decider=forget_decider,
        )
        self._count = 0

    def ingest_session(self, text: str, *, title: str) -> bool:
        result = self._m.remember(text, title=title)
        if result.written:
            self._count += 1
        return result.written

    def recall(self, query: str, *, top_k: int) -> list[str]:
        hits = self._m.recall(query, top_k=top_k)
        return [h.text for h in hits]

    def consolidate(self) -> int:
        if not self._do_consolidate:
            return 0
        result = self._m.consolidate(
            method="embedding_v1",
            embedding_threshold=0.70,
            embedding_linkage="complete",
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


CONFIGS = {
    "vstash-raw": lambda proj, db: VstashAdapter(proj, db),
    "merken-recall": lambda proj, db: MerkenAdapter(
        "merken-recall", proj, db,
    ),
    "merken-consol": lambda proj, db: MerkenAdapter(
        "merken-consol", proj, db, do_consolidate=True,
    ),
    "merken-full": lambda proj, db: MerkenAdapter(
        "merken-full", proj, db, do_consolidate=True, do_forget=True,
    ),
}


# ------------------------------------------------------------------ judge


def _build_answerer_prompt(context_chunks: list[str], question: str) -> str:
    ctx = "\n---\n".join(context_chunks)
    return (
        "You are answering questions about a long-term conversation between two people.\n"
        "Based ONLY on the conversation excerpts below, answer the question concisely.\n"
        "If the excerpts do not contain enough information, say 'I don't know'.\n\n"
        f"## Conversation excerpts\n{ctx}\n\n"
        f"## Question\n{question}\n\n"
        "## Answer (concise, factual)"
    )


def _build_judge_prompt(question: str, gold: str, predicted: str) -> str:
    return (
        "You are judging whether a predicted answer is correct given a gold answer.\n"
        "The predicted answer does not need to match word-for-word, but must convey the same factual content.\n"
        "If the gold answer is a date/time, allow reasonable paraphrasing.\n"
        "If the predicted answer says 'I don't know' or equivalent, that is INCORRECT.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}\n\n"
        "Is the predicted answer correct? Reply with exactly YES or NO."
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

    def judge_answer(self, question: str, gold: str, predicted: str) -> bool:
        prompt = _build_judge_prompt(question, gold, predicted)
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt,
        )
        self._calls += 1
        text = (resp.text or "").strip().upper()
        return text.startswith("YES")

    @property
    def total_calls(self) -> int:
        return self._calls


# ------------------------------------------------------------------ eval core


@dataclass
class QAResult:
    question: str
    gold: str
    predicted: str
    correct: bool
    category: int
    category_name: str
    context_tokens: int
    num_chunks: int


@dataclass
class ConvResult:
    sample_id: str
    config: str
    turns_ingested: int
    docs_stored: int
    facts_created: int
    docs_forgotten: int
    ingest_time_s: float
    eval_time_s: float
    qa_results: list[QAResult] = field(default_factory=list)


@dataclass
class RunResult:
    config: str
    conversations: list[ConvResult]
    total_api_calls: int

    @property
    def all_qa(self) -> list[QAResult]:
        return [q for c in self.conversations for q in c.qa_results]

    def accuracy_by_category(self) -> dict[str, tuple[float, int]]:
        by_cat: dict[str, list[bool]] = {}
        for q in self.all_qa:
            by_cat.setdefault(q.category_name, []).append(q.correct)
        return {
            cat: (sum(vals) / len(vals), len(vals))
            for cat, vals in sorted(by_cat.items())
        }

    def overall_accuracy(self) -> tuple[float, int]:
        vals = [q.correct for q in self.all_qa]
        return (sum(vals) / len(vals) if vals else 0.0, len(vals))


def bootstrap_ci(
    values: list[bool],
    *,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_iter)
    )
    lo = int((alpha / 2) * n_iter)
    hi = int((1 - alpha / 2) * n_iter) - 1
    return (means[lo], means[hi])


def run_conversation(
    conv: Conversation,
    adapter: Adapter,
    judge: Judge,
    *,
    categories: list[int],
    top_k: int,
) -> ConvResult:
    # 1. Ingest sessions as whole documents
    t0 = time.perf_counter()
    ingested = 0
    for session in conv.sessions:
        lines = [f"{turn.speaker}: {turn.text}" for turn in session.turns]
        session_text = f"[{session.date_time}]\n" + "\n".join(lines)
        title = f"{conv.speaker_a} & {conv.speaker_b} - Session {session.index} ({session.date_time})"
        if adapter.ingest_session(session_text, title=title):
            ingested += 1

    # 2. Consolidate + forget
    facts = adapter.consolidate()
    forgotten = adapter.forget()
    ingest_time = time.perf_counter() - t0

    # 3. Evaluate QA
    t1 = time.perf_counter()
    target_qa = [q for q in conv.qa_pairs if q.category in categories]
    qa_results = []

    for qa in target_qa:
        chunks = adapter.recall(qa.question, top_k=top_k)
        context_tokens = sum(_count_tokens(c) for c in chunks)

        predicted = judge.generate_answer(chunks, qa.question)
        correct = judge.judge_answer(qa.question, qa.answer, predicted)

        qa_results.append(QAResult(
            question=qa.question,
            gold=qa.answer,
            predicted=predicted,
            correct=correct,
            category=qa.category,
            category_name=qa.category_name,
            context_tokens=context_tokens,
            num_chunks=len(chunks),
        ))

    eval_time = time.perf_counter() - t1

    return ConvResult(
        sample_id=conv.sample_id,
        config=adapter.name,
        turns_ingested=ingested,
        docs_stored=adapter.doc_count(),
        facts_created=facts,
        docs_forgotten=forgotten,
        ingest_time_s=ingest_time,
        eval_time_s=eval_time,
        qa_results=qa_results,
    )


def run_config(
    config_name: str,
    conversations: list[Conversation],
    judge: Judge,
    *,
    categories: list[int],
    top_k: int,
    db_dir: Path,
    max_convs: int | None = None,
) -> RunResult:
    convs = conversations[:max_convs] if max_convs else conversations
    results = []

    for i, conv in enumerate(convs):
        db = db_dir / f"{config_name}_{conv.sample_id}.db"
        adapter = CONFIGS[config_name](f"locomo_{conv.sample_id}", db)
        try:
            cr = run_conversation(
                conv, adapter, judge,
                categories=categories, top_k=top_k,
            )
            results.append(cr)
            n_correct = sum(1 for q in cr.qa_results if q.correct)
            n_total = len(cr.qa_results)
            acc = n_correct / n_total if n_total else 0
            print(
                f"  [{config_name}] conv {i+1}/{len(convs)} "
                f"({conv.sample_id}): {n_correct}/{n_total} = {acc:.1%} "
                f"| ingested={cr.turns_ingested} facts={cr.facts_created} "
                f"forgot={cr.docs_forgotten} | {cr.eval_time_s:.1f}s"
            )
        finally:
            adapter.close()

    return RunResult(
        config=config_name,
        conversations=results,
        total_api_calls=judge.total_calls,
    )


# ------------------------------------------------------------------ reporting


def format_report(results: list[RunResult]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("LoCoMo End-to-End Results")
    lines.append("=" * 80)

    for rr in results:
        lines.append(f"\n--- {rr.config} ---")
        overall_acc, overall_n = rr.overall_accuracy()
        ci = bootstrap_ci([q.correct for q in rr.all_qa])
        lines.append(f"  Overall: {overall_acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (n={overall_n})")

        for cat_name, (acc, n) in rr.accuracy_by_category().items():
            cat_vals = [q.correct for q in rr.all_qa if q.category_name == cat_name]
            cat_ci = bootstrap_ci(cat_vals)
            lines.append(f"  {cat_name}: {acc:.3f} [{cat_ci[0]:.3f}, {cat_ci[1]:.3f}] (n={n})")

        total_ctx = sum(q.context_tokens for q in rr.all_qa)
        avg_ctx = total_ctx / len(rr.all_qa) if rr.all_qa else 0
        total_ingest = sum(c.turns_ingested for c in rr.conversations)
        total_facts = sum(c.facts_created for c in rr.conversations)
        total_forgot = sum(c.docs_forgotten for c in rr.conversations)
        lines.append(f"  Tokens fed (avg/query): {avg_ctx:.0f}")
        lines.append(f"  Turns ingested: {total_ingest}")
        lines.append(f"  Facts created: {total_facts}")
        lines.append(f"  Docs forgotten: {total_forgot}")

    # Comparison table
    lines.append("\n" + "=" * 80)
    lines.append("Comparison Table")
    lines.append("=" * 80)
    header = f"{'Config':<20}"
    cats_seen = set()
    for rr in results:
        for q in rr.all_qa:
            cats_seen.add(q.category_name)
    cat_list = sorted(cats_seen)
    for c in cat_list:
        header += f" {c:>12}"
    header += f" {'overall':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for rr in results:
        row = f"{rr.config:<20}"
        by_cat = rr.accuracy_by_category()
        for c in cat_list:
            if c in by_cat:
                row += f" {by_cat[c][0]:>11.3f}"
            else:
                row += f" {'--':>12}"
        overall, _ = rr.overall_accuracy()
        row += f" {overall:>11.3f}"
        lines.append(row)

    return "\n".join(lines)


def save_detailed_json(results: list[RunResult], path: Path) -> None:
    out = []
    for rr in results:
        for cr in rr.conversations:
            for qa in cr.qa_results:
                out.append({
                    "config": rr.config,
                    "sample_id": cr.sample_id,
                    "category": qa.category,
                    "category_name": qa.category_name,
                    "question": qa.question,
                    "gold": qa.gold,
                    "predicted": qa.predicted,
                    "correct": qa.correct,
                    "context_tokens": qa.context_tokens,
                    "num_chunks": qa.num_chunks,
                })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoCoMo end-to-end QA runner for merken.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "locomo10.json",
        help="Path to locomo10.json",
    )
    p.add_argument(
        "--config",
        action="append",
        choices=sorted(CONFIGS),
        help="Configs to run (default: all 4)",
    )
    p.add_argument(
        "--category",
        type=int,
        action="append",
        help=f"Category IDs to evaluate (default: {DEFAULT_CATEGORIES}). {CATEGORY_NAMES}",
    )
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--max-convs", type=int, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--db-dir", type=Path, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results_detail.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    conversations = load_locomo(args.data)
    categories = args.category or DEFAULT_CATEGORIES
    configs = args.config or sorted(CONFIGS)

    cat_names = [CATEGORY_NAMES.get(c, f"cat_{c}") for c in categories]
    total_qa = sum(
        len([q for q in conv.qa_pairs if q.category in categories])
        for conv in conversations
    )

    print(f"LoCoMo E2E | {len(conversations)} convs | categories: {cat_names}")
    print(f"Total QA pairs: {total_qa} | configs: {configs}")
    print(f"Model: {args.model} | top_k: {args.top_k}")
    print()

    judge = Judge(model=args.model)
    all_results = []

    with tempfile.TemporaryDirectory(prefix="merken_locomo_") as td:
        db_dir = args.db_dir or Path(td)
        db_dir.mkdir(parents=True, exist_ok=True)

        for config_name in configs:
            print(f"\n=== Running {config_name} ===")
            result = run_config(
                config_name,
                conversations,
                judge,
                categories=categories,
                top_k=args.top_k,
                db_dir=db_dir,
                max_convs=args.max_convs,
            )
            all_results.append(result)

    report = format_report(all_results)
    print(f"\n{report}")

    save_detailed_json(all_results, args.output)
    print(f"\nDetailed results: {args.output}")
    print(f"Total Gemini API calls: {judge.total_calls}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
