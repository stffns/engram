"""Scenario runner for loop_quality benchmarks.

Ingests every event in a scenario into a fresh ``Memory``, runs
``consolidate()``, then asks each query. A query passes if any of
the top-k hits from ``recall(layer="semantic")`` is a fact whose
provenance (its ``derived_from`` list, encoded in the fact's tags)
points back at an event with the query's expected topic.

Metrics reported per scenario:

- **query_pass_rate** — fraction of queries that passed. Primary metric.
- **facts_written** — how many clusters the consolidator produced.
- **cluster_purity** — of all multi-event facts, the fraction whose
  derived events all share a topic. Secondary metric.
- **topic_coverage** — of all topics with ≥ 2 events, the fraction
  that produced at least one fact.

The runner is intentionally a CLI + library pair so tests can call
``run_scenario`` directly with a ``Scenario`` object while the CLI
wraps file I/O.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from engram import Memory, PeriodicConsolidator
from engram.consolidation import Fact, fact_fingerprint
from experiments.loop_quality.scenario import (
    Scenario,
    ScenarioEvent,
    ScenarioQuery,
    load_scenario,
)


@dataclass(frozen=True)
class QueryOutcome:
    question: str
    expect_topic: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    n_events: int
    facts_written: int
    query_pass_rate: float
    queries_passing: int
    queries_total: int
    cluster_purity: float
    topic_coverage: float
    elapsed_s: float
    query_outcomes: list[QueryOutcome] = field(default_factory=list)


def _ingest_scenario(mem: Memory, scenario: Scenario) -> dict[str, str]:
    """Ingest every event and return ``{doc_path: topic}``.

    The doc_path is the vstash key under which the event was stored
    (``text://event_<id>``). Having this map lets us resolve a
    fact's ``derived_from`` back to ground-truth topics.
    """
    path_to_topic: dict[str, str] = {}
    for event in scenario.events:
        result = mem.remember(
            event.text,
            title=f"event_{event.id}",
            tags=f"topic:{event.topic}",
        )
        if result.written and result.ingest is not None:
            path_to_topic[result.ingest.source] = event.topic
    return path_to_topic


def _fact_path_to_topic(
    facts: list[Fact],
    path_to_topic: dict[str, str],
) -> dict[str, str]:
    """Map each written fact's vstash path to the topic of its source cluster.

    A fact is considered to have a topic only if every event in its
    ``derived_from`` list shares the same topic — mixed clusters are
    excluded (they will neither pass nor fail a topic-matched query;
    they show up as cluster_purity misses instead).

    The fact's path is reconstructed from ``fact_fingerprint`` — the
    same function ``Memory.consolidate`` uses to name the semantic
    doc. This couples the runner to the naming scheme, which is
    acceptable: the runner's whole job is to probe a specific engram
    build.
    """
    result: dict[str, str] = {}
    for fact in facts:
        topics = {path_to_topic.get(p) for p in fact.derived_from}
        topics.discard(None)
        if len(topics) != 1:
            continue
        path = f"text://fact_{fact_fingerprint(fact)}"
        result[path] = next(iter(topics))
    return result


def _compute_cluster_purity(
    facts: list,
    path_to_topic: dict[str, str],
) -> float:
    """Of all multi-event facts, the fraction whose derived events
    all share a topic. Singletons are ignored (they have no cluster
    to be pure or impure about)."""
    multi = [f for f in facts if f.cluster_size >= 2]
    if not multi:
        return 1.0  # no clusters → vacuously pure
    pure = 0
    for fact in multi:
        topics = {path_to_topic.get(p) for p in fact.derived_from}
        topics.discard(None)
        if len(topics) == 1:
            pure += 1
    return pure / len(multi)


def _compute_topic_coverage(
    facts: list,
    scenario: Scenario,
    path_to_topic: dict[str, str],
) -> float:
    """Of all topics with ≥ 2 events, the fraction that produced at
    least one multi-event fact."""
    eligible = {t for t, n in scenario.topic_counts.items() if n >= 2}
    if not eligible:
        return 1.0
    covered: set[str] = set()
    for fact in facts:
        if fact.cluster_size < 2:
            continue
        topics = {path_to_topic.get(p) for p in fact.derived_from}
        topics.discard(None)
        if len(topics) == 1:
            covered.add(next(iter(topics)))
    return len(covered & eligible) / len(eligible)


def _evaluate_query(
    mem: Memory,
    query: ScenarioQuery,
    path_to_topic: dict[str, str],
    *,
    top_k: int,
) -> QueryOutcome:
    """Evaluate one query through the full engram recall path.

    ``path_to_topic`` is a *unified* lookup that includes both
    ingested episodic events and materialized semantic facts: a hit
    passes if its path maps to the expected topic via either route.
    This change (2026-04-09) lets ``should_recall``'s episodic
    fallback contribute to the pass rate when consolidation missed a
    cluster but the original events are still findable.
    """
    # No explicit layer → routes via should_recall decider
    hits = mem.recall(query.question, top_k=top_k)
    if not hits:
        return QueryOutcome(
            question=query.question,
            expect_topic=query.expect_topic,
            passed=False,
            reason="no_hits",
        )

    for hit in hits:
        hit_path = getattr(hit, "path", None)
        if hit_path is None:
            continue
        topic = path_to_topic.get(hit_path)
        if topic != query.expect_topic:
            continue

        if query.expect_contains:
            hit_text = (hit.text or "").lower()
            if not all(s.lower() in hit_text for s in query.expect_contains):
                continue

        return QueryOutcome(
            question=query.question,
            expect_topic=query.expect_topic,
            passed=True,
            reason=f"matched:{hit_path}",
        )

    return QueryOutcome(
        question=query.question,
        expect_topic=query.expect_topic,
        passed=False,
        reason="no_hit_matched_expected_topic",
    )


def run_scenario(
    scenario: Scenario,
    *,
    db: Path,
    top_k: int = 5,
    embedding_threshold: float = 0.65,
    embedding_linkage: str = "complete",
    write_decider: Any = None,
    recall_decider: Any = None,
    consolidate_decider: Any = None,
    forget_decider: Any = None,
) -> ScenarioResult:
    """Run one scenario end-to-end against a fresh ``Memory``.

    The four ``*_decider`` kwargs let a caller plug in custom
    deciders to validate them against a real scenario without
    writing their own driver script. ``None`` means "use the
    engram default for this primitive" (except
    ``consolidate_decider``, which defaults to
    ``PeriodicConsolidator(min_events=2)`` to match the
    scenario sizes — 2 is low enough to fire on every
    scenario's 12 or 20 events).

    Added 2026-04-09 in response to the extending.md review
    (Jay) — the runner must be runnable with custom deciders
    for the "validate before landing" workflow to be real.
    """
    t0 = time.perf_counter()

    cons_decider = consolidate_decider or PeriodicConsolidator(min_events=2)

    with Memory(
        project=f"loop_quality_{scenario.name}",
        db=db,
        write_decider=write_decider,
        recall_decider=recall_decider,
        consolidate_decider=cons_decider,
        forget_decider=forget_decider,
    ) as mem:
        path_to_topic = _ingest_scenario(mem, scenario)

        consolidation = mem.consolidate(
            method="embedding_v1",
            embedding_threshold=embedding_threshold,
            embedding_linkage=embedding_linkage,
        )

        fact_topics = _fact_path_to_topic(consolidation.facts, path_to_topic)

        # Unified lookup: a hit's path may be an episodic event path
        # (from the ingest phase) or a semantic fact path (derived
        # here from the consolidation). Either can satisfy a query.
        unified_topic: dict[str, str] = {**path_to_topic, **fact_topics}

        outcomes = [
            _evaluate_query(mem, q, unified_topic, top_k=top_k)
            for q in scenario.queries
        ]

        purity = _compute_cluster_purity(consolidation.facts, path_to_topic)
        coverage = _compute_topic_coverage(
            consolidation.facts, scenario, path_to_topic
        )

    passing = sum(1 for o in outcomes if o.passed)
    total = len(outcomes)

    return ScenarioResult(
        scenario=scenario.name,
        n_events=len(scenario.events),
        facts_written=consolidation.facts_written,
        query_pass_rate=(passing / total) if total else 0.0,
        queries_passing=passing,
        queries_total=total,
        cluster_purity=purity,
        topic_coverage=coverage,
        elapsed_s=time.perf_counter() - t0,
        query_outcomes=outcomes,
    )


def format_result(result: ScenarioResult) -> str:
    lines = [
        f"# scenario: {result.scenario}",
        f"  events:          {result.n_events}",
        f"  facts_written:   {result.facts_written}",
        f"  query_pass_rate: {result.query_pass_rate:.2%}"
        f"  ({result.queries_passing}/{result.queries_total})",
        f"  cluster_purity:  {result.cluster_purity:.2%}",
        f"  topic_coverage:  {result.topic_coverage:.2%}",
        f"  elapsed:         {result.elapsed_s:.1f}s",
        "",
        "  queries:",
    ]
    for o in result.query_outcomes:
        mark = "✓" if o.passed else "✗"
        lines.append(f"    {mark} [{o.expect_topic}] {o.question}")
        if not o.passed:
            lines.append(f"        reason: {o.reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


_SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def _discover_scenarios() -> list[Path]:
    return sorted(_SCENARIOS_DIR.glob("*.json"))


def _import_decider(dotted_path: str) -> Any:
    """Import and default-construct a decider class from a dotted path.

    Example: ``my_package.my_module.MyDecider``. The class must be
    default-constructible (no required args); for parameterized
    deciders, define a subclass that hard-codes the params::

        class MyAggressiveDecider(PeriodicConsolidator):
            def __init__(self):
                super().__init__(min_events=2)

    Added 2026-04-09 to support the "validate your custom decider
    on a loop_quality scenario" workflow from docs/extending.md.
    """
    if "." not in dotted_path:
        raise ValueError(
            f"decider path must be dotted (e.g. my_pkg.MyDecider), got {dotted_path!r}"
        )
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"could not import {module_path!r} for decider {dotted_path!r}: {e}"
        ) from e

    if not hasattr(module, class_name):
        raise ValueError(
            f"module {module_path!r} has no attribute {class_name!r}"
        )
    cls = getattr(module, class_name)
    try:
        return cls()
    except TypeError as e:
        raise ValueError(
            f"{dotted_path!r} requires constructor args; "
            f"subclass and hard-code them. Error: {e}"
        ) from e


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="experiments.loop_quality.runner",
        description="Run loop-quality scenarios against current engram.",
    )
    p.add_argument(
        "--scenario",
        type=Path,
        action="append",
        default=None,
        help="Path to a scenario JSON. Repeat to run multiple. "
        "Default: every *.json in scenarios/.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--embedding-threshold", type=float, default=0.70)
    p.add_argument(
        "--embedding-linkage",
        default="complete",
        choices=["complete", "average", "single"],
        help="clustering linkage (default: complete)",
    )
    p.add_argument(
        "--write-decider",
        default=None,
        metavar="DOTTED.PATH",
        help=(
            "dotted import path to a WriteDecider class (e.g. "
            "my_pkg.MyDecider). Must be default-constructible. "
            "Falls back to engram default when omitted."
        ),
    )
    p.add_argument(
        "--recall-decider",
        default=None,
        metavar="DOTTED.PATH",
        help="same format as --write-decider, for RecallDecider",
    )
    p.add_argument(
        "--consolidate-decider",
        default=None,
        metavar="DOTTED.PATH",
        help="same format as --write-decider, for ConsolidateDecider",
    )
    p.add_argument(
        "--forget-decider",
        default=None,
        metavar="DOTTED.PATH",
        help="same format as --write-decider, for ForgetDecider",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenario_paths = args.scenario or _discover_scenarios()

    if not scenario_paths:
        print("no scenarios found", file=sys.stderr)
        return 2

    # Resolve decider overrides before running anything — fail fast
    # if any dotted path is wrong.
    write_dec = _import_decider(args.write_decider) if args.write_decider else None
    recall_dec = _import_decider(args.recall_decider) if args.recall_decider else None
    cons_dec = (
        _import_decider(args.consolidate_decider) if args.consolidate_decider else None
    )
    forget_dec = _import_decider(args.forget_decider) if args.forget_decider else None

    with tempfile.TemporaryDirectory(prefix="engram_loop_quality_") as td:
        db_dir = Path(td)
        for path in scenario_paths:
            scenario = load_scenario(path)
            db = db_dir / f"{scenario.name}.db"
            result = run_scenario(
                scenario,
                db=db,
                top_k=args.top_k,
                embedding_threshold=args.embedding_threshold,
                embedding_linkage=args.embedding_linkage,
                write_decider=write_dec,
                recall_decider=recall_dec,
                consolidate_decider=cons_dec,
                forget_decider=forget_dec,
            )
            print(format_result(result))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
