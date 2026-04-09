"""Tests for the loop_quality scenario runner.

Uses a synthetic scenario with fake vectors so the runner's metric
math is tested in isolation from the real embedder. The real
``session_2026_04_09`` scenario is exercised as an end-to-end smoke
test that just checks the runner finishes and produces sensible
bounds — the actual numbers are recorded in ``RESULTS.md`` and can
shift as engram's consolidation improves.
"""

from __future__ import annotations

from pathlib import Path

from engram.consolidation import Fact, fact_fingerprint
from experiments.loop_quality.runner import (
    _compute_cluster_purity,
    _compute_topic_coverage,
    _fact_path_to_topic,
    format_result,
    run_scenario,
)
from experiments.loop_quality.scenario import (
    Scenario,
    ScenarioEvent,
    ScenarioQuery,
    load_scenario,
)

FIXTURE_DIR = (
    Path(__file__).parent.parent
    / "experiments"
    / "loop_quality"
    / "scenarios"
)


# ----------------------------------------------------------------- loader


def test_load_real_scenario_schema() -> None:
    """The committed session fixture must parse into a Scenario."""
    scenario = load_scenario(FIXTURE_DIR / "session_2026_04_09.json")
    assert scenario.name == "session_2026_04_09"
    assert len(scenario.events) == 12
    assert len(scenario.queries) == 4
    assert len(scenario.topics) == 6
    # Every topic has exactly 2 events in this fixture.
    assert all(count == 2 for count in scenario.topic_counts.values())


# --------------------------------------------------------------- pure metrics


def _fact(derived_from: list[str], text: str = "x") -> Fact:
    return Fact(
        text=text,
        derived_from=derived_from,
        cluster_size=len(derived_from),
        method="concat_v1",
    )


def test_cluster_purity_all_pure() -> None:
    facts = [
        _fact(["p1", "p2"]),
        _fact(["p3", "p4"]),
    ]
    path_to_topic = {
        "p1": "alpha",
        "p2": "alpha",
        "p3": "beta",
        "p4": "beta",
    }
    assert _compute_cluster_purity(facts, path_to_topic) == 1.0


def test_cluster_purity_one_mixed() -> None:
    facts = [
        _fact(["p1", "p2"]),  # pure
        _fact(["p3", "p4"]),  # mixed
    ]
    path_to_topic = {
        "p1": "alpha",
        "p2": "alpha",
        "p3": "beta",
        "p4": "gamma",
    }
    assert _compute_cluster_purity(facts, path_to_topic) == 0.5


def test_cluster_purity_no_multi_clusters_is_vacuously_pure() -> None:
    facts = [_fact(["p1"])]  # singleton
    assert _compute_cluster_purity(facts, {"p1": "alpha"}) == 1.0


def test_topic_coverage_all_covered() -> None:
    scenario = Scenario(
        name="t",
        description="",
        events=[
            ScenarioEvent(id="1", text="x", topic="alpha"),
            ScenarioEvent(id="2", text="y", topic="alpha"),
            ScenarioEvent(id="3", text="z", topic="beta"),
            ScenarioEvent(id="4", text="w", topic="beta"),
        ],
        queries=[],
    )
    facts = [_fact(["p1", "p2"]), _fact(["p3", "p4"])]
    path_to_topic = {"p1": "alpha", "p2": "alpha", "p3": "beta", "p4": "beta"}
    assert _compute_topic_coverage(facts, scenario, path_to_topic) == 1.0


def test_topic_coverage_ignores_singletons() -> None:
    """Topics with only 1 event aren't eligible — shouldn't penalize coverage."""
    scenario = Scenario(
        name="t",
        description="",
        events=[
            ScenarioEvent(id="1", text="x", topic="alpha"),
            ScenarioEvent(id="2", text="y", topic="alpha"),
            ScenarioEvent(id="3", text="z", topic="loner"),  # only 1 event
        ],
        queries=[],
    )
    facts = [_fact(["p1", "p2"])]
    path_to_topic = {"p1": "alpha", "p2": "alpha"}
    # Only alpha is eligible and it's covered → 100%
    assert _compute_topic_coverage(facts, scenario, path_to_topic) == 1.0


def test_topic_coverage_partial() -> None:
    scenario = Scenario(
        name="t",
        description="",
        events=[
            ScenarioEvent(id="1", text="x", topic="alpha"),
            ScenarioEvent(id="2", text="y", topic="alpha"),
            ScenarioEvent(id="3", text="z", topic="beta"),
            ScenarioEvent(id="4", text="w", topic="beta"),
        ],
        queries=[],
    )
    facts = [_fact(["p1", "p2"])]  # only alpha covered
    path_to_topic = {"p1": "alpha", "p2": "alpha"}
    assert _compute_topic_coverage(facts, scenario, path_to_topic) == 0.5


def test_fact_path_to_topic_skips_mixed_clusters() -> None:
    facts = [
        _fact(["p1", "p2"], text="pure"),
        _fact(["p3", "p4"], text="mixed"),
    ]
    path_to_topic = {
        "p1": "alpha",
        "p2": "alpha",
        "p3": "beta",
        "p4": "gamma",
    }
    result = _fact_path_to_topic(facts, path_to_topic)
    # Pure fact is mapped; mixed fact is excluded
    assert len(result) == 1
    pure_path = f"text://fact_{fact_fingerprint(facts[0])}"
    assert result[pure_path] == "alpha"


# ------------------------------------------------------------- end-to-end smoke


def _discover_scenarios() -> list[Path]:
    """Auto-discover every *.json scenario in the fixtures dir.

    New scenarios (including the real-vstash snapshots) get picked
    up by the parametrized smoke test automatically. No scenario
    lives outside CI just because someone forgot to add a test.
    """
    return sorted(FIXTURE_DIR.glob("*.json"))


import pytest  # noqa: E402


@pytest.mark.parametrize(
    "scenario_path",
    _discover_scenarios(),
    ids=lambda p: p.stem,
)
def test_runner_completes_on_every_scenario(
    scenario_path: Path,
    tmp_path: Path,
) -> None:
    """Auto-discovered smoke test. Every scenario in fixtures/ runs
    through the full runner and must produce sensible metric bounds.
    Exact numbers for each scenario live in RESULTS.md.

    This is the test that catches "smoke lives outside CI" (Silt,
    2026-04-09). Adding a new scenario JSON is enough to get it
    into the safety net — no pytest edit required.
    """
    scenario = load_scenario(scenario_path)
    result = run_scenario(scenario, db=tmp_path / f"{scenario.name}.db")

    assert result.scenario == scenario.name
    assert result.n_events == len(scenario.events)
    assert 0 <= result.facts_written <= result.n_events
    assert 0.0 <= result.query_pass_rate <= 1.0
    assert 0.0 <= result.cluster_purity <= 1.0
    assert 0.0 <= result.topic_coverage <= 1.0
    assert result.queries_total == len(scenario.queries)
    assert len(result.query_outcomes) == len(scenario.queries)
    # At least one query must pass — if the loop can't answer *any*
    # question on a scenario, that's a regression worth alarming on.
    assert result.queries_passing >= 1, (
        f"no queries passed on {scenario.name} — engram regressed? "
        f"outcomes: {result.query_outcomes}"
    )


def test_format_result_is_human_readable(tmp_path: Path) -> None:
    scenario = load_scenario(FIXTURE_DIR / "session_2026_04_09.json")
    result = run_scenario(scenario, db=tmp_path / "lq.db")
    text = format_result(result)
    assert "scenario: session_2026_04_09" in text
    assert "query_pass_rate" in text
    assert "cluster_purity" in text
    assert "topic_coverage" in text
