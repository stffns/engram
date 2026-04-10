"""Scenario data model + JSON loader for loop_quality benchmarks.

A scenario is a self-contained description of an agent's episodic
stream plus a list of queries the agent should be able to answer
against the consolidated semantic memory. The runner in ``runner.py``
loads these and reports pass rates.

Schema (intentionally minimal):

```json
{
    "name": "session_2026_04_09",
    "description": "12 snippets from engram's design session — 6 topics, 2 events each",
    "events": [
        {"id": "e01", "text": "...", "topic": "mempalace"},
        ...
    ],
    "queries": [
        {
            "question": "why did we not target mempalace?",
            "expect_topic": "mempalace",
            "expect_contains": ["mempalace"]
        },
        ...
    ]
}
```

``expect_topic`` is ground truth — the runner checks whether the
top-k semantic hits include a fact whose provenance leads back to
an event with this topic. ``expect_contains`` is an optional
secondary check on the fact text itself, useful for disambiguating
scenarios where multiple facts might match the topic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ScenarioEvent:
    """One event in the episodic stream, with ground-truth topic label.

    The optional ``content_type`` field maps to A-MAC-style type priors
    (e.g. ``decision``, ``tool_echo``, ``ack``, ``ambient_chat``). When
    present, the runner encodes it as a ``type:<kind>`` tag alongside
    the ``topic:<topic>`` tag, so a ``ContentTypePriorDecider`` can act
    on it.
    """

    id: str
    text: str
    topic: str
    content_type: str | None = None


@dataclass(frozen=True)
class ScenarioQuery:
    """One query against the consolidated semantic layer."""

    question: str
    expect_topic: str
    expect_contains: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Scenario:
    """A loop-quality scenario: events + queries + ground-truth labels."""

    name: str
    description: str
    events: list[ScenarioEvent]
    queries: list[ScenarioQuery]

    @property
    def topics(self) -> set[str]:
        return {e.topic for e in self.events}

    @property
    def topic_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.topic] = counts.get(e.topic, 0) + 1
        return counts


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario JSON file into a ``Scenario`` dataclass."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    events = [
        ScenarioEvent(
            id=e["id"],
            text=e["text"],
            topic=e["topic"],
            content_type=e.get("content_type"),
        )
        for e in data["events"]
    ]
    queries = [
        ScenarioQuery(
            question=q["question"],
            expect_topic=q["expect_topic"],
            expect_contains=q.get("expect_contains", []),
        )
        for q in data["queries"]
    ]

    return Scenario(
        name=data["name"],
        description=data["description"],
        events=events,
        queries=queries,
    )
