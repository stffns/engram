"""Tests for the merken CLI.

Every test exercises ``merken.cli.main(argv)`` end-to-end on a
throwaway DB under ``tmp_path``. No mocking; the CLI is thin
enough that mocking it is more work than calling it.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from merken.cli import default_db_path, main


def _base(tmp_path: Path, *extra: str) -> list[str]:
    """Canonical base argv with pinned --db for test isolation."""
    return ["--db", str(tmp_path / "e.db"), "--project", "test_cli", *extra]


# ------------------------------------------------------------------ defaults


def test_default_db_path_isolated_from_vstash() -> None:
    """Default DB must live under ~/.merken/, not ~/.vstash/."""
    p = default_db_path("my_project")
    assert str(p).endswith("/.merken/my_project.db")
    assert ".vstash" not in str(p)


# -------------------------------------------------------------------- remember


def test_remember_writes_and_prints_human(tmp_path: Path, capsys) -> None:
    code = main(_base(tmp_path, "remember", "the user said analytics is on fire"))
    out = capsys.readouterr().out

    assert code == 0
    assert "wrote" in out
    assert "novel" in out
    assert "HeuristicWriteDecider" in out


def test_remember_json_output_is_valid(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "--db",
            str(tmp_path / "e.db"),
            "--project",
            "test_cli",
            "--json",
            "remember",
            "a json payload event long enough to pass vstash ingest guardrails",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)
    assert payload["written"] is True
    assert payload["decision"]["reason"] == "novel"
    assert payload["decision"]["policy"] == "HeuristicWriteDecider"


def test_remember_skipped_on_duplicate(tmp_path: Path, capsys) -> None:
    text = "exact duplicate event for the dedup test"
    main(_base(tmp_path, "remember", text))
    capsys.readouterr()
    code = main(_base(tmp_path, "remember", text))
    out = capsys.readouterr().out

    assert code == 0
    assert "skipped" in out
    assert "dup_exact" in out


def test_remember_stdin(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        StringIO(
            "an event via stdin about volcanoes that is definitely long enough"
        ),
    )
    code = main(_base(tmp_path, "remember", "--stdin"))
    out = capsys.readouterr().out

    assert code == 0
    assert "wrote" in out


def test_remember_errors_without_text_or_stdin(tmp_path: Path, capsys) -> None:
    code = main(_base(tmp_path, "remember"))
    err = capsys.readouterr().err

    assert code == 2
    assert "provide TEXT" in err


def test_remember_respects_layer_flag(tmp_path: Path, capsys) -> None:
    main(
        _base(
            tmp_path,
            "remember",
            "a semantic note long enough to pass vstash ingest guardrail",
            "--layer",
            "semantic",
        )
    )
    capsys.readouterr()
    code = main(_base(tmp_path, "--json", "status"))
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)
    assert payload["layers"].get("semantic") == 1


# ---------------------------------------------------------------------- recall


def test_recall_finds_recent_event(tmp_path: Path, capsys) -> None:
    main(
        _base(
            tmp_path,
            "remember",
            "the kafka merchant pipeline was discussed yesterday",
        )
    )
    capsys.readouterr()
    code = main(_base(tmp_path, "recall", "kafka merchant pipeline"))
    out = capsys.readouterr().out

    assert code == 0
    assert "kafka" in out.lower()


def test_recall_empty_db_prints_no_hits(tmp_path: Path, capsys) -> None:
    code = main(_base(tmp_path, "recall", "anything"))
    out = capsys.readouterr().out
    assert code == 0
    assert "no hits" in out


def test_recall_json_emits_list_of_hits(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "a json recall canary about penguins"))
    capsys.readouterr()
    code = main(_base(tmp_path, "--json", "recall", "penguins"))
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert payload
    assert "penguins" in payload[0]["text"].lower()


# ----------------------------------------------------------------- consolidate


def test_consolidate_on_empty_db_skips_cleanly(tmp_path: Path, capsys) -> None:
    code = main(_base(tmp_path, "--json", "consolidate"))
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)
    assert payload["events_examined"] == 0
    assert payload["skipped"] is True


def test_consolidate_force_builds_semantic_fact(tmp_path: Path, capsys) -> None:
    main(
        _base(
            tmp_path,
            "remember",
            "The team picked Postgres for the new analytics project.",
        )
    )
    main(
        _base(
            tmp_path,
            "remember",
            "User chose Postgres for the analytics project this sprint.",
        )
    )
    capsys.readouterr()

    code = main(_base(tmp_path, "--json", "consolidate", "--force"))
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)
    assert payload["events_examined"] == 2
    assert payload["facts_written"] == 1
    assert payload["skipped"] is False


# --------------------------------------------------------------------- forget


def test_forget_default_never_is_noop(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "a normal event"))
    capsys.readouterr()
    code = main(_base(tmp_path, "--json", "forget"))
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)
    assert payload["tombstoned"] == []
    assert payload["decider"] == "NeverForget"


def test_forget_force_tombstones_everything(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "event one about marine biology"))
    main(_base(tmp_path, "remember", "event two about terrestrial ecology"))
    capsys.readouterr()

    code = main(_base(tmp_path, "--json", "forget", "--force"))
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert len(payload["tombstoned"]) == 2

    capsys.readouterr()
    main(_base(tmp_path, "recall", "marine biology"))
    out = capsys.readouterr().out
    assert "marine biology" not in out.lower()


def test_forget_consolidated_mode_via_cli(tmp_path: Path, capsys) -> None:
    main(
        _base(
            tmp_path,
            "remember",
            "The team picked Postgres for the new analytics project.",
        )
    )
    main(
        _base(
            tmp_path,
            "remember",
            "User chose Postgres for the analytics project this sprint.",
        )
    )
    main(_base(tmp_path, "consolidate", "--force"))
    capsys.readouterr()

    code = main(_base(tmp_path, "--json", "forget", "--decider", "consolidated"))
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert len(payload["tombstoned"]) == 2
    assert payload["decider"] == "ForgetConsolidated"


# ---------------------------------------------------------------------- audit


def test_audit_returns_decision_rows(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "an audit test event"))
    capsys.readouterr()
    code = main(_base(tmp_path, "audit", "should_remember"))
    out = capsys.readouterr().out

    assert code == 0
    assert "should_remember" in out
    assert "novel" in out


def test_audit_empty_db_prints_no_rows(tmp_path: Path, capsys) -> None:
    code = main(_base(tmp_path, "audit"))
    out = capsys.readouterr().out
    assert code == 0
    assert "no audit rows" in out


# ------------------------------------------------------------------ tombstones


def test_tombstones_after_forget(tmp_path: Path, capsys) -> None:
    main(
        _base(
            tmp_path,
            "remember",
            "a distinctive event about quasars for tombstone test",
        )
    )
    main(_base(tmp_path, "forget", "--force"))
    capsys.readouterr()

    code = main(_base(tmp_path, "tombstones", "quasars"))
    out = capsys.readouterr().out
    assert code == 0
    assert "quasars" in out.lower()


# -------------------------------------------------------------------- status


def test_status_shows_layer_counts(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "the first event about analytics and reporting"))
    main(_base(tmp_path, "remember", "the second event about release logistics for thursday"))
    capsys.readouterr()

    code = main(_base(tmp_path, "--json", "status"))
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["project"] == "test_cli"
    assert payload["total_events"] == 2
    assert payload["layers"]["episodic"] == 2


def test_status_human_output(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "a single event about the status command smoke test"))
    capsys.readouterr()
    code = main(_base(tmp_path, "status"))
    out = capsys.readouterr().out

    assert code == 0
    assert "project:" in out
    assert "test_cli" in out
    assert "episodic" in out


# ---------------------------------------------------------------------- stats


def test_stats_prints_something(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "a stats test event"))
    capsys.readouterr()
    code = main(_base(tmp_path, "stats"))
    out = capsys.readouterr().out
    assert code == 0
    assert "document" in out.lower() or "chunks" in out.lower()


def test_stats_json(tmp_path: Path, capsys) -> None:
    main(_base(tmp_path, "remember", "a stats json event"))
    capsys.readouterr()
    code = main(_base(tmp_path, "--json", "stats"))
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert "documents" in payload or "chunks" in payload


# --------------------------------------------------------------- project env


def test_env_project_picked_up(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_PROJECT", "from_env")
    code = main(
        ["--db", str(tmp_path / "e.db"), "remember", "an env test event"]
    )
    capsys.readouterr()
    assert code == 0

    code = main(["--db", str(tmp_path / "e.db"), "--json", "status"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["project"] == "from_env"


# ------------------------------------------------------------- subcommand required


def test_no_subcommand_exits_with_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
