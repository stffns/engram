"""Smoke tests for `merken-midloop-dataset` CLI.

Uses --no-embed so the suite stays offline; semantic-filter behavior
is covered by ``test_midloop_dataset.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merken.training.midloop_dataset_cli import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_cli_align_no_embed_writes_jsonl(tmp_path: Path) -> None:
    in_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [
        {"case_id": "c1", "truth": "a b c", "model_response": "a X c"},
        {"case_id": "c2", "truth": "a b c", "model_response": "a b c"},
    ])

    rc = main([
        "align", "--in", str(in_path), "--out", str(out_path),
        "--no-embed",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2
    c1 = next(r for r in rows if r["case_id"] == "c1")
    c2 = next(r for r in rows if r["case_id"] == "c2")
    assert c1["n_divergent"] == 1
    assert c1["n_interventions"] == 1  # no semantic filter -> all kept
    assert c2["n_divergent"] == 0


def test_cli_align_assigns_stable_id_if_missing(tmp_path: Path) -> None:
    in_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [
        {"truth": "the dose is 50 mg", "model_response": "the dose is 80 mg"},
    ])

    rc = main([
        "align", "--in", str(in_path), "--out", str(out_path),
        "--no-embed",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 1
    cid = rows[0]["case_id"]
    assert cid.startswith("case_"), f"unexpected stable id: {cid!r}"

    # Same input -> same id (deterministic).
    rc = main([
        "align", "--in", str(in_path), "--out", str(out_path),
        "--no-embed",
    ])
    rows2 = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert rows2[0]["case_id"] == cid


def test_cli_align_skips_malformed_lines(tmp_path: Path, capsys) -> None:
    in_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "out.jsonl"
    in_path.write_text(
        '{"truth": "a b c", "model_response": "a X c"}\n'
        'not json at all\n'
        '{"truth": "only truth"}\n'  # missing model_response
        '{"truth": "p q r", "model_response": "p Y r"}\n'
    )

    rc = main([
        "align", "--in", str(in_path), "--out", str(out_path),
        "--no-embed",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2  # only the well-formed cases land
    err = capsys.readouterr().err
    assert "skipping" in err  # warned about both bad lines


def test_cli_align_compact_drops_subseq_text(tmp_path: Path) -> None:
    in_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [
        {"case_id": "c1", "truth": "a b c", "model_response": "a X c"},
    ])
    rc = main([
        "align", "--in", str(in_path), "--out", str(out_path),
        "--no-embed", "--compact",
    ])
    assert rc == 0
    row = json.loads(out_path.read_text())
    iv = row["interventions"][0]
    assert "truth_text" not in iv
    assert "model_text" not in iv
    assert "model_token_start" in iv  # indices retained


def test_cli_align_missing_input_returns_2(tmp_path: Path, capsys) -> None:
    rc = main([
        "align", "--in", str(tmp_path / "nope.jsonl"),
        "--out", str(tmp_path / "out.jsonl"),
        "--no-embed",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_cli_unknown_subcommand_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["bogus"])
