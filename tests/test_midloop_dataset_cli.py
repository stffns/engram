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


# ----------------------------------------------------------------- generate-cases

def test_cli_generate_cases_uses_injected_llm_client(
    tmp_path: Path, monkeypatch
) -> None:
    """generate-cases CLI calls anthropic via _resolve_llm_client.

    We monkeypatch the resolver to return a fake LLMClient so the
    test never touches the real API. Verifies the wiring: protocols
    input -> CaseGenerator -> JSONL output.
    """
    in_path = tmp_path / "protocols.jsonl"
    out_path = tmp_path / "cases.jsonl"
    _write_jsonl(in_path, [
        {
            "protocol_id": "test_proto",
            "text": "amoxicillin 50 mg/kg/day for pneumonia in children",
            "metadata": {"source": "WHO_test"},
        },
    ])

    canned_response = (
        '[{"prompt": "5 yo child with cough and fever", '
        '  "truth": "amoxicillin 50 mg/kg/day"},'
        ' {"prompt": "3 yo child with productive cough", '
        '  "truth": "amoxicillin 50 mg/kg/day"}]'
    )
    fake_client = lambda system, user: canned_response  # noqa: E731

    import merken.training.midloop_dataset_cli as cli_mod
    monkeypatch.setattr(cli_mod, "_resolve_llm_client", lambda args: fake_client)

    rc = main([
        "generate-cases",
        "--in", str(in_path),
        "--out", str(out_path),
        "--n", "2",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["case_id"] == "test_proto__case_000"
    assert rows[0]["metadata"]["protocol_id"] == "test_proto"
    assert rows[0]["metadata"]["source"] == "WHO_test"


def test_cli_generate_cases_skips_malformed_protocols(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    in_path = tmp_path / "protocols.jsonl"
    out_path = tmp_path / "cases.jsonl"
    in_path.write_text(
        '{"protocol_id": "good", "text": "ok"}\n'
        'not json\n'
        '{"text": "no id here"}\n'
        '{"protocol_id": "good2", "text": "also ok"}\n'
    )

    canned = '[{"prompt": "p", "truth": "t"}]'
    fake_client = lambda system, user: canned  # noqa: E731

    import merken.training.midloop_dataset_cli as cli_mod
    monkeypatch.setattr(cli_mod, "_resolve_llm_client", lambda args: fake_client)

    rc = main([
        "generate-cases",
        "--in", str(in_path),
        "--out", str(out_path),
        "--n", "1",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    # Only "good" and "good2" survive; each emits 1 case.
    assert len(rows) == 2
    err = capsys.readouterr().err
    assert "skipping" in err


def test_cli_generate_cases_missing_input_returns_2(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import merken.training.midloop_dataset_cli as cli_mod
    monkeypatch.setattr(cli_mod, "_resolve_llm_client", lambda args: lambda s, u: "[]")
    rc = main([
        "generate-cases",
        "--in", str(tmp_path / "missing.jsonl"),
        "--out", str(tmp_path / "out.jsonl"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


# ----------------------------------------------------------------- generate-responses

def test_cli_generate_responses_uses_injected_gen_fn(
    tmp_path: Path, monkeypatch
) -> None:
    """generate-responses CLI uses _resolve_generate_fn.

    Monkeypatch keeps HF transformers + model download out of tests.
    """
    in_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "responses.jsonl"
    _write_jsonl(in_path, [
        {
            "case_id": "c1", "prompt": "what dose for kid pneumonia?",
            "truth": "50 mg/kg/day",
            "metadata": {"protocol_id": "p1"},
        },
        {
            "case_id": "c2", "prompt": "what about adult pneumonia?",
            "truth": "depends on weight",
            "metadata": {"protocol_id": "p1"},
        },
    ])

    fake_gen_fn = lambda prompt: f"answer for: {prompt[:30]}"  # noqa: E731

    import merken.training.midloop_dataset_cli as cli_mod
    monkeypatch.setattr(cli_mod, "_resolve_generate_fn", lambda args: fake_gen_fn)

    rc = main([
        "generate-responses",
        "--in", str(in_path),
        "--out", str(out_path),
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["case_id"] == "c1"
    assert rows[0]["truth"] == "50 mg/kg/day"
    assert rows[0]["model_response"].startswith("answer for")
    assert rows[0]["metadata"]["protocol_id"] == "p1"


# ----------------------------------------------------------------- build

def test_cli_build_chains_all_three_steps(tmp_path: Path, monkeypatch) -> None:
    """build orchestrator runs cases -> responses -> align end to end."""
    protocols_path = tmp_path / "protocols.jsonl"
    out_dir = tmp_path / "build_out"
    _write_jsonl(protocols_path, [
        {"protocol_id": "p1", "text": "the dose is 50 mg/kg/day"},
    ])

    canned_cases = (
        '[{"prompt": "child with pneumonia", '
        '  "truth": "the dose is 50 mg/kg/day for five days"}]'
    )
    fake_client = lambda system, user: canned_cases  # noqa: E731
    fake_gen_fn = lambda prompt: "the dose is 80 mg/kg/day for five days"  # noqa: E731

    import merken.training.midloop_dataset_cli as cli_mod
    monkeypatch.setattr(cli_mod, "_resolve_llm_client", lambda args: fake_client)
    monkeypatch.setattr(cli_mod, "_resolve_generate_fn", lambda args: fake_gen_fn)

    rc = main([
        "build",
        "--protocols", str(protocols_path),
        "--out-dir", str(out_dir),
        "--n", "1",
        "--no-embed",  # skip the real embedder for the align step
    ])
    assert rc == 0
    # All three artifacts exist.
    assert (out_dir / "cases.jsonl").exists()
    assert (out_dir / "responses.jsonl").exists()
    assert (out_dir / "aligned.jsonl").exists()

    # The aligned output should have an intervention -- 50 vs 80 differs.
    aligned = json.loads((out_dir / "aligned.jsonl").read_text())
    assert aligned["n_interventions"] >= 1
