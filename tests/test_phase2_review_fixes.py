"""Regression tests for PR #49 review fixes."""

from __future__ import annotations

import json


def test_step4a_locomo_qid_hashes_full_question() -> None:
    from experiments.phase2_distillation.step4a_generate_sft_data import _qid

    prefix = "What did Caroline decide after the very long shared prefix " * 2
    a = _qid({
        "source": "locomo",
        "sample_id": "conv-1",
        "question": prefix + "A",
    })
    b = _qid({
        "source": "locomo",
        "sample_id": "conv-1",
        "question": prefix + "B",
    })

    assert a.startswith("locomo:conv-1::")
    assert b.startswith("locomo:conv-1::")
    assert a != b


def test_step3_summary_excludes_errored_rows_from_quality_metrics() -> None:
    from experiments.phase2_distillation.step3_zero_shot_smoke import _summarize

    summary = _summarize([
        {
            "source": "lme",
            "question_type": "multi-session",
            "reasoning_content": "trace",
            "content": "answer",
            "wall_s": 1.0,
        },
        {
            "source": "lme",
            "question_type": "multi-session",
            "error": "URLError: connection refused",
            "reasoning_content": "",
            "content": "",
            "wall_s": 0.1,
        },
    ])

    assert summary["n_calls"] == 2
    assert summary["n_errors"] == 1
    assert summary["frac_reasoning_nonempty"] == 1.0
    shape = summary["by_lme_question_type"]["lme:multi-session"]
    assert shape["n"] == 2
    assert shape["n_ok"] == 1
    assert shape["n_errors"] == 1
    assert shape["frac_reasoning_nonempty"] == 1.0
    assert shape["n_empty_content"] == 0


def test_step4c_skips_rows_missing_messages(tmp_path, monkeypatch) -> None:
    from experiments.phase2_distillation.step4c_prepare_mlx_train import main

    src = tmp_path / "train.jsonl"
    out = tmp_path / "mlx_split"
    src.write_text(
        "\n".join([
            json.dumps({"messages": [{"role": "user", "content": "ok"}]}),
            json.dumps({"qid": "bad"}),
        ])
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "step4c",
            "--in",
            str(src),
            "--out-dir",
            str(out),
            "--frac-valid",
            "0",
            "--frac-test",
            "0",
        ],
    )

    assert main() == 0
    lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "messages": [{"role": "user", "content": "ok"}],
    }
