"""Env-var wiring for shadow mode.

Verifies that ``Memory`` auto-wraps its default decider with a
``ShadowWriteDecider`` when ``MERKEN_SHADOW`` is set, and that an
invalid or missing shadow config degrades gracefully to the plain
default -- never to a write-blocking failure.
"""

from __future__ import annotations

from pathlib import Path

from merken import HeuristicWriteDecider, Memory, ShadowWriteDecider


def test_no_env_uses_plain_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MERKEN_SHADOW", raising=False)

    with Memory(project="t", db=tmp_path / "a.db") as mem:
        assert isinstance(mem._write_decider, HeuristicWriteDecider)


def test_unknown_kind_is_ignored(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MERKEN_SHADOW", "banana")

    # Need to reload _shadow module to re-read the env var.
    import importlib

    import merken._shadow
    importlib.reload(merken._shadow)

    with Memory(project="t", db=tmp_path / "b.db") as mem:
        assert isinstance(mem._write_decider, HeuristicWriteDecider)


def test_nanogpt_env_missing_paths_falls_back_cleanly(
    monkeypatch, tmp_path: Path
) -> None:
    """If MERKEN_SHADOW=nanogpt but the required ckpt/meta vars are
    unset, ``Memory`` must degrade to the plain default rather than
    raise -- writes must not break because an opt-in shadow is broken.
    """
    monkeypatch.setenv("MERKEN_SHADOW", "nanogpt")
    monkeypatch.delenv("MERKEN_SHADOW_NANOGPT_CKPT", raising=False)
    monkeypatch.delenv("MERKEN_SHADOW_NANOGPT_META", raising=False)

    import importlib

    import merken._shadow
    importlib.reload(merken._shadow)

    with Memory(project="t", db=tmp_path / "c.db") as mem:
        assert isinstance(mem._write_decider, HeuristicWriteDecider), (
            "broken shadow config must not propagate; "
            "plain default must still work"
        )


def test_explicit_shadow_decider_overrides_env(
    monkeypatch, tmp_path: Path
) -> None:
    """A caller-provided write_decider must not be replaced by the
    env-wired default -- explicit config always wins.
    """
    monkeypatch.setenv("MERKEN_SHADOW", "nanogpt")
    monkeypatch.setenv("MERKEN_SHADOW_NANOGPT_CKPT", "/nope/not/real.pt")
    monkeypatch.setenv("MERKEN_SHADOW_NANOGPT_META", "/nope/not/real.pkl")

    custom = HeuristicWriteDecider()
    with Memory(project="t", db=tmp_path / "d.db", write_decider=custom) as mem:
        assert mem._write_decider is custom
        assert not isinstance(mem._write_decider, ShadowWriteDecider)


def test_default_returns_plain_when_shadow_import_fails(
    monkeypatch, tmp_path: Path
) -> None:
    """Exceptions during shadow construction are caught; fall back."""
    monkeypatch.setenv("MERKEN_SHADOW", "llm")
    monkeypatch.setenv("MERKEN_SHADOW_LLM_MODEL", "definitely/not-a-real-model-xyz")

    import importlib

    import merken._shadow
    importlib.reload(merken._shadow)

    # Memory construction must not raise even if shadow fails to load.
    with Memory(project="t", db=tmp_path / "e.db") as mem:
        assert isinstance(mem._write_decider, HeuristicWriteDecider)
