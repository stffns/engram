"""One-shot probe: run the bilingual scenario under two different embedders
to confirm whether the 83% topic_coverage ceiling in the default run is
caused by the embedder (BAAI/bge-small-en-v1.5) rather than the loop.

Built 2026-04-14 as the follow-up to ``threshold_grid.py``, which showed
that no value of ``embedding_threshold`` between 0.45 and 0.70 recovered
the missing es/en topic. That rules out the threshold knob and points
at the embedder — this probe confirms or refutes that.

Usage:
    python -m experiments.loop_quality.bilingual_embedder_probe
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from experiments.loop_quality.runner import run_scenario
from experiments.loop_quality.scenario import load_scenario

SCENARIO = Path(__file__).parent / "scenarios" / "bilingual_es_en_2026_04_14.json"

EMBEDDERS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def _write_vstash_toml(tmpdir: Path, model: str) -> Path:
    """Write a minimal vstash.toml pinning the embedding model.

    vstash.Memory calls ``load_config()`` on construction, which honors
    the ``VSTASH_CONFIG`` env var. Pointing that at a file we control
    is the cleanest way to force a specific embedder without monkey-
    patching internal state.
    """
    toml_path = tmpdir / "vstash.toml"
    toml_path.write_text(
        f'[embeddings]\nmodel = "{model}"\nbackend = "auto"\n',
        encoding="utf-8",
    )
    return toml_path


def main() -> None:
    scenario = load_scenario(SCENARIO)
    print(f"# scenario: {scenario.name}")
    print(f"# events: {len(scenario.events)} across {len(scenario.topics)} topics\n")

    print(f"{'embedder':<65}  {'pass':>5}  {'cov':>5}  {'facts':>5}")
    print("-" * 90)

    for model in EMBEDDERS:
        with tempfile.TemporaryDirectory(prefix="merken_bi_probe_") as td:
            td_path = Path(td)
            toml = _write_vstash_toml(td_path, model)
            prev = os.environ.get("VSTASH_CONFIG")
            os.environ["VSTASH_CONFIG"] = str(toml)
            try:
                db = td_path / "run.db"
                result = run_scenario(scenario, db=db)
            finally:
                if prev is None:
                    os.environ.pop("VSTASH_CONFIG", None)
                else:
                    os.environ["VSTASH_CONFIG"] = prev

            short = model.split("/")[-1]
            print(
                f"{short:<65}  "
                f"{result.query_pass_rate:>5.0%}  "
                f"{result.topic_coverage:>5.0%}  "
                f"{result.facts_written:>5d}"
            )


if __name__ == "__main__":
    main()
