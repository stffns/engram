"""Grid search for temporal_weight across all loop_quality scenarios.

Produces a table showing query_pass_rate at each weight for every
scenario.  Used to validate the empirical bar before changing defaults.

Usage:
    python -m experiments.loop_quality.temporal_grid
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from experiments.loop_quality.runner import run_scenario
from experiments.loop_quality.scenario import load_scenario

SCENARIO_DIR = Path(__file__).parent / "scenarios"

WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def main() -> None:
    scenario_paths = sorted(SCENARIO_DIR.glob("*.json"))
    scenarios = [load_scenario(p) for p in scenario_paths]

    # Header
    name_w = max(len(s.name) for s in scenarios)
    header = f"{'scenario':<{name_w}}"
    for w in WEIGHTS:
        header += f"  {w:>5.2f}"
    print(header)
    print("-" * len(header))

    with tempfile.TemporaryDirectory(prefix="merken_temporal_grid_") as td:
        for scenario in scenarios:
            row = f"{scenario.name:<{name_w}}"
            for w in WEIGHTS:
                db = Path(td) / f"{scenario.name}_w{w}.db"
                result = run_scenario(scenario, db=db, temporal_weight=w)
                row += f"  {result.query_pass_rate:>5.0%}"
            print(row)


if __name__ == "__main__":
    main()
