"""Grid search for consolidation ``embedding_threshold`` across all
loop_quality scenarios.

Built 2026-04-14 to probe whether lowering the 0.65 default recovers
the one missing es/en pair in ``bilingual_es_en_2026_04_14`` without
regressing the other five scenarios. Reports both query_pass_rate
(primary) and topic_coverage (the metric the multilingual scenario
actually moved).

Usage:
    python -m experiments.loop_quality.threshold_grid
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from experiments.loop_quality.runner import run_scenario
from experiments.loop_quality.scenario import load_scenario

SCENARIO_DIR = Path(__file__).parent / "scenarios"

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> None:
    scenario_paths = sorted(SCENARIO_DIR.glob("*.json"))
    scenarios = [load_scenario(p) for p in scenario_paths]

    name_w = max(len(s.name) for s in scenarios)

    for metric_name, metric_attr in (
        ("query_pass_rate", "query_pass_rate"),
        ("topic_coverage", "topic_coverage"),
    ):
        print(f"\n== {metric_name} ==")
        header = f"{'scenario':<{name_w}}"
        for t in THRESHOLDS:
            header += f"  {t:>5.2f}"
        print(header)
        print("-" * len(header))

        with tempfile.TemporaryDirectory(prefix=f"merken_thresh_{metric_name}_") as td:
            for scenario in scenarios:
                row = f"{scenario.name:<{name_w}}"
                for t in THRESHOLDS:
                    db = Path(td) / f"{scenario.name}_t{t}.db"
                    result = run_scenario(scenario, db=db, embedding_threshold=t)
                    value = getattr(result, metric_attr)
                    row += f"  {value:>5.0%}"
                print(row)


if __name__ == "__main__":
    main()
