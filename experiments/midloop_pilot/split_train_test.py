"""Group-based train/test split for midloop_v0 by protocol_id.

Holds out ~10% of protocols ENTIRE (no case-level mixing) so the
test set contains protocols the model has never seen. This is the
harder generalization setup required by the Phase 3 plan
(notes/midloop-training-plan.md) -- stratified-within-protocol
would leak 4 of every 5 cases from a holdout protocol into train.

Deterministic via --seed (default 42).

Usage:
  python -m experiments.midloop_pilot.split_train_test
  python -m experiments.midloop_pilot.split_train_test \
      --in training_data/midloop_v0.jsonl \
      --train training_data/midloop_v0_train.jsonl \
      --test training_data/midloop_v0_test.jsonl \
      --test-protocols 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_protocol(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        pid = r.get("metadata", {}).get("protocol_id")
        if not pid:
            raise ValueError(f"row without protocol_id: case_id={r.get('case_id')}")
        groups[pid].append(r)
    return dict(groups)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def split(
    in_path: Path,
    train_path: Path,
    test_path: Path,
    test_protocols: int,
    seed: int,
) -> tuple[int, int, list[str]]:
    rows = _load(in_path)
    groups = _group_by_protocol(rows)
    protocol_ids = sorted(groups.keys())

    if test_protocols >= len(protocol_ids):
        raise ValueError(
            f"test_protocols={test_protocols} >= total protocols={len(protocol_ids)}"
        )

    rng = random.Random(seed)
    shuffled = protocol_ids[:]
    rng.shuffle(shuffled)
    test_ids = set(shuffled[:test_protocols])

    train_rows = [r for pid in protocol_ids if pid not in test_ids for r in groups[pid]]
    test_rows = [r for pid in protocol_ids if pid in test_ids for r in groups[pid]]

    _write(train_path, train_rows)
    _write(test_path, test_rows)

    return len(train_rows), len(test_rows), sorted(test_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="in_path",
        default="experiments/midloop_pilot/training_data/midloop_v0.jsonl",
    )
    parser.add_argument(
        "--train",
        default="experiments/midloop_pilot/training_data/midloop_v0_train.jsonl",
    )
    parser.add_argument(
        "--test",
        default="experiments/midloop_pilot/training_data/midloop_v0_test.jsonl",
    )
    parser.add_argument("--test-protocols", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_train, n_test, test_ids = split(
        Path(args.in_path),
        Path(args.train),
        Path(args.test),
        args.test_protocols,
        args.seed,
    )

    total = n_train + n_test
    print(f"train: {n_train} cases ({n_train / total * 100:.1f}%)")
    print(f"test:  {n_test} cases ({n_test / total * 100:.1f}%)")
    print(f"test protocols ({len(test_ids)}):")
    for pid in test_ids:
        print(f"  - {pid}")


if __name__ == "__main__":
    main()
