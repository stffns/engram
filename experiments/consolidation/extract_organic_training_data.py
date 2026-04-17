"""Extract organic DECISION samples for nanoGPT training augmentation.

Two sources, both known-DECISION (Jay's merken loop already decided
they were worth keeping):

1. `~/.merken/*.db` documents (excluding audit and tombstone
   collections).
2. `jay_vstash_2026_04_09_snapshot.json` events. Split by topic:
   4 topics -> train, 2 topics -> val.

Skips documents whose title mentions nanoGPT/merken training
experiments -- meta-content that would leak this very work into the
model.

Writes two JSON files:
- `<out>_train.json`: organic DECISIONs for training (merken dbs +
  held-in scenario topics).
- `<out>_val.json`: held-out topics from the scenario, for
  organic-recall validation.
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path

SKIP_TITLE_HINTS = (
    "nanoGPT",
    "nanoGPT-",
    "write filter",
    "borderline",
    "merken-bpe",
)

SCENARIO_PATH = (
    Path(__file__).parent.parent
    / "loop_quality"
    / "scenarios"
    / "jay_vstash_2026_04_09_snapshot.json"
)

# Topic-level split for the snapshot. Val topics stay out of training
# so organic-recall generalization is not memorization.
VAL_TOPICS = {"medlocal_clinical", "vstash_notes"}


def extract_all(merken_dir: Path) -> list[dict]:
    rows = []
    seen_titles: set[str] = set()

    for db_path in sorted(glob.glob(str(merken_dir / "*.db"))):
        db = sqlite3.connect(db_path)
        try:
            cur = db.execute(
                """
                SELECT d.title, GROUP_CONCAT(c.text, '\n\n')
                FROM documents d
                JOIN chunks c ON c.doc_id = d.id
                WHERE d.collection NOT IN ('merken_audit', 'merken_tombstones')
                GROUP BY d.id, d.title
                ORDER BY d.added_at DESC
                """
            )
            for title, text in cur.fetchall():
                if title in seen_titles:
                    continue
                if any(h.lower() in title.lower() for h in SKIP_TITLE_HINTS):
                    continue
                seen_titles.add(title)
                rows.append(
                    {
                        "title": title,
                        "text": text,
                        "source_db": Path(db_path).name,
                        "proposed_label": "DECISION",
                    }
                )
        finally:
            db.close()

    return rows


def extract_snapshot_events() -> tuple[list[dict], list[dict]]:
    """Return (train_rows, val_rows) split by topic."""
    scenario = json.loads(SCENARIO_PATH.read_text())
    train, val = [], []
    for e in scenario["events"]:
        row = {
            "title": e["id"],
            "text": e["text"],
            "source_db": "jay_vstash_snapshot",
            "proposed_label": "DECISION",
            "topic": e["topic"],
        }
        (val if e["topic"] in VAL_TOPICS else train).append(row)
    return train, val


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--merken-dir",
        type=Path,
        default=Path.home() / ".merken",
        help="Directory containing <project>.db files",
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Path prefix; writes <prefix>_train.json and <prefix>_val.json",
    )
    args = p.parse_args()

    db_rows = extract_all(args.merken_dir)
    snapshot_train, snapshot_val = extract_snapshot_events()

    train_rows = db_rows + snapshot_train
    val_rows = snapshot_val

    train_path = args.out_prefix.with_name(args.out_prefix.name + "_train.json")
    val_path = args.out_prefix.with_name(args.out_prefix.name + "_val.json")
    train_path.write_text(json.dumps(train_rows, indent=2, ensure_ascii=False))
    val_path.write_text(json.dumps(val_rows, indent=2, ensure_ascii=False))

    print(f"Train: {len(train_rows)} samples")
    print(f"  from ~/.merken/*.db: {len(db_rows)}")
    print(f"  from snapshot (held-in topics): {len(snapshot_train)}")
    print(f"Val: {len(val_rows)} samples (held-out topics: {sorted(VAL_TOPICS)})")

    by_db: dict[str, int] = {}
    for r in train_rows:
        by_db[r["source_db"]] = by_db.get(r["source_db"], 0) + 1
    print("\nTrain breakdown:")
    for db_name, cnt in sorted(by_db.items(), key=lambda x: -x[1]):
        print(f"  {db_name}: {cnt}")

    lens = sorted(len(r["text"]) for r in train_rows)
    if lens:
        mid = lens[len(lens) // 2]
        print(f"\nTrain text length: min={lens[0]} median={mid} max={lens[-1]}")

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
