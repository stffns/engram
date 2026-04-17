"""Export ``merken_labels`` across all projects into a training-ready JSONL.

Walks every DB in ``~/.merken/``, reads the ``merken_labels``
collection, and dumps one record per label. Output is one line per
label, ready for nanoGPT ``prepare.py`` or any other supervised
fine-tune pipeline.

Each JSONL record::

    {
      "text": "...",               # event text the oracle labeled
      "label": "DECISION" | "NOISE" | "UNCERTAIN",
      "confidence": 0.0-1.0,        # oracle confidence
      "backend": "gemini-2.0-flash" # which oracle spoke
      "project": "engram",          # source project (derived from DB filename)
      "source": "bootstrap_from_transcripts" | "bootstrap_retro_labels" | null,
      "rationale": "...",           # oracle's one-sentence reason
      "shadow_reason": "P(D)=... P(N)=...",  # v6 confidence at the disagreement
      "event_title": "label:..."    # raw vstash title, for traceability
    }

Default filters:

- drops UNCERTAIN (not useful for binary supervised training)
- drops confidence < 0.60 (low-conviction oracle calls)
- drops rows where we could not parse the JSON body

Both filters are overridable. Use ``--min-confidence 0`` and
``--include-uncertain`` for the full set.

Usage::

    # export DECISION + NOISE with default filters
    python3 -m experiments.extract_labels_to_jsonl \
        --out data/merken_labels_v7.jsonl

    # export only NOISE (rarer class, you may want to balance)
    python3 -m experiments.extract_labels_to_jsonl \
        --out data/merken_labels_noise.jsonl --label NOISE
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

MERKEN_DBS_DIR = Path.home() / ".merken"


def iter_labels_from_db(db_path: Path):
    """Yield (raw_title, raw_body_text, project) tuples from a project DB.

    Reads documents and chunks separately, then concatenates chunk
    text per document in ``seq`` order. A single label document may
    span multiple chunks if vstash's chunker split it; joining raw
    would yield partial JSON rows that ``json.loads`` would reject.

    We don't go through ``merken.Memory`` here -- this script is pure
    ETL and should not pay the fastembed load cost to copy text out.
    """
    project = db_path.stem
    try:
        con = sqlite3.connect(str(db_path))
    except sqlite3.OperationalError:
        return
    try:
        cur = con.cursor()
        # Fetch ordered (doc_id, title, chunk_text, seq) then group.
        rows = cur.execute(
            """
            SELECT d.id, d.title, c.text, c.seq, d.added_at
            FROM documents d
            JOIN chunks c ON c.doc_id = d.id
            WHERE d.collection = 'merken_labels'
            ORDER BY d.added_at, d.id, c.seq
            """
        ).fetchall()
    except sqlite3.OperationalError:
        con.close()
        return
    con.close()

    current_id: str | None = None
    current_title = ""
    current_parts: list[str] = []
    for doc_id, title, chunk_text, _seq, _added in rows:
        if doc_id != current_id:
            if current_id is not None:
                yield current_title, "".join(current_parts), project
            current_id = doc_id
            current_title = title or ""
            current_parts = []
        current_parts.append(chunk_text or "")
    if current_id is not None:
        yield current_title, "".join(current_parts), project


def parse_label_body(body: str) -> dict | None:
    """Labels are stored as pretty-printed JSON in the chunk text."""
    try:
        obj = json.loads(body)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract merken_labels across all projects into JSONL."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path (parent dir must exist).",
    )
    parser.add_argument(
        "--label",
        action="append",
        choices=["DECISION", "NOISE", "UNCERTAIN"],
        default=None,
        help="Keep only these label classes (repeatable). Default: DECISION + NOISE.",
    )
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Include UNCERTAIN rows (default: drop).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Drop oracle labels below this confidence (default: 0.60).",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Keep only labels with this `source` field (repeatable).",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=None,
        help="Keep only labels from these projects (repeatable).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted_labels: set[str]
    if args.label:
        wanted_labels = set(args.label)
    else:
        wanted_labels = {"DECISION", "NOISE"}
        if args.include_uncertain:
            wanted_labels.add("UNCERTAIN")

    wanted_sources: set[str] | None = set(args.source) if args.source else None
    wanted_projects: set[str] | None = (
        set(args.project) if args.project else None
    )

    dbs = sorted(MERKEN_DBS_DIR.glob("*.db"))
    if not dbs:
        print(f"no DBs in {MERKEN_DBS_DIR}")
        return 1

    written = 0
    label_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as out:
        for db in dbs:
            project = db.stem
            if wanted_projects and project not in wanted_projects:
                continue
            for title, body, _ in iter_labels_from_db(db):
                obj = parse_label_body(body)
                if obj is None:
                    drop_counts["unparseable_body"] += 1
                    continue
                label = obj.get("decision")
                if label not in wanted_labels:
                    drop_counts[f"label_{label}"] += 1
                    continue
                confidence = float(obj.get("confidence") or 0.0)
                if confidence < args.min_confidence:
                    drop_counts["low_confidence"] += 1
                    continue
                source = obj.get("source")
                if wanted_sources and source not in wanted_sources:
                    drop_counts[f"source_{source}"] += 1
                    continue
                text = obj.get("event_text_preview") or ""
                if not text.strip():
                    drop_counts["empty_text"] += 1
                    continue

                record = {
                    "text": text,
                    "label": label,
                    "confidence": confidence,
                    "backend": obj.get("backend"),
                    "project": project,
                    "source": source,
                    "rationale": obj.get("rationale"),
                    "shadow_reason": obj.get("shadow_reason"),
                    "event_title": title,
                }
                out.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                written += 1
                label_counts[label] += 1

    print(f"wrote {written} rows to {out_path}")
    print("label distribution:")
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v}")
    if drop_counts:
        print("dropped:")
        for k, v in sorted(drop_counts.items()):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
