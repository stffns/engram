# ruff: noqa: I001, E402
"""H13 + "audit 100% scores" -- check every held-out scenario for
training-data contamination.

Jay's second paper review caught that `jay_vstash_2026_04_09_snapshot`
has 13/20 events also present in `/tmp/organic_train.json` (loaded
into v7 training as DECISION:organic:v1). That invalidates the 100%
score on that scenario.

This script audits EVERY held-out scenario referenced in
`experiments/eval_v7_vs_v6.py` against EVERY training source listed
in `nanoGPT/data/merken_bpe_v7/prepare.py`:

  - /tmp/organic_train.json (DECISION:organic:v1)
  - experiments/data/markdown_noise_v6.json (NOISE)
  - merken_bpe/borderline_noise.json if found (NOISE)
  - knowledge_update{,_hard,_20topics,_50topics}.json (used both as
    training scenario AND sometimes as held-out; flag direct overlap)
  - data/merken_labels_v7.jsonl (DECISION:transcript:v1 contributes
    1026 entries from the bootstrap)

Output: per-scenario % of events that are also in some training set,
plus the list of overlapping texts so a cleaned scenario file can
be derived.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

SCENARIOS = {
    "markdown_tables_held_out": REPO / "experiments/loop_quality/scenarios/markdown_tables_held_out.json",
    "organic_val_held_out": REPO / "experiments/loop_quality/scenarios/organic_val_held_out.json",
    "jay_vstash_snapshot": REPO / "experiments/loop_quality/scenarios/jay_vstash_2026_04_09_snapshot.json",
    "knowledge_update_50t": REPO / "experiments/loop_quality/scenarios/knowledge_update_50topics.json",
    "analytics_project": REPO / "experiments/loop_quality/scenarios/analytics_project.json",
    "session_2026_04_09": REPO / "experiments/loop_quality/scenarios/session_2026_04_09.json",
    "bilingual_es_en_2026_04_14": REPO / "experiments/loop_quality/scenarios/bilingual_es_en_2026_04_14.json",
    "noisy_agent_stream": REPO / "experiments/loop_quality/scenarios/noisy_agent_stream.json",
    "knowledge_update": REPO / "experiments/loop_quality/scenarios/knowledge_update.json",
    "knowledge_update_hard": REPO / "experiments/loop_quality/scenarios/knowledge_update_hard.json",
    "knowledge_update_20topics": REPO / "experiments/loop_quality/scenarios/knowledge_update_20topics.json",
}

TRAINING_SOURCES = {
    # organic_train location mirrors nanoGPT/data/merken_bpe_v7/prepare.py
    # (hardcoded to /tmp/organic_train.json there). Override via env.
    "organic_train": Path(
        os.environ.get("MERKEN_ORGANIC_TRAIN_JSON", "/tmp/organic_train.json")
    ),
    "markdown_noise_v6": REPO / "experiments/data/markdown_noise_v6.json",
    "borderline_noise": REPO / "merken" / "borderline_noise.json",  # may not exist
    # v7 training scenarios are themselves both training and benchmark:
    "knowledge_update": REPO / "experiments/loop_quality/scenarios/knowledge_update.json",
    "knowledge_update_hard": REPO / "experiments/loop_quality/scenarios/knowledge_update_hard.json",
    "knowledge_update_20topics": REPO / "experiments/loop_quality/scenarios/knowledge_update_20topics.json",
    "knowledge_update_50topics": REPO / "experiments/loop_quality/scenarios/knowledge_update_50topics.json",
    "merken_labels_v7_jsonl": REPO / "data" / "merken_labels_v7.jsonl",
}


def load_scenario_texts(path: Path) -> list[str]:
    """Return stripped event texts from a scenario JSON."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e.get("text", "").strip() for e in data.get("events", []) if e.get("text")]


def load_training_texts(name: str, path: Path) -> set[str]:
    """Return set of stripped training texts from various source formats."""
    if not path.exists():
        return set()
    if name == "organic_train" or name == "markdown_noise_v6" or name == "borderline_noise":
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return set()
        return {r.get("text", "").strip() for r in rows if r.get("text")}
    if name.startswith("knowledge_update"):
        data = json.loads(path.read_text(encoding="utf-8"))
        return {e.get("text", "").strip() for e in data.get("events", []) if e.get("text")}
    if name == "merken_labels_v7_jsonl":
        out = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = (r.get("event_text_preview") or r.get("text") or "").strip()
                if t:
                    out.add(t)
        return out
    return set()


def main() -> int:
    training_sets: dict[str, set[str]] = {}
    for name, path in TRAINING_SOURCES.items():
        s = load_training_texts(name, path)
        training_sets[name] = s
        print(f"training source {name}: {len(s)} unique texts "
              f"({'FOUND' if s else 'missing'} at {path})")

    print(f"\n{'scenario':<30} {'n':>4} {'overlap':>8} {'pct':>5}   overlap_sources")
    print("-" * 100)
    audit = {}
    for name, path in SCENARIOS.items():
        texts = load_scenario_texts(path)
        if not texts:
            print(f"{name:<30} MISSING at {path}")
            continue
        # self-overlap (knowledge_update scenarios are in TRAINING_SOURCES too
        # via prepare.py's synthetic loader) -- we report these as EXPECTED
        # contamination, not a bug, so the user knows.
        per_source_overlap = {}
        union_overlap_idx: set[int] = set()
        for src, tset in training_sets.items():
            if not tset:
                continue
            idxs = [i for i, t in enumerate(texts) if t in tset]
            if idxs:
                per_source_overlap[src] = idxs
                union_overlap_idx.update(idxs)
        n = len(texts)
        ov = len(union_overlap_idx)
        pct = ov / n * 100 if n else 0.0
        src_str = ", ".join(f"{s}({len(idxs)})" for s, idxs in per_source_overlap.items())
        print(f"{name:<30} {n:>4} {ov:>8} {pct:>4.1f}%   {src_str}")
        audit[name] = {
            "n": n,
            "overlap": ov,
            "pct": pct,
            "per_source": {s: idxs for s, idxs in per_source_overlap.items()},
            "clean_idx": [i for i in range(n) if i not in union_overlap_idx],
        }

    # Also write cleaned scenario files for anything with contamination
    # in a held-out scenario (i.e. not the knowledge_update scenarios
    # which are training on purpose).
    HELD_OUT = {
        "markdown_tables_held_out", "organic_val_held_out",
        "jay_vstash_snapshot", "bilingual_es_en_2026_04_14",
        "analytics_project", "session_2026_04_09",
        "noisy_agent_stream",
    }
    print(f"\n=== Writing cleaned held-out scenarios ===")
    for name, info in audit.items():
        if name not in HELD_OUT:
            continue
        if info["overlap"] == 0:
            continue
        source_path = SCENARIOS[name]
        data = json.loads(source_path.read_text(encoding="utf-8"))
        events = data.get("events", [])
        clean_events = [events[i] for i in info["clean_idx"]]
        # Write to a new filename so original scenario stays untouched.
        # Also bump the `name` field so the loop_quality runner doesn't
        # collide with the original's SQLite DB path.
        out_path = source_path.parent / f"{source_path.stem}_decontam.json"
        out_data = {
            **data,
            "name": f"{data.get('name', source_path.stem)}_decontam",
            "description": (
                f"DECONTAMINATED subset of {data.get('name', source_path.stem)}. "
                f"Original had {len(events)} events; "
                f"{len(events) - len(clean_events)} of them were also in a "
                f"training source loaded by prepare.py. "
                f"Kept {len(clean_events)} non-contaminated events for honest "
                f"held-out evaluation. Derived by "
                f"experiments/contamination_audit.py."
            ),
            "events": clean_events,
        }
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  {name}: wrote {out_path.name} "
              f"({len(clean_events)}/{len(events)} events kept)")

    # Save audit report
    out = REPO / "experiments" / "nanogpt" / "contamination_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull audit saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
