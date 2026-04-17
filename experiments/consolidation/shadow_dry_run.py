# ruff: noqa: I001, E402
"""Shadow-mode dry run: dimension the disagreement rate.

Runs HeuristicWriteDecider + NanoGPTWriteDecider (BPE v4) over a mixed
set of events (organic docs from ~/.merken/*.db plus every loop-quality
scenario on disk) and reports the four agree/disagree buckets:

  both write   -- primary says keep, shadow says keep   (low signal)
  both skip    -- primary says drop, shadow says drop   (high-signal negatives)
  primary-write/shadow-skip -- shadow flags as noise    (review target)
  primary-skip/shadow-write -- shadow rescues          (rarer; also interesting)

The point is to size shadow mode honestly BEFORE committing to a
review cadence: if we see 5 disagreements per 100 events on realistic
content, a 200-sample target is ~40 days of typical use; if we see 50,
it's a week. The answer shapes whether oracle-based labeling is
urgent or optional.

No Gemini calls, no vstash writes: pure classifier comparison.
"""

from __future__ import annotations

# Torch must load before any merken import (Mistake #10).
import torch  # noqa: F401

import argparse
import glob
import json
import sqlite3
from collections import Counter
from pathlib import Path

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies import Event, HeuristicWriteDecider, WriteContext

REPO_ROOT = Path(__file__).parent.parent.parent
NANOGPT_DIR = REPO_ROOT.parent / "nanoGPT"
SCENARIO_DIR = REPO_ROOT / "experiments" / "loop_quality" / "scenarios"


def load_merken_events(merken_dir: Path) -> list[tuple[str, str, str]]:
    """Return (source, title, text) for every real doc in local stores."""
    rows: list[tuple[str, str, str]] = []
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
                """
            )
            for title, text in cur.fetchall():
                rows.append((f"merken:{Path(db_path).stem}", title or "", text or ""))
        except sqlite3.OperationalError:
            continue
        finally:
            db.close()
    return rows


def load_scenario_events(scenario_dir: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(scenario_dir.glob("*.json")):
        try:
            scenario = json.loads(path.read_text())
        except Exception:
            continue
        events = scenario.get("events") or []
        for e in events:
            rows.append((
                f"scenario:{path.stem}",
                e.get("id", ""),
                e.get("text", ""),
            ))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--merken-dir", type=Path, default=Path.home() / ".merken",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=NANOGPT_DIR / "out-merken-bpe" / "ckpt.pt",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=NANOGPT_DIR / "data" / "merken_bpe" / "meta.pkl",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--per-source-breakdown",
        action="store_true",
        help="print counts per source",
    )
    args = p.parse_args()

    events = load_merken_events(args.merken_dir) + load_scenario_events(SCENARIO_DIR)
    print(f"Loaded {len(events)} events (merken dbs + scenarios)")

    shadow = NanoGPTWriteDecider(
        args.ckpt, args.meta, confidence_threshold=args.threshold
    )
    ctx = WriteContext(project="dryrun")

    bucket: Counter = Counter()
    per_source: dict[str, Counter] = {}

    # Fresh HeuristicWriteDecider per source so cross-source text
    # collisions (e.g. organic_val events also appear in
    # jay_vstash_snapshot) do not show up as fake dup_exact skips.
    # Within a single source, dedup still applies and is realistic.
    primary_per_source: dict[str, HeuristicWriteDecider] = {}

    for source, title, text in events:
        primary = primary_per_source.setdefault(source, HeuristicWriteDecider())
        ev = Event(text=text, title=title or None)
        p_d = primary.decide(ev, ctx)
        s_d = shadow.decide(ev, ctx)
        key = (
            ("write" if p_d.write else "skip"),
            ("write" if s_d.write else "skip"),
        )
        bucket[key] += 1
        per_source.setdefault(source, Counter())[key] += 1

    def pct(n: int) -> str:
        return f"{(100 * n / len(events)):.1f}%" if events else "0.0%"

    both_write = bucket[("write", "write")]
    both_skip = bucket[("skip", "skip")]
    disagree_flag_skip = bucket[("write", "skip")]   # shadow thinks noise
    disagree_flag_write = bucket[("skip", "write")]  # shadow thinks decision

    print()
    print(f"{'bucket':<32} {'count':>6}  {'pct':>6}")
    print("-" * 52)
    print(f"{'both write':<32} {both_write:>6}  {pct(both_write):>6}")
    print(f"{'both skip':<32} {both_skip:>6}  {pct(both_skip):>6}")
    print(
        f"{'primary=write shadow=skip':<32} {disagree_flag_skip:>6}  "
        f"{pct(disagree_flag_skip):>6}"
    )
    print(
        f"{'primary=skip shadow=write':<32} {disagree_flag_write:>6}  "
        f"{pct(disagree_flag_write):>6}"
    )
    print("-" * 52)
    total_disagree = disagree_flag_skip + disagree_flag_write
    print(f"{'total disagreements':<32} {total_disagree:>6}  {pct(total_disagree):>6}")
    print()

    # Sizing: how many events to see N disagreements?
    rate = total_disagree / len(events) if events else 0
    if rate > 0:
        for target in (50, 100, 200):
            needed = int(target / rate) if rate > 0 else 0
            print(f"~{needed} events to accumulate {target} disagreements")
    else:
        print("no disagreements observed; shadow tells you nothing on this sample")

    if args.per_source_breakdown:
        print("\nPer-source:")
        for source, counts in sorted(per_source.items()):
            total = sum(counts.values())
            dis = counts[("write", "skip")] + counts[("skip", "write")]
            dis_pct = f"{100 * dis / total:.1f}%" if total else "0.0%"
            print(f"  {source:<40} n={total:>4}  disagree={dis:>3} ({dis_pct})")


if __name__ == "__main__":
    main()
