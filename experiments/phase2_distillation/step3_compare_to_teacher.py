"""Side-by-side comparison of step3 (student) vs step2 (teacher).

The step2 stratified picks (5 per shape) and step3 picks (first 2
per shape) overlap by construction -- step3 slices step2's first 2.
So joining on (source, question) recovers per-question diffs.

Outputs:
  - per-shape table comparing teacher vs student medians.
  - a small markdown excerpt for one row per LoCoMo category and
    one per LME question_type, showing both reasoning lengths and
    a content head.

Usage::

    python -m experiments.phase2_distillation.step3_compare_to_teacher \\
        [step3_run_dir] [step2_run_dir]

If args omitted, picks the most recent step3_* and step2_* runs.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "experiments" / "phase2_distillation" / "runs"


def _latest(prefix: str) -> Path:
    candidates = sorted(RUNS_DIR.glob(f"{prefix}_*"))
    if not candidates:
        sys.exit(f"no {prefix}_* run dirs found")
    return candidates[-1]


def _load(run_dir: Path) -> list[dict]:
    p = run_dir / "rows.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _key(row: dict) -> tuple:
    return (row.get("source"), row.get("question"))


def _shape_of(row: dict) -> str:
    src = row.get("source")
    if src == "locomo":
        return f"locomo:{row.get('category_name')}"
    return f"lme:{row.get('question_type')}"


def _student_reasoning(row: dict) -> str:
    # student uses reasoning_content (LM Studio)
    return row.get("reasoning_content") or ""


def _teacher_reasoning(row: dict) -> str:
    # teacher uses reasoning (Cerebras)
    return row.get("reasoning") or ""


def main() -> int:
    step3_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest("step3")
    step2_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else _latest("step2")
    print(f"# Step 3 vs Step 2 comparison\n")
    print(f"- student run: {step3_dir.name}")
    print(f"- teacher run: {step2_dir.name}\n")

    student_rows = _load(step3_dir)
    teacher_rows = _load(step2_dir)
    teacher_by_key = {_key(r): r for r in teacher_rows}

    joined = []
    unmatched = 0
    for sr in student_rows:
        if sr.get("error"):
            continue
        tr = teacher_by_key.get(_key(sr))
        if tr is None:
            unmatched += 1
            continue
        joined.append((sr, tr))
    print(f"- joined: {len(joined)} rows (student errors / unmatched: "
          f"{len(student_rows) - len(joined) - unmatched} / {unmatched})\n")

    # Per-shape aggregates
    by_shape: dict[str, list] = {}
    for sr, tr in joined:
        shape = _shape_of(sr)
        by_shape.setdefault(shape, []).append((sr, tr))

    print("## Per-shape medians\n")
    print("| shape | n | student reasoning med | teacher reasoning med | ratio | student wall_s med |")
    print("|---|---|---|---|---|---|")
    for shape, pairs in sorted(by_shape.items()):
        srs = [s for s, _ in pairs]
        trs = [t for _, t in pairs]
        s_r_lens = [len(_student_reasoning(s)) for s in srs]
        t_r_lens = [len(_teacher_reasoning(t)) for t in trs]
        s_walls = [s.get("wall_s", 0) or 0 for s in srs]
        s_med = statistics.median(s_r_lens) if s_r_lens else 0
        t_med = statistics.median(t_r_lens) if t_r_lens else 0
        ratio = (s_med / t_med) if t_med else float("inf")
        print(
            f"| {shape} | {len(pairs)} | {int(s_med)} | {int(t_med)} | {ratio:.1f}x | "
            f"{statistics.median(s_walls):.0f}s |"
        )
    print()

    # One sample per shape
    print("## Hand-inspect (1 per shape)\n")
    for shape, pairs in sorted(by_shape.items()):
        sr, tr = pairs[0]
        print(f"### {shape}\n")
        print(f"**Q:** {sr.get('question')}")
        print(f"**GT:** {sr.get('ground_truth')}\n")
        s_c = sr.get("content") or ""
        t_c = tr.get("answer") or tr.get("content") or ""
        s_r = _student_reasoning(sr)
        t_r = _teacher_reasoning(tr)
        print(
            f"**Lengths:** student reasoning {len(s_r)}c / content {len(s_c)}c | "
            f"teacher reasoning {len(t_r)}c / content {len(t_c)}c\n"
        )
        print(f"**Student content head:**\n")
        print((s_c or "(empty)")[:300])
        print()
        print(f"**Teacher content head:**\n")
        print((t_c or "(empty)")[:300])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
