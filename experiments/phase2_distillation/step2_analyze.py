"""Analyze a step2 stratified-capture run.

Produces per-shape markdown tables and prints sample traces
for hand-inspection on the shapes most likely to surprise:
LME ``temporal-reasoning`` and ``knowledge-update`` (the
documented teacher regression) and LoCoMo ``temporal``,
``adversarial`` (the speaker-disambiguation case).

Usage::

    python -m experiments.phase2_distillation.step2_analyze [run_dir]

If ``run_dir`` is omitted, the most recent ``step2_*`` under
``experiments/phase2_distillation/runs/`` is used.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "experiments" / "phase2_distillation" / "runs"

# Hand-inspect 1-2 of these shapes specifically -- they are the
# ones the Phase 2 plan flags as most informative for distillation
# quality. The rest get table-level stats only.
HAND_INSPECT_SHAPES = {
    "lme:temporal-reasoning",
    "lme:knowledge-update",
    "lme:multi-session",
    "locomo:temporal",
    "locomo:adversarial",
}


def _latest_run() -> Path:
    candidates = sorted(RUNS_DIR.glob("step2_*"))
    if not candidates:
        sys.exit("no step2_* run dirs found")
    return candidates[-1]


def _per_shape(rows: list[dict], key: str, prefix: str) -> list[dict]:
    groups: dict[str, list] = {}
    for r in rows:
        shape = r.get(key)
        if shape:
            groups.setdefault(f"{prefix}:{shape}", []).append(r)
    out = []
    for shape, group in sorted(groups.items()):
        n = len(group)
        n_with_text = sum(
            1 for r in group if (r.get("reasoning") or "").strip()
        )
        r_lens = [len(r.get("reasoning") or "") for r in group]
        c_lens = [len(r.get("content") or "") for r in group]
        n_burned = sum(
            1
            for r in group
            if not (r.get("content") or "").strip()
            and (r.get("reasoning") or "").strip()
        )
        n_errors = sum(1 for r in group if r.get("error"))
        out.append({
            "shape": shape,
            "n": n,
            "n_reasoning": n_with_text,
            "frac": n_with_text / n if n else 0.0,
            "rmedian": int(statistics.median(r_lens)) if r_lens else 0,
            "rmin": min(r_lens) if r_lens else 0,
            "rmax": max(r_lens) if r_lens else 0,
            "cmedian": int(statistics.median(c_lens)) if c_lens else 0,
            "n_burned": n_burned,
            "n_errors": n_errors,
        })
    return out


def _print_table(rows: list[dict]) -> None:
    print("| shape | n | reasoning rate | r chars (min/med/max) | content med | burned | errors |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['shape']} | {r['n']} | "
            f"{r['n_reasoning']}/{r['n']} = {r['frac']:.0%} | "
            f"{r['rmin']}/{r['rmedian']}/{r['rmax']} | "
            f"{r['cmedian']} | {r['n_burned']} | {r['n_errors']} |"
        )


def _excerpt(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + " [...]"


def _sample_for_shape(rows: list[dict], shape: str, n: int = 1) -> list[dict]:
    src, _, key = shape.partition(":")
    field = "category_name" if src == "locomo" else "question_type"
    matches = [r for r in rows if r.get("source") == src and r.get(field) == key]
    return matches[:n]


def main() -> int:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = _latest_run()
    print(f"# Step 2 analysis: {run_dir.name}\n")

    rows_path = run_dir / "rows.jsonl"
    if not rows_path.exists():
        sys.exit(f"missing {rows_path}")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    print(f"rows: {len(rows)}\n")

    locomo_rows = [r for r in rows if r.get("source") == "locomo"]
    lme_rows = [r for r in rows if r.get("source") == "lme"]
    locomo_stats = _per_shape(locomo_rows, "category_name", "locomo")
    lme_stats = _per_shape(lme_rows, "question_type", "lme")

    print("## LoCoMo by category\n")
    _print_table(locomo_stats)
    print()
    print("## LME by question_type\n")
    _print_table(lme_stats)
    print()

    print("## Hand-inspect samples\n")
    for shape in sorted(HAND_INSPECT_SHAPES):
        sample = _sample_for_shape(rows, shape, n=1)
        if not sample:
            continue
        r = sample[0]
        print(f"### {shape}\n")
        print(f"**Q:** {r.get('question')}")
        print(f"**GT:** {r.get('ground_truth')}\n")
        print(f"**Content ({len(r.get('content') or '')} chars):**\n")
        print(_excerpt(r.get('content') or '', 400))
        print()
        print(f"**Reasoning ({len(r.get('reasoning') or '')} chars):**\n")
        print(_excerpt(r.get('reasoning') or '', 800))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
