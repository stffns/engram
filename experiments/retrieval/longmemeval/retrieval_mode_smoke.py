"""Validate vstash 0.33.0 retrieval_mode='vec_only' on LongMemEval.

Ingests seed=44 N=5 question haystacks into a fresh vstash Memory,
then runs the same question under three retrieval_modes and measures
R@5 on the answer sessions. A functional and performance check in one.

No LLM spend.
"""

from __future__ import annotations

# ruff: noqa: E402
import torch  # noqa: F401

import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

import vstash

from dataset import load_longmemeval  # noqa: E402
from experiments.retrieval.longmemeval.runner import (  # noqa: E402
    _format_turn as _format_turn_from_runner,
)

SEED = 44
N_SAMPLE = 30
N_MEASURE = 5
TOP_K = 5
MODES = ["hybrid", "vec_only", "fts_only"]


def _session_from_title(title: str | None) -> str | None:
    if not title or "::" not in title:
        return None
    parts = title.split("::", 2)
    if len(parts) < 2:
        return None
    return parts[1]


def main() -> int:
    print(f"[vstash] {vstash.__version__}", flush=True)
    convs = load_longmemeval("longmemeval_s")
    rnd = random.Random(SEED)
    sampled = rnd.sample(convs, N_SAMPLE)[:N_MEASURE]

    results_by_mode: dict[str, list[dict]] = {m: [] for m in MODES}
    wall_by_mode: dict[str, list[float]] = {m: [] for m in MODES}

    for qi, conv in enumerate(sampled):
        print(f"\n[{qi+1}/{N_MEASURE}] qid={conv.question_id} type={conv.question_type} n_turns={conv.n_turns}", flush=True)
        ans_set = set(conv.answer_session_ids)
        with tempfile.TemporaryDirectory(prefix="retmode_") as td:
            mem = vstash.Memory(db=str(Path(td) / "mem.db"))
            t0 = time.perf_counter()
            for sid, turns in conv.haystack_sessions.items():
                for i, t in enumerate(turns):
                    mem.remember(
                        _format_turn_from_runner(t),
                        title=f"{conv.question_id}::{sid}::{i}",
                        collection="default",
                        layer="episodic",
                    )
            ingest_s = time.perf_counter() - t0
            print(f"  ingest: {ingest_s:.1f}s  ({conv.n_turns} turns)", flush=True)

            for mode in MODES:
                t0 = time.perf_counter()
                hits = mem.search(
                    conv.question,
                    top_k=TOP_K,
                    collection="default",
                    layer="episodic",
                    retrieval_mode=mode,
                )
                dt = time.perf_counter() - t0
                hit_sids = [_session_from_title(getattr(h, "title", None)) for h in hits]
                answer_hits = [s for s in hit_sids if s in ans_set]
                r_at_k = 1 if answer_hits else 0

                results_by_mode[mode].append({
                    "qid": conv.question_id,
                    "type": conv.question_type,
                    "answer_sessions": sorted(ans_set),
                    "hit_sids": hit_sids,
                    "n_answer_hits": len(answer_hits),
                    "r_at_5": r_at_k,
                    "dt": dt,
                })
                wall_by_mode[mode].append(dt)
                print(
                    f"    {mode:10s}  hit_sids={hit_sids}  answer_hits={len(answer_hits)}/{TOP_K}  r@5={r_at_k}  ({dt*1000:.0f}ms)",
                    flush=True,
                )

    print("", flush=True)
    print("=== aggregate R@5 (answer-session in top-5) on seed=44 N=5 ===", flush=True)
    for mode in MODES:
        rs = [r["r_at_5"] for r in results_by_mode[mode]]
        hits = sum(r["n_answer_hits"] for r in results_by_mode[mode])
        walls = wall_by_mode[mode]
        print(
            f"  {mode:10s}  R@5={sum(rs)}/{len(rs)}  mean_hits={hits/len(rs):.1f}  "
            f"mean_wall={statistics.mean(walls)*1000:.0f}ms",
            flush=True,
        )

    out = Path(__file__).parent / "retrieval_mode_seed44_n5.json"
    out.write_text(json.dumps({
        "vstash_version": vstash.__version__,
        "seed": SEED,
        "n_measure": N_MEASURE,
        "top_k": TOP_K,
        "per_mode": results_by_mode,
    }, indent=2))
    print(f"\n[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
