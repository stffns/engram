"""For a LoCoMo per-session merken-recall jsonl, take each missed
question and replay the retrieval -- so we can see what vstash
actually returned at top_k for that question, vs what the
evidence pointed at.

Output: one JSON per miss with question, gold, answer, oracle
verdict + rationale, evidence, and the top-k retrieved (title,
score, char_excerpt). That output is what a human reads to
categorize bottlenecks (retrieval miss / Builder ignored /
oracle severe / question ambiguous / etc).
"""
# ruff: noqa: I001, E402
from __future__ import annotations

import torch  # noqa: F401

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

os.environ.setdefault("VSTASH_BACKEND", "cerebras")
os.environ.setdefault("VSTASH_MODEL", "llama3.1-8b")

import vstash  # noqa: E402

from experiments.retrieval.locomo.runner import (  # noqa: E402
    Conversation,
    load_locomo,
)
from experiments.retrieval.locomo.runner_phase2 import (  # noqa: E402
    _format_turn,
)


def _ingest_session_blocks(mem, conv: Conversation):
    for session in conv.sessions:
        sid = f"session_{session.index}"
        lines = [_format_turn(t) for t in session.turns]
        text = f"[{session.date_time}]\n" + "\n".join(lines)
        mem.remember(text, title=f"{conv.sample_id}::{sid}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument(
        "--data", type=Path,
        default=Path(__file__).parent / "data" / "locomo10.json",
    )
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument(
        "--n-per-shape", type=int, default=6,
        help="Stratified sample of fails per category. 5 cats * 6 = 30.",
    )
    ap.add_argument(
        "--seed", type=int, default=44,
        help="Stratified-sample RNG seed.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.inp.open()]
    fails = [r for r in rows if r["oracle"]["verdict"] in ("contradicts", "neutral")]
    print(f"Total rows: {len(rows)}, fails: {len(fails)}", flush=True)

    # Stratified sample
    import random
    rng = random.Random(args.seed)
    by_shape: dict[str, list] = {}
    for r in fails:
        by_shape.setdefault(r["category_name"], []).append(r)
    sampled = []
    for shape, lst in sorted(by_shape.items()):
        rng.shuffle(lst)
        take = lst[:args.n_per_shape]
        sampled.extend(take)
        print(f"  {shape:<14} {len(take)}/{len(lst)}", flush=True)
    print(f"sampled: {len(sampled)}", flush=True)

    # Group by sample_id so we re-ingest each conv only once
    by_conv: dict[str, list] = {}
    for r in sampled:
        by_conv.setdefault(r["sample_id"], []).append(r)

    convs = {c.sample_id: c for c in load_locomo(args.data)}

    out_rows = []
    for sid, items in sorted(by_conv.items()):
        print(f"\n[conv {sid}] {len(items)} misses to enrich", flush=True)
        conv = convs[sid]
        with tempfile.TemporaryDirectory(prefix=f"enrich_{sid}_") as td:
            db = str(Path(td) / "mem.db")
            mem = vstash.Memory(db=db)
            try:
                _ingest_session_blocks(mem, conv)
                for r in items:
                    hits = mem.search(r["question"], top_k=args.top_k)
                    retrieved = [
                        {
                            "rank": i,
                            "title": h.title,
                            "score": getattr(h, "score", None),
                            "excerpt": h.text[:300],
                        }
                        for i, h in enumerate(hits)
                    ]
                    out_rows.append({
                        "sample_id": r["sample_id"],
                        "category": r["category_name"],
                        "question": r["question"],
                        "ground_truth": r["ground_truth"],
                        "answer": r["answer"],
                        "verdict": r["oracle"]["verdict"],
                        "rationale": r["oracle"]["rationale"],
                        "evidence_ids": r["evidence"],
                        "retrieved": retrieved,
                    })
            finally:
                mem.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[out] {args.out}", flush=True)
    print(f"rows: {len(out_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
