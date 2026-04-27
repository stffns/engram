"""Diagnose why vstash retrieval didn't surface answer-session turns
for the 13 failing qids of the vstash.ask baseline (seed=44).

For each qid:
1. Ingest the haystack into a fresh vstash Memory.
2. Search top-30 with the question.
3. Check which turns (if any) from the expected answer_session_ids
   appear in top-30 and at what rank.
4. Report: "answer-session chunk present at rank K" or "not in top-30".

If answer-session chunks are present at rank 11-30 but NOT top-10,
raising top_k would fix it. If they're absent from top-30 entirely,
retrieval config needs tuning (weights, mmr_lambda, recency).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

import vstash

from experiments.retrieval.longmemeval.dataset import load_longmemeval
from experiments.retrieval.longmemeval.runner import _format_turn


FAILING_QIDS = [
    # 11 neutrals
    "gpt4_ec93e27f", "eac54adc", "80ec1f4f", "a96c20ee", "gpt4_731e37d7",
    "57f827a0", "gpt4_1e4a8aec", "370a8ff4", "gpt4_213fd887",
    "6e984302", "73d42213",
    # 2 contradicts
    "gpt4_0a05b494", "0100672e",
]


def _parse_title(title: str) -> tuple[str, str, int]:
    """Split `qid::sid::idx` title format used at ingest."""
    parts = title.split("::")
    if len(parts) >= 3:
        qid, sid, idx = parts[0], parts[1], parts[2]
        try:
            idx_int = int(idx)
        except ValueError:
            idx_int = -1
        return qid, sid, idx_int
    return (title, "", -1)


def diagnose_one(conv, top_k: int = 30) -> dict:
    out = {
        "qid": conv.question_id,
        "question": conv.question,
        "answer_session_ids": list(conv.answer_session_ids),
        "top_k_probed": top_k,
    }
    with tempfile.TemporaryDirectory(prefix="vstash_diag_") as td:
        db_path = Path(td) / "mem.db"
        mem = vstash.Memory(db=str(db_path))
        collection = "default"
        # Ingest
        n = 0
        for sid, turns in conv.haystack_sessions.items():
            for i, turn in enumerate(turns):
                mem.remember(
                    _format_turn(turn),
                    title=f"{conv.question_id}::{sid}::{i}",
                    collection=collection,
                )
                n += 1
        out["n_ingested"] = n
        # Search top_k=30
        results = mem.search(conv.question, top_k=top_k, collection=collection)
        out["n_results"] = len(results)
        # For each result, extract (rank, title, is_answer_session, score)
        details = []
        answer_ranks = []
        for rank, chunk in enumerate(results):
            title = chunk.title if hasattr(chunk, "title") else getattr(chunk, "path", "?")
            _, sid, _ = _parse_title(title)
            is_answer = sid in conv.answer_session_ids
            score = getattr(chunk, "score", None)
            details.append({
                "rank": rank + 1,
                "title": title,
                "sid": sid,
                "is_answer_session": is_answer,
                "score": score,
            })
            if is_answer:
                answer_ranks.append(rank + 1)
        out["first_answer_session_rank"] = min(answer_ranks) if answer_ranks else None
        out["all_answer_session_ranks"] = answer_ranks
        out["top_10_has_answer"] = any(r <= 10 for r in answer_ranks)
        out["top_30_has_answer"] = len(answer_ranks) > 0
        out["top_5_titles"] = [d["title"] for d in details[:5]]
    return out


def main() -> int:
    convs = load_longmemeval("longmemeval_s")
    index = {c.question_id: c for c in convs}
    print(f"{'qid':22s} type           top-10? top-30? first@rank answer-ranks@30")
    print("-" * 90)
    all_results = []
    summary = {"in_top_10": 0, "rank_11_to_30": 0, "missing_top_30": 0}
    for qid in FAILING_QIDS:
        conv = index.get(qid)
        if conv is None:
            print(f"{qid:22s} NOT FOUND")
            continue
        d = diagnose_one(conv, top_k=30)
        if d["top_10_has_answer"]:
            summary["in_top_10"] += 1
            cat = "in top-10"
        elif d["top_30_has_answer"]:
            summary["rank_11_to_30"] += 1
            cat = "rank 11-30"
        else:
            summary["missing_top_30"] += 1
            cat = "MISSING"
        first = d["first_answer_session_rank"] or "—"
        ranks = d["all_answer_session_ranks"][:5]
        print(f"{qid:22s} {conv.question_type[:14]:14s} "
              f"{'YES' if d['top_10_has_answer'] else 'no ':>3s}    "
              f"{'YES' if d['top_30_has_answer'] else 'no':>3s}    "
              f"{str(first):>10s}  {ranks}")
        all_results.append(d)

    print()
    print("=== SUMMARY ===")
    print(f"  in top-10       : {summary['in_top_10']:2d}  (current k=10 already has chunk; Builder is bottleneck)")
    print(f"  rank 11-30      : {summary['rank_11_to_30']:2d}  (raising top_k would surface)")
    print(f"  missing top-30  : {summary['missing_top_30']:2d}  (retrieval config tuning needed)")

    out_path = ENGRAM / "experiments/retrieval/longmemeval/pipeline_runs/vstash_miss_diagnostic.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
