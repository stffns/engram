"""Post-hoc retrieval diagnosis for a vstash_ask run.

For every qid in a ``vstash_ask_seed*.jsonl`` file, re-ingest the same
haystack into a fresh vstash.Memory, then call ``miss_analysis`` with
the chunks from the gold ``answer_session_ids`` as expected targets.
Records the best rank achieved across all gold chunks, the pipeline
stage that dropped the chunk (if any), and vstash's suggestions.

Use this to separate two distinct failure modes:

    retrieval_ceiling  -> the answer chunk IS in the top-k; any loss is
                          the Builder's (prompt / reasoning).
    retrieval_miss     -> the answer chunk is NOT in the top-k; the
                          Builder cannot have gotten it right.

Usage:
    python3 experiments/retrieval/longmemeval/run_miss_analysis.py \\
        --input experiments/retrieval/longmemeval/pipeline_runs/vstash_ask_seed42_n30_*.jsonl \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

# miss_analysis does not call the inference backend; set to something
# cheap and safe so vstash does not try to reach Cerebras here.
os.environ.setdefault("VSTASH_BACKEND", "cerebras")
os.environ.setdefault("VSTASH_MODEL", "llama3.1-8b")

import vstash  # noqa: E402

from experiments.retrieval.longmemeval.dataset import load_longmemeval  # noqa: E402
from experiments.retrieval.longmemeval.runner import _format_turn  # noqa: E402


def _ingest(mem: vstash.Memory, conv, collection: str) -> int:
    n = 0
    for sid, turns in conv.haystack_sessions.items():
        for i, turn in enumerate(turns):
            mem.remember(
                _format_turn(turn),
                title=f"{conv.question_id}::{sid}::{i}",
                collection=collection,
            )
            n += 1
    return n


def _gold_paths(conv) -> list[str]:
    """Paths of every turn belonging to the gold answer session(s)."""
    paths: list[str] = []
    for sid in conv.answer_session_ids:
        turns = conv.haystack_sessions.get(sid, [])
        for i in range(len(turns)):
            paths.append(f"text://{conv.question_id}::{sid}::{i}")
    return paths


def _diagnose(conv, top_k: int) -> dict:
    """Re-ingest and run miss_analysis against every gold chunk.

    Returns a dict with best_rank (across gold chunks), appeared (bool),
    per_chunk traces, and the actual top-k list from the first call.
    """
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="miss_") as td:
        db_path = Path(td) / "mem.db"
        mem = vstash.Memory(db=str(db_path))
        collection = "default"
        n_ing = _ingest(mem, conv, collection)

        gold = _gold_paths(conv)
        if not gold:
            return {
                "qid": conv.question_id,
                "error": "no answer_session_ids resolved to ingested turns",
                "n_ingested": n_ing,
            }

        per_chunk: list[dict] = []
        best_rank: int | None = None
        best_appeared = False
        actual_top_k: list[dict] | None = None
        dropped_at_counter: Counter = Counter()
        suggestions_union: set[str] = set()

        for path in gold:
            try:
                ma = mem.miss_analysis(
                    conv.question,
                    expected_path=path,
                    top_k=top_k,
                    collection=collection,
                )
            except Exception as exc:  # noqa: BLE001
                per_chunk.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
                continue

            if actual_top_k is None:
                actual_top_k = [
                    {"rank": r.rank, "path": r.path, "title": r.title, "score": r.score}
                    for r in ma.actual_top_k
                ]
            per_chunk.append({
                "path": path,
                "appeared": ma.appeared_in_results,
                "final_rank": ma.final_rank,
                "dropped_at": ma.dropped_at,
                "suggestions": list(ma.suggestions),
            })
            if ma.dropped_at:
                dropped_at_counter[ma.dropped_at] += 1
            for s in ma.suggestions:
                suggestions_union.add(s)
            if ma.appeared_in_results:
                best_appeared = True
                if best_rank is None or (ma.final_rank is not None and ma.final_rank < best_rank):
                    best_rank = ma.final_rank

        return {
            "qid": conv.question_id,
            "n_ingested": n_ing,
            "n_gold_chunks": len(gold),
            "best_appeared": best_appeared,
            "best_rank": best_rank,
            "dropped_at_counts": dict(dropped_at_counter),
            "suggestions": sorted(suggestions_union),
            "per_chunk": per_chunk,
            "actual_top_k": actual_top_k or [],
            "wall_s": time.perf_counter() - t0,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Glob or path to one or more vstash_ask_*.jsonl files")
    ap.add_argument("--subset", default="longmemeval_s")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--only-failing", action="store_true",
                    help="Diagnose only qids where verdict is not in "
                         "{supports, partial} (default: diagnose ALL qids, "
                         "so supports rows can verify retrieval was clean)")
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/retrieval/longmemeval/pipeline_runs"))
    args = ap.parse_args()

    inputs = sorted(glob.glob(args.input)) if any(c in args.input for c in "*?[") \
        else [args.input]
    inputs = [p for p in inputs if os.path.isfile(p)]
    if not inputs:
        print(f"No input files matched: {args.input}", file=sys.stderr)
        return 2
    print(f"[inputs] {len(inputs)} file(s)", flush=True)

    convs = load_longmemeval(args.subset)
    index = {c.question_id: c for c in convs}

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for in_path in inputs:
        # Read rows
        rows: list[dict] = []
        with open(in_path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not rows:
            print(f"[skip] {in_path}: no valid rows", flush=True)
            continue

        base = Path(in_path).stem
        out_path = args.out / f"miss_{base}_{ts}.jsonl"
        print(f"\n[process] {in_path}", flush=True)
        print(f"[out]     {out_path}", flush=True)
        print(f"[rows]    {len(rows)} input rows", flush=True)

        counters = Counter()
        diag_counts = Counter()
        dropped_hist: Counter = Counter()
        processed = 0
        t0 = time.perf_counter()
        with out_path.open("w") as fout:
            for i, row in enumerate(rows):
                qid = row.get("qid")
                verdict = (row.get("oracle") or {}).get("verdict")
                counters[verdict] += 1

                failing = verdict not in ("supports", "partial")
                if args.only_failing and not failing:
                    continue
                conv = index.get(qid)
                if conv is None:
                    print(f"  [{i+1}/{len(rows)}] qid={qid} -> not in {args.subset}",
                          flush=True)
                    continue
                try:
                    diag = _diagnose(conv, args.top_k)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    diag = {"qid": qid, "error": f"{type(exc).__name__}: {exc}"}

                diag["verdict"] = verdict
                diag["question_type"] = conv.question_type
                fout.write(json.dumps(diag, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if diag.get("best_appeared"):
                    diag_counts["appeared"] += 1
                elif "error" in diag:
                    diag_counts["error"] += 1
                else:
                    diag_counts["missed"] += 1
                for k, v in (diag.get("dropped_at_counts") or {}).items():
                    dropped_hist[k] += v

                tag = "appeared" if diag.get("best_appeared") else "MISSED"
                rank = diag.get("best_rank")
                print(f"  [{i+1}/{len(rows)}] qid={qid} verdict={verdict} "
                      f"-> {tag} rank={rank} wall={diag.get('wall_s', 0):.2f}s",
                      flush=True)

        # Summary for this input file
        print(f"\n=== SUMMARY for {Path(in_path).name} ===", flush=True)
        print(f"  total rows        : {len(rows)}", flush=True)
        print(f"  verdicts          : {dict(counters)}", flush=True)
        print(f"  diagnosed         : {processed}", flush=True)
        print(f"  retrieval-appeared: {diag_counts['appeared']}", flush=True)
        print(f"  retrieval-missed  : {diag_counts['missed']}", flush=True)
        if diag_counts["error"]:
            print(f"  diagnostic errors : {diag_counts['error']}", flush=True)
        if dropped_hist:
            print(f"  dropped_at hist   : {dict(dropped_hist)}", flush=True)
        print(f"  wall              : {time.perf_counter() - t0:.1f}s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
