"""Step 2 variant: brief_v1 per-session consolidation.

Original Step 2 (`brief_v1_smoke.py`) called generate_briefs on all 466
events at once. Result: LLM identified only 3 topics (Kubernetes, music
advertising, abstract art) and dropped the leadership/diversity thread
that carries the answer. Aggressive "ignore noise / only significant
topics" in the brief_v1 prompt, combined with a heterogeneous 46-topic
haystack, produced under-coverage.

This variant splits the haystack by session and runs brief_v1 once per
session. Rationale: in LongMemEval each session is already a topic-
coherent exchange, so asking "identify topics" across 1 session is a
degenerate case that should yield >=1 brief per session naturally.

Gate:
  - The two answer-sessions produce briefs that capture the
    leadership/diversity/20% fact -> Step 2 PASS, proceed to Step 3.
  - Answer-sessions produce vague or off-topic briefs -> STOP.

Cost: ~46 Cerebras calls, ~2k in + ~500 out each, ~$0.10 total,
~2-3 min wall time with a thread pool.
"""

from __future__ import annotations

# ruff: noqa: E402
import torch  # noqa: F401

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402
from merken.consolidation import generate_briefs  # noqa: E402

SEED = 44
N_SAMPLE = 30
SAMPLE_INDEX = 0  # qid=099778bb

CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"
MAX_COMPLETION_TOKENS = 1200
TEMPERATURE = 0.0
N_WORKERS = 4  # concurrent Cerebras calls; keep modest to avoid 429s


def _cerebras_synthesize(prompts: list[str]) -> str:
    from cerebras.cloud.sdk import APIStatusError, Cerebras

    client = Cerebras()
    prompt = prompts[0]
    backoffs = [2, 4, 8]
    for attempt in range(len(backoffs) + 1):
        try:
            resp = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[
                    {"role": "system", "content": "You synthesize temporal briefs from event streams."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=TEMPERATURE,
            )
            break
        except APIStatusError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = isinstance(status, int) and (status >= 500 or status == 429)
            if not transient or attempt == len(backoffs):
                raise
            time.sleep(backoffs[attempt])
    return resp.choices[0].message.content or ""


def brief_one_session(sid: str, turns, today: str) -> tuple[str, list[str], float, int]:
    events: list[tuple[str, str]] = [
        (f"{sid}:{i}", f"[{t.role}] {t.content}") for i, t in enumerate(turns)
    ]
    t0 = time.perf_counter()
    briefs = generate_briefs(events, _cerebras_synthesize, today=today)
    dt = time.perf_counter() - t0
    return sid, briefs, dt, len(events)


def main() -> int:
    print(f"[load] longmemeval_s seed={SEED} N={N_SAMPLE}", flush=True)
    convs = load_longmemeval("longmemeval_s")
    rnd = random.Random(SEED)
    sampled = rnd.sample(convs, N_SAMPLE)
    conv = sampled[SAMPLE_INDEX]

    print(
        f"[probe] qid={conv.question_id} type={conv.question_type} "
        f"n_sessions={conv.n_sessions} n_turns={conv.n_turns}",
        flush=True,
    )
    print(f"[probe] question: {conv.question}", flush=True)
    print(f"[probe] answer: {conv.answer}", flush=True)
    print(f"[probe] answer_session_ids: {conv.answer_session_ids}", flush=True)

    today = "2026-04-24"
    sessions = list(conv.haystack_sessions.items())
    print(
        f"[call] {len(sessions)} per-session briefs via {CEREBRAS_MODEL} "
        f"(workers={N_WORKERS}, temp={TEMPERATURE})",
        flush=True,
    )

    results: list[dict] = []
    answer_set = set(conv.answer_session_ids)
    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(brief_one_session, sid, turns, today) for sid, turns in sessions]
        for fut in futures:
            sid, briefs, dt, n_ev = fut.result()
            is_answer = sid in answer_set
            mark = " <-- ANSWER" if is_answer else ""
            print(
                f"  [{sid}] n_briefs={len(briefs)} n_events={n_ev} dt={dt:.1f}s{mark}",
                flush=True,
            )
            results.append({
                "sid": sid,
                "is_answer_session": is_answer,
                "n_events": n_ev,
                "n_briefs": len(briefs),
                "briefs": briefs,
                "dt": dt,
            })
    total_wall = time.perf_counter() - t_all
    total_briefs = sum(r["n_briefs"] for r in results)
    print(
        f"[done] sessions={len(results)} total_briefs={total_briefs} "
        f"total_wall={total_wall:.1f}s",
        flush=True,
    )

    # Needle probe on answer sessions
    needle_words = ["women", "leadership", "diversity", "gender", "20"]
    print("", flush=True)
    print("=== ANSWER-SESSION briefs ===", flush=True)
    pass_count = 0
    for r in results:
        if not r["is_answer_session"]:
            continue
        print(f"\n--- session {r['sid']} ({r['n_briefs']} briefs) ---", flush=True)
        for bi, brief in enumerate(r["briefs"]):
            hits = [w for w in needle_words if w.lower() in brief.lower()]
            mark = f"  HITS={hits}" if hits else ""
            print(f"\n[{r['sid']} brief {bi}]{mark}\n{brief}", flush=True)
            if hits:
                pass_count += 1

    print("", flush=True)
    print("=== all-session HEADER preview (topic coverage sanity) ===", flush=True)
    for r in results:
        for bi, brief in enumerate(r["briefs"]):
            head = brief.split("\n")[0][:120]
            mark = " <-- answer-session" if r["is_answer_session"] else ""
            print(f"  [{r['sid']} b{bi}] {head}{mark}", flush=True)

    # Gate
    if pass_count > 0:
        verdict = "PASS"
    else:
        verdict = "STOP"
    print("", flush=True)
    print(f"=== verdict: {verdict} ===", flush=True)
    print(
        f"  answer-session briefs with any needle hit: {pass_count}",
        flush=True,
    )

    out = {
        "qid": conv.question_id,
        "question_type": conv.question_type,
        "question": conv.question,
        "answer": conv.answer,
        "answer_session_ids": conv.answer_session_ids,
        "seed": SEED,
        "sample_index": SAMPLE_INDEX,
        "cerebras_model": CEREBRAS_MODEL,
        "temperature": TEMPERATURE,
        "n_workers": N_WORKERS,
        "total_wall_seconds": total_wall,
        "n_sessions": len(results),
        "total_briefs": total_briefs,
        "needle_words": needle_words,
        "n_answer_briefs_with_needle": pass_count,
        "verdict": verdict,
        "per_session": results,
    }
    out_path = Path(__file__).parent / "brief_v1_seed44_q0_per_session.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
