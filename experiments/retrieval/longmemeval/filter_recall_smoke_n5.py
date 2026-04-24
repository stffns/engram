"""Step 1 extension: filter recall over 5 seed=44 questions.

Extends ``filter_recall_smoke.py`` (qid=099778bb, 70.8%). Goal: see
whether 70.8% is an unlucky draw or the OOD average, before honoring
the STOP gate and falling back to AlwaysWrite.

Reports per-question answer-session recall AND whether the oracle
answer text appears in any kept vs dropped turn (needle-turn
recall as a secondary signal).

No LLM spend. Local CPU inference only.
"""

from __future__ import annotations

# ruff: noqa: E402
import torch  # noqa: F401  (import torch first, merken "Mistake #10")

import json
import random
import statistics
import sys
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402
from merken.classifiers.nanogpt import NanoGPTWriteDecider  # noqa: E402
from merken.policies.types import Event, WriteContext  # noqa: E402


NANOGPT = ENGRAM.parent / "nanoGPT"
V7_CKPT = NANOGPT / "out-merken-bpe-v7" / "ckpt.pt"
V7_META = NANOGPT / "data" / "merken_bpe_v7" / "meta.pkl"

SEED = 44
N_SAMPLE = 30
N_MEASURE = 5


def _needle_candidates(answer: str) -> list[str]:
    """A few surface-form candidates for locating the answer in raw text."""
    cands = {answer.strip()}
    a = answer.strip()
    if a.endswith("%") and a[:-1].replace(".", "").isdigit():
        n = a[:-1]
        cands.update({n, f"{n}%", f"{n} percent", f"{n}%%"})
    return [c for c in cands if c]


def main() -> int:
    print(f"[load] v7 decider  ckpt={V7_CKPT.exists()}  meta={V7_META.exists()}", flush=True)
    decider = NanoGPTWriteDecider(str(V7_CKPT), str(V7_META))
    ctx = WriteContext(project="longmemeval_filter_smoke_n5")

    print(f"[load] longmemeval_s, seed={SEED}, N={N_SAMPLE}", flush=True)
    convs = load_longmemeval("longmemeval_s")
    rnd = random.Random(SEED)
    sampled = rnd.sample(convs, N_SAMPLE)

    per_q: list[dict] = []

    for qi in range(N_MEASURE):
        conv = sampled[qi]
        needles = _needle_candidates(conv.answer)

        answer_turns: list[tuple[str, int, str, str]] = []
        for sid in conv.answer_session_ids:
            if sid not in conv.haystack_sessions:
                continue
            for i, t in enumerate(conv.haystack_sessions[sid]):
                answer_turns.append((sid, i, t.role, t.content))

        kept = 0
        dropped = 0
        needle_turns: list[dict] = []  # turns that contain a needle candidate

        for sid, idx, role, content in answer_turns:
            d = decider.decide(Event(text=content), ctx)
            if d.write:
                kept += 1
            else:
                dropped += 1

            hit_needle = any(cand in content for cand in needles)
            if hit_needle:
                needle_turns.append({
                    "sid": sid, "idx": idx, "role": role,
                    "write": d.write, "reason": d.reason,
                    "content_preview": content[:160].replace("\n", " "),
                })

        total = kept + dropped
        recall = kept / total if total else 0.0

        # Needle-turn recall
        n_needle = len(needle_turns)
        n_needle_kept = sum(1 for r in needle_turns if r["write"])
        needle_recall = n_needle_kept / n_needle if n_needle else None

        per_q.append({
            "idx": qi,
            "qid": conv.question_id,
            "question_type": conv.question_type,
            "question": conv.question,
            "answer": conv.answer,
            "needles": needles,
            "n_answer_turns": total,
            "n_kept": kept,
            "n_dropped": dropped,
            "answer_session_recall": recall,
            "n_needle_turns": n_needle,
            "n_needle_kept": n_needle_kept,
            "needle_recall": needle_recall,
            "needle_turns": needle_turns,
        })

        nrec = f"{needle_recall*100:.0f}%" if needle_recall is not None else "N/A"
        print(
            f"[{qi}] qid={conv.question_id}  type={conv.question_type}  "
            f"kept={kept}/{total} ({recall*100:.1f}%)  "
            f"needle={n_needle_kept}/{n_needle} ({nrec})  "
            f"ans={conv.answer!r}",
            flush=True,
        )

    # Aggregate
    recalls = [r["answer_session_recall"] for r in per_q]
    needle_recalls = [r["needle_recall"] for r in per_q if r["needle_recall"] is not None]
    n_needle_total = sum(r["n_needle_turns"] for r in per_q)
    n_needle_kept_total = sum(r["n_needle_kept"] for r in per_q)

    print("", flush=True)
    print("=== aggregate over 5 questions ===", flush=True)
    print(
        f"answer-session recall: mean={statistics.mean(recalls)*100:.1f}%  "
        f"median={statistics.median(recalls)*100:.1f}%  "
        f"min={min(recalls)*100:.1f}%  max={max(recalls)*100:.1f}%",
        flush=True,
    )
    if needle_recalls:
        print(
            f"needle-turn recall (per-q): mean={statistics.mean(needle_recalls)*100:.1f}%  "
            f"min={min(needle_recalls)*100:.1f}%  max={max(needle_recalls)*100:.1f}%",
            flush=True,
        )
    if n_needle_total:
        print(
            f"needle-turn recall (pooled): {n_needle_kept_total}/{n_needle_total} "
            f"({n_needle_kept_total/n_needle_total*100:.1f}%)",
            flush=True,
        )

    # Gate interpretation
    print("", flush=True)
    print("=== gate interpretation (answer-session recall, per the plan) ===", flush=True)
    n_pass_95 = sum(1 for r in recalls if r >= 0.95)
    n_in_caveat = sum(1 for r in recalls if 0.75 <= r < 0.95)
    n_stop = sum(1 for r in recalls if r < 0.75)
    print(
        f"  >=95% proceed      : {n_pass_95}/5",
        flush=True,
    )
    print(
        f"  75-94% caveat      : {n_in_caveat}/5",
        flush=True,
    )
    print(
        f"  <75% STOP          : {n_stop}/5",
        flush=True,
    )

    out = {
        "seed": SEED,
        "n_sample": N_SAMPLE,
        "n_measure": N_MEASURE,
        "per_question": per_q,
        "aggregate": {
            "answer_session_recall_mean": statistics.mean(recalls),
            "answer_session_recall_median": statistics.median(recalls),
            "answer_session_recall_min": min(recalls),
            "answer_session_recall_max": max(recalls),
            "needle_recall_per_q_mean": (
                statistics.mean(needle_recalls) if needle_recalls else None
            ),
            "needle_recall_pooled": (
                n_needle_kept_total / n_needle_total if n_needle_total else None
            ),
            "n_needle_kept": n_needle_kept_total,
            "n_needle_total": n_needle_total,
            "n_q_pass_95": n_pass_95,
            "n_q_caveat_75_94": n_in_caveat,
            "n_q_stop_below_75": n_stop,
        },
    }
    out_path = Path(__file__).parent / "filter_recall_seed44_n5.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
