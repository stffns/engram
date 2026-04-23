"""Step 1 of the merken full-pipeline plan (notes/merken-full-pipeline-longmemeval.md).

Measure nanoGPT v7 write-filter recall on the answer-session turns of the
first seed=44 question. Gate: >=95% proceeds, 75-94% proceeds w/ caveat,
<75% falls back to AlwaysWrite + brief_v1-only.

No LLM spend. Local CPU inference only. Writes a result JSON for the
record; no ingestion, no consolidation.
"""

from __future__ import annotations

# ruff: noqa: E402
import torch  # noqa: F401  (import torch first, merken "Mistake #10")

import json
import random
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
N = 30
SAMPLE_INDEX = 0  # first seed=44 question


def main() -> int:
    print(f"[load] v7 decider  ckpt={V7_CKPT.exists()}  meta={V7_META.exists()}", flush=True)
    decider = NanoGPTWriteDecider(str(V7_CKPT), str(V7_META))
    ctx = WriteContext(project="longmemeval_filter_smoke")

    print(f"[load] longmemeval_s, sampling seed={SEED} n={N}", flush=True)
    convs = load_longmemeval("longmemeval_s")
    rnd = random.Random(SEED)
    sampled = rnd.sample(convs, N)
    conv = sampled[SAMPLE_INDEX]

    print(f"[qid] {conv.question_id}", flush=True)
    print(f"[type] {conv.question_type}", flush=True)
    print(f"[question] {conv.question}", flush=True)
    print(f"[answer] {conv.answer}", flush=True)
    print(f"[answer_session_ids] {conv.answer_session_ids}", flush=True)
    print(
        f"[haystack] n_sessions={conv.n_sessions} n_turns={conv.n_turns}",
        flush=True,
    )

    answer_turns = []
    for sid in conv.answer_session_ids:
        if sid not in conv.haystack_sessions:
            print(f"[warn] answer session {sid} missing from haystack", flush=True)
            continue
        for i, t in enumerate(conv.haystack_sessions[sid]):
            answer_turns.append((sid, i, t.role, t.content))

    total = len(answer_turns)
    print(f"[answer-session turns] {total}", flush=True)

    kept: list[dict] = []
    dropped: list[dict] = []
    for sid, idx, role, content in answer_turns:
        d = decider.decide(Event(text=content), ctx)
        row = {
            "sid": sid,
            "idx": idx,
            "role": role,
            "write": d.write,
            "confidence": d.confidence,
            "reason": d.reason,
            "content_preview": content[:160].replace("\n", " "),
            "content_len": len(content),
        }
        (kept if d.write else dropped).append(row)

    recall = len(kept) / total if total else 0.0
    if recall >= 0.95:
        verdict = "PROCEED_STEP_2"
    elif recall >= 0.75:
        verdict = "PROCEED_WITH_CAVEAT"
    else:
        verdict = "STOP_FALLBACK_ALWAYSWRITE"

    print("", flush=True)
    print("=== result ===", flush=True)
    print(f"kept    : {len(kept)}/{total} ({recall*100:.1f}%)", flush=True)
    print(f"dropped : {len(dropped)}/{total}", flush=True)
    print(f"verdict : {verdict}", flush=True)

    print("", flush=True)
    print("=== DROPPED (potential needle loss) ===", flush=True)
    for r in dropped:
        print(
            f"  [{r['sid']}:{r['idx']}] role={r['role']}  {r['reason']}  len={r['content_len']}",
            flush=True,
        )
        print(f"    -> {r['content_preview']}", flush=True)

    print("", flush=True)
    print("=== KEPT (sample) ===", flush=True)
    for r in kept[:5]:
        print(
            f"  [{r['sid']}:{r['idx']}] role={r['role']}  {r['reason']}  len={r['content_len']}",
            flush=True,
        )
        print(f"    -> {r['content_preview']}", flush=True)

    out = {
        "qid": conv.question_id,
        "question_type": conv.question_type,
        "question": conv.question,
        "answer": conv.answer,
        "seed": SEED,
        "n_sampled": N,
        "sample_index": SAMPLE_INDEX,
        "answer_session_ids": conv.answer_session_ids,
        "n_answer_turns": total,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "recall": recall,
        "verdict": verdict,
        "dropped_turns": dropped,
        "kept_turns": kept,
    }
    out_path = Path(__file__).parent / "filter_recall_seed44_smoke.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
