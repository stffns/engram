"""Phase 1 validation for v8c (frozen encoder + MLP head).

Sibling of filter_recall_v8_smoke.py but loads FrozenEncoderWriteDecider
instead of NanoGPTWriteDecider. Same seed=44 N=5 protocol, same gate
thresholds, so v7, v8 (nanoGPT), and v8c (frozen encoder) results are
directly comparable.

Gate:
- >= 95% median -> PROCEED_PHASE_2
- 75-94%       -> PROCEED_WITH_CAVEAT
- < 75%        -> STOP

Zero LLM spend.
"""

from __future__ import annotations

# ruff: noqa: E402
import torch  # noqa: F401

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402
from merken.classifiers.frozen_encoder import FrozenEncoderWriteDecider  # noqa: E402
from merken.policies.types import Event, WriteContext  # noqa: E402


SEED = 44
N_SAMPLE = 30
N_MEASURE = 5


def _needle_candidates(answer) -> list[str]:
    if not isinstance(answer, str):
        answer = str(answer)
    cands = {answer.strip()}
    a = answer.strip()
    if a.endswith("%") and a[:-1].replace(".", "").isdigit():
        n = a[:-1]
        cands.update({n, f"{n}%", f"{n} percent"})
    return [c for c in cands if c]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--head",
        type=Path,
        default=Path("experiments/nanogpt/v8c_frozen_encoder/head.pt"),
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("experiments/nanogpt/v8c_frozen_encoder/meta.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tag", default="v8c")
    parser.add_argument("--n", type=int, default=N_MEASURE)
    args = parser.parse_args()

    print(f"[load] decider tag={args.tag}  head={args.head.exists()}  meta={args.meta.exists()}", flush=True)
    if not args.head.exists() or not args.meta.exists():
        raise SystemExit(f"missing head or meta at {args.head} / {args.meta}")
    decider = FrozenEncoderWriteDecider(
        str(args.head), str(args.meta),
        confidence_threshold=args.threshold,
    )
    ctx = WriteContext(project=f"longmemeval_filter_smoke_{args.tag}")

    convs = load_longmemeval("longmemeval_s")
    rnd = random.Random(SEED)
    sampled = rnd.sample(convs, N_SAMPLE)

    per_q: list[dict] = []
    for qi in range(args.n):
        conv = sampled[qi]
        needles = _needle_candidates(conv.answer)
        answer_turns = []
        for sid in conv.answer_session_ids:
            if sid not in conv.haystack_sessions:
                continue
            for i, t in enumerate(conv.haystack_sessions[sid]):
                answer_turns.append((sid, i, t.role, t.content))

        kept = dropped = 0
        needle_turns = []
        for sid, idx, role, content in answer_turns:
            d = decider.decide(Event(text=content), ctx)
            if d.write:
                kept += 1
            else:
                dropped += 1
            if any(n in content for n in needles):
                needle_turns.append({"sid": sid, "idx": idx, "role": role,
                                      "write": d.write, "reason": d.reason})

        total = kept + dropped
        recall = kept / total if total else 0.0
        n_needle = len(needle_turns)
        n_needle_kept = sum(1 for r in needle_turns if r["write"])
        needle_recall = (n_needle_kept / n_needle) if n_needle else None

        per_q.append({
            "idx": qi, "qid": conv.question_id, "question_type": conv.question_type,
            "n_answer_turns": total, "n_kept": kept, "n_dropped": dropped,
            "answer_session_recall": recall,
            "n_needle_turns": n_needle, "n_needle_kept": n_needle_kept,
            "needle_recall": needle_recall,
        })
        nrec = f"{needle_recall*100:.0f}%" if needle_recall is not None else "N/A"
        print(
            f"[{qi}] qid={conv.question_id}  type={conv.question_type}  "
            f"kept={kept}/{total} ({recall*100:.1f}%)  "
            f"needle={n_needle_kept}/{n_needle} ({nrec})",
            flush=True,
        )

    recalls = [r["answer_session_recall"] for r in per_q]
    print("", flush=True)
    print(f"=== aggregate over {args.n} questions ({args.tag}) ===", flush=True)
    print(
        f"answer-session recall: mean={statistics.mean(recalls)*100:.1f}%  "
        f"median={statistics.median(recalls)*100:.1f}%  "
        f"min={min(recalls)*100:.1f}%  max={max(recalls)*100:.1f}%",
        flush=True,
    )

    med = statistics.median(recalls)
    if med >= 0.95:
        verdict = "PROCEED_PHASE_2"
    elif med >= 0.75:
        verdict = "PROCEED_WITH_CAVEAT"
    else:
        verdict = "STOP_RETRAIN_OR_ABANDON"
    print(f"\ngate verdict: {verdict}  (median {med*100:.1f}%)", flush=True)

    out = {
        "tag": args.tag, "head": str(args.head), "meta": str(args.meta),
        "threshold": args.threshold, "seed": SEED, "n_measure": args.n,
        "per_question": per_q,
        "aggregate": {
            "mean": statistics.mean(recalls), "median": statistics.median(recalls),
            "min": min(recalls), "max": max(recalls),
        },
        "verdict": verdict,
    }
    out_path = Path(__file__).parent / f"filter_recall_seed44_n{args.n}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[out] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
