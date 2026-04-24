"""Step 0 of notes/cerebras-writer-loop-plan.md.

Qualitative inspection of the free session-level labels on the
LongMemEval holdout pool. Samples positive turns (from answer_*
sessions) and negative turns (from distractor sessions) and
prints them so a human can judge label quality before investing
training time.

Decision:
- Positives look mostly fact-bearing -> proceed to Phase 1.
- Positives are prohibitively chit-chat-heavy -> move to Phase 2b
  (Cerebras refinement via gpt-oss-120b on ambiguous turns).

Zero API spend.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402


HELD_OUT_SEEDS = [42, 43, 44]
HELD_OUT_N = 30

SAMPLE_POS = 20
SAMPLE_NEG = 20
SEED_INSPECT = 777


def main() -> int:
    convs = load_longmemeval("longmemeval_s")

    # Decontaminate: drop qids used in seed 42/43/44 N=30
    used = set()
    for s in HELD_OUT_SEEDS:
        rnd = random.Random(s)
        for c in rnd.sample(convs, HELD_OUT_N):
            used.add(c.question_id)

    holdout = [c for c in convs if c.question_id not in used]
    print(f"[pool] {len(holdout)} holdout questions (out of {len(convs)})")

    # Collect all positive and negative turns with (qid, sid, idx, role, content, question)
    pos: list[tuple[str, str, int, str, str, str]] = []
    neg: list[tuple[str, str, int, str, str, str]] = []
    for c in holdout:
        ans_set = set(c.answer_session_ids)
        for sid, turns in c.haystack_sessions.items():
            bucket = pos if sid in ans_set else neg
            for i, t in enumerate(turns):
                bucket.append((c.question_id, sid, i, t.role, t.content, c.question))
    print(f"[raw] positives: {len(pos)}  negatives: {len(neg)}")
    print(f"[raw] positive fraction: {len(pos)/(len(pos)+len(neg))*100:.1f}%")

    rnd = random.Random(SEED_INSPECT)
    pos_sample = rnd.sample(pos, min(SAMPLE_POS, len(pos)))
    neg_sample = rnd.sample(neg, min(SAMPLE_NEG, len(neg)))

    print(f"\n{'='*80}")
    print(f"POSITIVE turns (in answer_* sessions)  n={len(pos_sample)}")
    print(f"{'='*80}")
    for qid, sid, i, role, content, question in pos_sample:
        print(f"\n[{qid}::{sid}::{i}]  role={role}  len={len(content)}")
        print(f"  Q: {question[:120]}")
        print(f"  T: {content[:400].replace(chr(10), ' ')}")

    print(f"\n\n{'='*80}")
    print(f"NEGATIVE turns (in distractor sessions)  n={len(neg_sample)}")
    print(f"{'='*80}")
    for qid, sid, i, role, content, question in neg_sample:
        print(f"\n[{qid}::{sid}::{i}]  role={role}  len={len(content)}")
        print(f"  Q: {question[:120]}")
        print(f"  T: {content[:400].replace(chr(10), ' ')}")

    # Length stats to flag chit-chat asymmetry
    import statistics
    pos_lens = [len(c) for _, _, _, _, c, _ in pos]
    neg_lens = [len(c) for _, _, _, _, c, _ in neg]
    print(f"\n\n=== length stats (chars) ===")
    print(f"positives  median={statistics.median(pos_lens):.0f}  mean={statistics.mean(pos_lens):.0f}  p10={sorted(pos_lens)[len(pos_lens)//10]}  p90={sorted(pos_lens)[9*len(pos_lens)//10]}")
    print(f"negatives  median={statistics.median(neg_lens):.0f}  mean={statistics.mean(neg_lens):.0f}  p10={sorted(neg_lens)[len(neg_lens)//10]}  p90={sorted(neg_lens)[9*len(neg_lens)//10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
