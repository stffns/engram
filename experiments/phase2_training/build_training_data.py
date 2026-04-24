"""Build SFT training data for Phase 2 Ministral LoRA.

Reads a pipeline_runner JSONL (with llama-8b Cerebras as teacher,
k=10 no-briefs, full 414 qids) and emits a prompts.jsonl with
`{system, user, target}` training examples for SFT.

Alignment twist (Jay's principle, 2026-04-24):
- If oracle verdict in {supports, partial}: use teacher's answer
  verbatim (the teacher was right; student should replicate).
- If oracle verdict is `contradicts`: substitute teacher's answer
  with an honest hedge. Reason: the teacher stated a wrong fact
  confidently, and we do NOT want the student to distill that
  hallucination. Hedge pattern: "I do not find enough information
  in memory to confirm this. The context mentions... but does not
  directly answer your question." The student learns that when the
  teacher got it wrong, the correct behavior is to doubt, not guess.
- If oracle verdict is `neutral`: keep teacher's answer (it was
  likely already a hedge; we want the student to learn to hedge too).
- If oracle verdict is missing: skip (eval failed).

Output:
- `data/phase2_sft/train.jsonl`: {system, user, target} per row.
- Split is NOT applied at this stage (caller decides per-seed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are answering as a memory system that must never state "
    "false facts as if true. Use ONLY the provided context.\n\n"
    "Context shape:\n"
    "- EXCERPTS: raw conversational turns from prior sessions.\n\n"
    "Rules (in priority order):\n"
    "1. If the context EXPLICITLY states the answer, provide it "
    "and quote specific values (numbers, names, dates, titles) "
    "verbatim from the context.\n"
    "2. If the context requires combining facts present across "
    "excerpts (e.g. ordering by date, computing a total from "
    "enumerated items), do the combination and state the result "
    "with a brief justification.\n"
    "3. If the context does NOT contain the specific fact the "
    "question asks for, or if answering would require guessing, "
    "inferring from absent evidence, or stretching an analogy, "
    "respond EXACTLY: 'not enough information in memory'. Do not "
    "substitute a related or adjacent fact.\n"
    "4. Prefer admitting insufficient information over stating a "
    "fact you cannot directly support from the context.\n"
    "5. Keep answers under 150 words. Respond in plain prose."
)

HEDGE_TEMPLATE = "not enough information in memory"


def _format_user(question: str, episodic_hits: list[dict]) -> str:
    excerpts_block = []
    for i, h in enumerate(episodic_hits or []):
        title = (h.get("title") or "").strip()
        text = (h.get("text") or "").strip()
        if len(text) > 1500:
            text = text[:1500] + " ...[truncated]"
        excerpts_block.append(f"[excerpt {i}] title={title}\n{text}")
    excerpts = "\n\n".join(excerpts_block) if excerpts_block else "(no excerpts retrieved)"
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved excerpts:\n{excerpts}\n\n"
        "Answer the question following the rules."
    )


def _target_for(row: dict) -> str | None:
    oracle = row.get("oracle") or {}
    verdict = oracle.get("verdict")
    if verdict is None:
        return None  # skip, eval failed
    teacher_ans = (row.get("builder_answer") or "").strip()
    if not teacher_ans:
        return None
    # supports / partial: teacher was right, distill verbatim.
    if verdict in ("supports", "partial"):
        return teacher_ans
    # neutral: teacher was already hedging (not refused entirely but
    # didn't answer), keep as-is so the student learns the hedge
    # shape teachers use.
    if verdict == "neutral":
        return teacher_ans
    # contradicts: teacher hallucinated. Replace with explicit hedge.
    if verdict == "contradicts":
        return HEDGE_TEMPLATE
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-run", required=True, type=Path,
                    help="Output jsonl from pipeline_runner (full 414 qids).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output jsonl with (system, user, target) rows.")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in args.pipeline_run.read_text().splitlines() if l.strip()]
    kept = 0
    by_verdict: dict[str, int] = {"supports": 0, "partial": 0, "neutral": 0, "contradicts": 0}
    skipped = 0
    with args.out.open("w") as f:
        for r in rows:
            v = (r.get("oracle") or {}).get("verdict")
            tgt = _target_for(r)
            if tgt is None:
                skipped += 1
                continue
            user = _format_user(r.get("question") or "",
                                r.get("episodic_hits") or [])
            out_row = {
                "qid": r.get("qid"),
                "question_type": r.get("question_type"),
                "teacher_verdict": v,
                "system": SYSTEM_PROMPT,
                "user": user,
                "target": tgt,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1
            by_verdict[v] = by_verdict.get(v, 0) + 1
    print(f"[build] kept {kept} rows, skipped {skipped}")
    print(f"[build] verdict distribution: {by_verdict}")
    print(f"[build] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
