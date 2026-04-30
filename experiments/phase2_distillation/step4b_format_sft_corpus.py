"""Phase 2 distillation -- Step 4b: format SFT corpus for training.

Reads the per-row capture produced by step4a (system prompt + user
prompt + teacher_reasoning + teacher_content) and emits a JSONL the
trainer can consume directly. The output format is OpenAI / Qwen
chat-template compatible::

    {
      "qid": "...",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>\\n<reasoning>\\n</think>\\n\\n<content>"}
      ],
      "meta": {"source": "...", "shape": "...", "reasoning_chars": ..., "content_chars": ...}
    }

The assistant target embeds the reasoning trace inside the
``<think>...</think>`` block that Qwen3.5 emits at inference time.
A student trained on this target learns to (a) produce a reasoning
trace, (b) close the trace, (c) commit to a visible answer. That
last one is what step3 showed the zero-shot student fails at --
4/22 calls truncated mid-reasoning without ever reaching content.

Filters (defaults sane for distillation):
- Drop rows with empty ``teacher_content`` (truncated calls leave
  reasoning but no answer; these would teach the student NOT to
  commit, exactly the failure mode we're trying to fix).
- Drop rows with empty ``teacher_reasoning`` (rare in practice,
  but Q->A-only would dilute the reasoning supervision signal).

Usage::

    # Convert the smoke
    python -m experiments.phase2_distillation.step4b_format_sft_corpus \\
        --in experiments/phase2_distillation/sft_smoke/rows.jsonl \\
        --out experiments/phase2_distillation/sft_smoke/train.jsonl

    # Convert the production corpus
    python -m experiments.phase2_distillation.step4b_format_sft_corpus \\
        --in experiments/phase2_distillation/sft_corpus_v1/rows.jsonl \\
        --out experiments/phase2_distillation/sft_corpus_v1/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"


def _format_assistant(reasoning: str, content: str) -> str:
    """Compose the assistant target the student learns to produce.

    Wraps the teacher's reasoning trace in <think>...</think> tags
    (Qwen3.5 reasoning-model convention) followed by the visible
    answer. Both pieces are stripped to avoid leading/trailing
    whitespace artifacts that would confuse tokenization.
    """
    return f"{THINK_OPEN}{(reasoning or '').strip()}{THINK_CLOSE}{(content or '').strip()}"


def _convert(row: dict, drop_empty_content: bool, drop_empty_reasoning: bool) -> tuple[dict | None, str]:
    """Returns (converted_row | None, drop_reason). drop_reason is
    empty when the row is kept.
    """
    qid = row.get("qid")
    messages_input = row.get("messages_input")
    if not isinstance(messages_input, list) or len(messages_input) < 2:
        return None, "missing_messages_input"
    system = messages_input[0].get("content", "")
    user = messages_input[1].get("content", "")
    if not system or not user:
        return None, "empty_system_or_user"

    reasoning = (row.get("teacher_reasoning") or "").strip()
    content = (row.get("teacher_content") or "").strip()
    if drop_empty_content and not content:
        return None, "empty_content"
    if drop_empty_reasoning and not reasoning:
        return None, "empty_reasoning"

    out = {
        "qid": qid,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _format_assistant(reasoning, content)},
        ],
        "meta": {
            "source": row.get("source"),
            "shape": row.get("category_name") or row.get("question_type"),
            "reasoning_chars": len(reasoning),
            "content_chars": len(content),
            "n_chunks_retrieved": row.get("n_chunks_retrieved"),
        },
    }
    return out, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                    help="Input rows.jsonl from step4a.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output train.jsonl in chat-messages format.")
    ap.add_argument("--keep-empty-content", action="store_true",
                    help="Keep rows where teacher truncated and produced no visible answer (NOT recommended for distillation).")
    ap.add_argument("--keep-empty-reasoning", action="store_true",
                    help="Keep rows where the teacher produced no reasoning trace (NOT recommended).")
    args = ap.parse_args()

    if not args.in_path.exists():
        sys.exit(f"missing input: {args.in_path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    drop_empty_content = not args.keep_empty_content
    drop_empty_reasoning = not args.keep_empty_reasoning

    n_in = 0
    n_kept = 0
    drop_reasons: dict[str, int] = {}
    target_lens: list[int] = []
    by_shape: dict[str, dict] = {}

    with args.in_path.open("r", encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                drop_reasons["json_decode_error"] = drop_reasons.get("json_decode_error", 0) + 1
                continue
            converted, reason = _convert(row, drop_empty_content, drop_empty_reasoning)
            if converted is None:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_kept += 1
            assistant = converted["messages"][2]["content"]
            target_lens.append(len(assistant))
            shape = converted["meta"]["shape"] or "unknown"
            d = by_shape.setdefault(shape, {"n": 0, "target_chars": []})
            d["n"] += 1
            d["target_chars"].append(len(assistant))

    summary = {
        "n_in": n_in,
        "n_kept": n_kept,
        "n_dropped": n_in - n_kept,
        "drop_reasons": drop_reasons,
        "target_chars": {
            "min": min(target_lens) if target_lens else 0,
            "median": int(statistics.median(target_lens)) if target_lens else 0,
            "max": max(target_lens) if target_lens else 0,
        },
        "by_shape": {
            shape: {
                "n": d["n"],
                "target_chars_median": int(statistics.median(d["target_chars"])),
            }
            for shape, d in sorted(by_shape.items())
        },
    }
    summary_path = args.out.parent / (args.out.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {n_kept}/{n_in} rows -> {args.out}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
