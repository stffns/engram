"""Re-score existing Mode C audit rows with the corrected oracle
extraction (find LAST complete <channel|>...<turn|> block instead of
the previous rsplit-last-<channel|>).

Rationale: 2026-04-22 smoke tests on qid=3b6f954b and qid=4fd1909e
showed the old extraction was silently losing correct answer blocks
when the model emitted multiple turns before budget ran out. Every
prior N=30 grid cell in ``mode_c_runs_v3/`` was scored with the
broken extraction; re-scoring with the fix reveals the true
correctness numbers without the ~20-30 minute cost of regenerating.

Run:
  python -m experiments.retrieval.longmemeval.mode_c_rescore \\
      --in experiments/retrieval/longmemeval/mode_c_runs_v3 \\
      --out experiments/retrieval/longmemeval/mode_c_runs_v3_rescored

Writes parallel .jsonl files with an extra ``oracle_rescored``
field per row; existing ``oracle`` verdict is preserved for A/B
comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from experiments.retrieval.longmemeval.mode_a_eval import (
    _oracle_client,
    oracle_score,
)


ANSWER_BLOCK = re.compile(r"<channel\|>(.*?)<turn\|>", re.DOTALL)
TURN_RUN = re.compile(r"(<turn\|>)+")
QWEN_MARKERS = re.compile(r"<\|im_(start|end)\|>")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
EXCERPT_BLOCK_V2 = re.compile(
    r"<<<MEMORY_EXCERPT[^>]*>>>.*?<<<END_MEMORY_EXCERPT>>>", re.DOTALL
)
EXCERPT_ORPHAN_V2 = re.compile(r"<<<(MEMORY_EXCERPT[^>]*|END_MEMORY_EXCERPT)>>>")
EXCERPT_V1_HEADER = re.compile(r"\[Source: [^\]]+\]\s*\n[^\[]*")


def _extract_answer(raw: str) -> str:
    """Fixed extraction: return the last complete answer block with
    Builder scaffolding stripped so the oracle sees only the tokens
    the model sampled as its user-facing answer. Covers:

    - gemma multi-channel extraction (<channel|>...<turn|>)
    - ChatML markers from Qwen (<|im_start|>, <|im_end|>)
    - Thinking blocks (<think>...</think>)
    - Regurgitated V2 envelope blocks
      (<<<MEMORY_EXCERPT ...>>>...<<<END_MEMORY_EXCERPT>>>)
    - Regurgitated V1 envelope headers ([Source: X] + chunk text)
    """
    blocks = ANSWER_BLOCK.findall(raw)
    if blocks:
        # Last complete block. Empirically best for safety-tuned
        # gemma. Abliterated gemma needs stop-at-first-turn at
        # generation time (plumbing TODO).
        candidate = blocks[-1].strip()
    elif "<channel|>" in raw:
        candidate = raw.rsplit("<channel|>", 1)[-1]
    else:
        candidate = raw
    candidate = TURN_RUN.sub("", candidate)
    candidate = QWEN_MARKERS.sub("", candidate)
    candidate = THINK_BLOCK.sub("", candidate)
    candidate = EXCERPT_BLOCK_V2.sub("", candidate)
    candidate = EXCERPT_ORPHAN_V2.sub("", candidate)
    candidate = EXCERPT_V1_HEADER.sub("", candidate)
    return candidate.strip()[-2000:]


def _correct(v: str) -> bool:
    return v in ("supports", "partial")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=Path, required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, required=True)
    parser.add_argument(
        "--pattern", default="mode_c_*.jsonl",
        help="glob for input files relative to --in"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oracle = _oracle_client()

    files = sorted(args.in_dir.glob(args.pattern))
    if not files:
        print(f"no files matching {args.pattern} in {args.in_dir}", file=sys.stderr)
        return 2

    print(f"{'file':<52} {'N':>3} {'old':>10} {'new':>10} {'delta':>6}")
    for fp in files:
        rows = [json.loads(l) for l in fp.open()]
        out_rows: list[dict] = []
        old_correct = 0
        new_correct = 0
        counted = 0
        for r in rows:
            if "mode_c" not in r:
                out_rows.append(r)
                continue
            counted += 1
            old_v = (r["mode_c"]["oracle"] or {}).get("verdict", "?")
            if _correct(old_v):
                old_correct += 1
            raw = r["mode_c"]["answer_text"]
            fixed_candidate = _extract_answer(raw)
            new_oracle = oracle_score(
                oracle, r["question"], str(r["ground_truth"]), fixed_candidate
            )
            new_v = (new_oracle or {}).get("verdict", "?")
            if _correct(new_v):
                new_correct += 1
            r["mode_c"]["oracle_rescored"] = new_oracle
            r["mode_c"]["oracle_candidate_rescored"] = fixed_candidate
            out_rows.append(r)
        out_path = args.out_dir / fp.name
        with out_path.open("w") as f:
            for r in out_rows:
                f.write(json.dumps(r, default=str) + "\n")
        delta = new_correct - old_correct
        old_pct = f"{old_correct/counted*100:5.1f}%" if counted else "  n/a "
        new_pct = f"{new_correct/counted*100:5.1f}%" if counted else "  n/a "
        sign = "+" if delta >= 0 else ""
        print(
            f"{fp.name:<52} {counted:>3} "
            f"{old_correct:>4}/{counted:<3} {old_pct:<6}"
            f" {new_correct:>4}/{counted:<3} {new_pct:<6}"
            f" {sign}{delta:>3}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
