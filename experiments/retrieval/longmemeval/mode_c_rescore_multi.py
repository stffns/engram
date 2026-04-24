"""Re-score a Mode C audit log with 3 extraction strategies in
parallel so we can report honest comparison numbers across
Builders that produce different output shapes (safety-tuned
gemma emits refined last-block; abliterated gemma hijacks its
turn after the first answer).

Strategies:
- last     : blocks[-1]  (favors gemma refinement)
- first    : blocks[0]   (favors abliterated pre-hijack)
- shortest : min(blocks, key=len)  (hijacks tend to be verbose)

Run:
  python -m experiments.retrieval.longmemeval.mode_c_rescore_multi \\
      --in <jsonl>
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


def _clean(candidate: str) -> str:
    candidate = TURN_RUN.sub("", candidate)
    candidate = QWEN_MARKERS.sub("", candidate)
    candidate = THINK_BLOCK.sub("", candidate)
    candidate = EXCERPT_BLOCK_V2.sub("", candidate)
    candidate = EXCERPT_ORPHAN_V2.sub("", candidate)
    candidate = EXCERPT_V1_HEADER.sub("", candidate)
    return candidate.strip()[-2000:]


def _extract(raw: str, strategy: str) -> str:
    blocks = ANSWER_BLOCK.findall(raw)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        if strategy == "last":
            return _clean(blocks[-1])
        if strategy == "first":
            return _clean(blocks[0])
        if strategy == "shortest":
            return _clean(min(blocks, key=len))
        raise ValueError(f"unknown strategy {strategy}")
    fallback = raw.rsplit("<channel|>", 1)[-1] if "<channel|>" in raw else raw
    return _clean(fallback)


def _correct(v: str) -> bool:
    return v in ("supports", "partial")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    args = parser.parse_args()

    oracle = _oracle_client()
    rows = [json.loads(l) for l in args.in_path.open()]
    results: dict[str, dict[str, int]] = {
        s: {"supports": 0, "partial": 0, "contradicts": 0, "neutral": 0}
        for s in ("last", "first", "shortest")
    }
    correct: dict[str, int] = {s: 0 for s in results}
    total = 0
    for r in rows:
        if "mode_c" not in r:
            continue
        total += 1
        raw = r["mode_c"]["answer_text"]
        for strategy in ("last", "first", "shortest"):
            candidate = _extract(raw, strategy)
            o = oracle_score(
                oracle, r["question"], str(r["ground_truth"]), candidate
            )
            v = (o or {}).get("verdict", "neutral")
            results[strategy][v] = results[strategy].get(v, 0) + 1
            if _correct(v):
                correct[strategy] += 1

    print(f"{'strategy':<10} {'correct':>10} {'rate':>7} sup/par/con/neu")
    for strategy in ("last", "first", "shortest"):
        r = results[strategy]
        print(
            f"{strategy:<10} {correct[strategy]:>3}/{total:<3}    "
            f"{correct[strategy]/total*100:>5.1f}%  "
            f"{r.get('supports',0)}/{r.get('partial',0)}/"
            f"{r.get('contradicts',0)}/{r.get('neutral',0)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
