"""Re-judge an existing run's answers with the Cerebras oracle so
its scores are comparable to a fresh Cerebras-oracle run (the Gemini
quota was hit; we need apples-to-apples on the same oracle).

Reads a jsonl with rows containing {question, ground_truth, answer},
calls the Cerebras oracle, writes a new jsonl with `oracle_cer`
field added (keeping the original `oracle` field intact).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

from experiments.retrieval.locomo.runner_rerank import _CerebrasOracle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    oracle = _CerebrasOracle()
    rows = [json.loads(l) for l in args.inp.open()]
    print(f"rows: {len(rows)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with args.out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, start=1):
            # If the row had an error, oracle_cer == neutral with same
            # rationale -- can't re-judge nothing.
            if "error" in r:
                r["oracle_cer"] = {
                    "verdict": "neutral",
                    "rationale": f"original_error: {r.get('error')}",
                }
            else:
                ans = r.get("answer") or r.get("vstash_answer") or ""
                gt = r.get("ground_truth", "")
                q = r.get("question", "")
                r["oracle_cer"] = oracle.score(q, gt, ans)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(rows):
                print(
                    f"  [{i}/{len(rows)}] wall={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
    print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
