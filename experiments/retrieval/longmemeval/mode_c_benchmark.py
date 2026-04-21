"""Mode C benchmark on LongMemEval -- head-to-head with RAG / Mode A.

The ``mode_a_eval.py`` grid runner handles Cerebras-backed
conditions (control / rag / rag_specific / mode_a). Mode C runs
on a local mlx model with streaming generation + KV-splice;
architecturally distinct enough that it gets its own runner.

This script reuses the LongMemEval question-and-haystack dataset
and the same Gemini 2.5 Flash oracle so results are comparable
to the N=50 runs already in ``mode_a_eval_grids/``.

Per question:

  1. Load the question's haystack into a fresh vstash db
     (matches ``mode_a_eval.py`` ingestion shape).
  2. Run ``mode_c_demo.run_mode_c`` end-to-end against that db.
  3. Oracle the answer with Gemini.
  4. Write one JSONL audit row capturing the question, ground
     truth, mode_c output, splice events, oracle verdict, and
     wall-time.

Default N=30 (subset of the seed=42 sample used for RAG
baseline) to keep wall time bounded. Each mode_c generation
takes ~15-20s on gemma-4-E2B-it local; 30 questions ~ 7-10
minutes plus ingestion. Oracle spend ~\\$0.01 per question.

Run:
  python experiments/retrieval/longmemeval/mode_c_benchmark.py \\
      --n 30 --seed 42 \\
      --out experiments/retrieval/longmemeval/mode_c_runs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import vstash

from experiments.midloop_concept.medlocal.mode_c_demo import (
    load_model,
    run_mode_c,
)
from experiments.retrieval.longmemeval.dataset import (
    Conversation,
    load_longmemeval,
)
from experiments.retrieval.longmemeval.mode_a_eval import (
    _ingest,
    _oracle_client,
    oracle_score,
)


def _load_questions(subset: str, n: int, seed: int) -> list[Conversation]:
    conversations = load_longmemeval(subset=subset)
    rnd = random.Random(seed)
    return rnd.sample(conversations, min(n, len(conversations)))


def _correct(v: str) -> bool:
    return v in ("supports", "partial")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="longmemeval_s")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/retrieval/longmemeval/mode_c_runs"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home()
        / ".lmstudio/models/lmstudio-community/gemma-4-E2B-it-MLX-4bit",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"mode_c_n{args.n}_seed{args.seed}.jsonl"

    print(f"[config] subset={args.subset} n={args.n} seed={args.seed}")
    print(f"[config] out={out_path}")
    print(f"[config] model={args.model}")

    sampled = _load_questions(args.subset, args.n, args.seed)
    print(f"[dataset] sampled {len(sampled)} conversations")

    # Load the mlx model ONCE up front and reuse across all N
    # questions -- each load is ~2-3s and doing N loads would
    # blow ~90s of pure wasted wall on N=30.
    model, tokenizer = load_model(args.model)

    oracle = _oracle_client()

    tmp_root = Path.home() / ".merken" / f"longmemeval_mode_c_{args.seed}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    fout = out_path.open("w")
    rows: list[dict] = []

    try:
        for i, conv in enumerate(sampled):
            gt_text = str(conv.answer) if conv.answer is not None else ""
            print(
                f"\n{'='*72}\n[{i+1}/{len(sampled)}] qid={conv.question_id} "
                f"type={conv.question_type}\n{'='*72}"
            )
            print(f"Q: {conv.question[:200]}")
            print(f"GT: {gt_text[:200]}")

            db_path = tmp_root / f"{conv.question_id}.db"
            if db_path.exists():
                db_path.unlink()
            mem = vstash.Memory(
                db=str(db_path),
                project=conv.question_id,
                collection="default",
            )

            try:
                t_ingest = time.perf_counter()
                _ingest(mem, conv)
                ingest_s = time.perf_counter() - t_ingest
                print(f"[ingest] {ingest_s:.1f}s")
            finally:
                mem.close()

            try:
                t_mc = time.perf_counter()
                mc_result = run_mode_c(
                    model_path=args.model,
                    db_path=db_path,
                    project=conv.question_id,
                    question=conv.question,
                    model=model,
                    tokenizer=tokenizer,
                )
                mc_wall = time.perf_counter() - t_mc
                print(
                    f"[mode_c] {mc_wall:.1f}s  "
                    f"tokens={mc_result.total_tokens_sampled}  "
                    f"splices={len(mc_result.splices)}  "
                    f"termination={mc_result.terminated_reason}"
                )

                # Mode C answer_text can be 3000-4000 chars:
                # 300-400 tokens of ``<|channel>thought``
                # preamble + splice payloads + the actual answer
                # body at the end. The oracle prompt truncates
                # ``candidate`` to the first 2000 chars, which
                # would silently discard the real answer. Strip
                # the preamble (last ``<channel|>`` marker
                # separates thought from response body) and
                # tail-truncate so the oracle sees the answer.
                raw = mc_result.answer_text
                answer_for_oracle = (
                    raw.rsplit("<channel|>", 1)[-1]
                    if "<channel|>" in raw else raw
                )
                answer_for_oracle = answer_for_oracle[-2000:]
                o = oracle_score(
                    oracle, conv.question, gt_text, answer_for_oracle
                )
                print(f"[oracle] verdict={o.get('verdict')}")

                row = {
                    "audit_id": uuid.uuid4().hex[:12],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt_text,
                    "mode_c": {
                        "answer_text": mc_result.answer_text,
                        "total_tokens_sampled": mc_result.total_tokens_sampled,
                        "wall_s": mc_wall,
                        "terminated_reason": mc_result.terminated_reason,
                        "splices": [
                            {
                                "trigger_token_index": s.trigger_token_index,
                                "reason": s.reason,
                                "source_id": s.source_id,
                                "source_score": s.source_score,
                                "splice_tokens": s.splice_tokens,
                            }
                            for s in mc_result.splices
                        ],
                        "oracle": o,
                    },
                    "ingest_s": ingest_s,
                }
                fout.write(json.dumps(row, default=str) + "\n")
                fout.flush()
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] qid={conv.question_id}: {exc!r}")
                traceback.print_exc()
                fout.write(json.dumps({
                    "question_id": conv.question_id,
                    "question": conv.question,
                    "error": repr(exc),
                }, default=str) + "\n")
                fout.flush()
            finally:
                if db_path.exists():
                    db_path.unlink()
    finally:
        fout.close()

    # Summary
    n_ok = sum(1 for r in rows if "mode_c" in r)
    n_correct = sum(
        1 for r in rows
        if "mode_c" in r
        and _correct(r["mode_c"]["oracle"]["verdict"])
    )
    total_splices = sum(
        len(r["mode_c"].get("splices", [])) for r in rows if "mode_c" in r
    )
    total_tokens = sum(
        r["mode_c"].get("total_tokens_sampled", 0) for r in rows if "mode_c" in r
    )
    total_wall = sum(r["mode_c"].get("wall_s", 0) for r in rows if "mode_c" in r)

    print("\n" + "=" * 72)
    print("MODE C BENCHMARK SUMMARY")
    print("=" * 72)
    if n_ok == 0:
        print("  no completed rows")
        return 1
    print(f"  n_completed:        {n_ok}/{len(sampled)}")
    print(
        f"  correct_rate:       {n_correct/n_ok*100:.1f}%  "
        f"({n_correct}/{n_ok})"
    )
    print(f"  avg tokens/q:       {total_tokens/n_ok:.0f}")
    print(f"  avg wall/q:         {total_wall/n_ok:.1f}s")
    print(f"  splices total:      {total_splices}")
    print(f"  splices/q:          {total_splices/n_ok:.2f}")
    print()
    print("Comparison anchors (from earlier LongMemEval runs on same seed 42):")
    print("  RAG-k3 temp=0.0:    70.0% correct / 1394 tok / 1.1s")
    print("  RAG-k5 temp=0.1-3:  72-74% correct / 2290 tok / 1.2s")
    print("  Mode A v4:          64-71% correct / 9048 tok / 5.3s")
    print(f"  Mode C (this run):  {n_correct/n_ok*100:.1f}% / "
          f"{total_tokens/n_ok:.0f} tok / {total_wall/n_ok:.1f}s")
    print(f"\nlog: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
