"""LongMemEval runner with cross-encoder reranking.

Mirrors experiments/retrieval/locomo/runner_rerank.py but on
LongMemEval data, using per-turn ingest (LME's natural unit).
Same builder + Cerebras oracle. Tests cross-benchmark transfer
of the reranker finding (LoCoMo +4.5pp).
"""
# ruff: noqa: I001, E402
from __future__ import annotations

import torch  # noqa: F401

import argparse
import json
import os
import signal
import sys
import tempfile
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

os.environ.setdefault("VSTASH_BACKEND", "cerebras")
os.environ.setdefault("VSTASH_MODEL", "llama3.1-8b")

import vstash  # noqa: E402
import vstash.chat as _vstash_chat  # noqa: E402

from experiments.retrieval.longmemeval.dataset import load_longmemeval  # noqa: E402
from experiments.retrieval.longmemeval.run_vstash_ask import (  # noqa: E402
    _ingest,
    _override_inference_config,
)
from experiments.retrieval.longmemeval.mode_a_eval import (  # noqa: E402
    _oracle_client,
    oracle_score,
)
from experiments.retrieval.locomo.runner_rerank import (  # noqa: E402
    _CerebrasOracle,
    _build_reranker,
)


class _TimeoutError(Exception):
    pass


@contextmanager
def _alarm(seconds: int):
    def _handler(signum, frame):
        raise _TimeoutError(f"timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="longmemeval_s")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--top-k-retrieve", type=int, default=30)
    ap.add_argument("--top-k-final", type=int, default=8)
    ap.add_argument(
        "--granularity", choices=["per-turn", "per-session"], default="per-turn",
        help="LME natural unit is per-turn (turns are document-sized).",
    )
    ap.add_argument(
        "--rerank-model", default="mixedbread-ai/mxbai-rerank-base-v1",
    )
    ap.add_argument("--vec-weight", type=float, default=None)
    ap.add_argument("--fts-weight", type=float, default=None)
    ap.add_argument(
        "--oracle", choices=["gemini", "cerebras"], default="cerebras",
        help="Cerebras default since Gemini quota is exhausted as of 2026-04-25.",
    )
    ap.add_argument("--tag", default="rerank")
    ap.add_argument(
        "--out", type=Path,
        default=Path("experiments/retrieval/longmemeval/pipeline_runs"),
    )
    args = ap.parse_args()

    convs = load_longmemeval(args.subset)
    import random
    rnd = random.Random(args.seed)
    sampled = rnd.sample(convs, min(args.n, len(convs)))

    print(
        f"[lme-rerank] retrieve_k={args.top_k_retrieve} "
        f"final_k={args.top_k_final} granularity={args.granularity} "
        f"seed={args.seed} n={len(sampled)} oracle={args.oracle}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / (
        f"vstash_ask_seed{args.seed}_n{len(sampled)}_"
        f"rerank_{args.top_k_retrieve}to{args.top_k_final}_"
        f"{args.tag}_{ts}.jsonl"
    )
    print(f"[out] {out_path}", flush=True)

    reranker = _build_reranker(args.rerank_model)
    if args.oracle == "cerebras":
        oracle = _CerebrasOracle()
        print("[oracle] using Cerebras llama3.1-8b", flush=True)
    else:
        oracle = _oracle_client()
        print("[oracle] using Gemini", flush=True)

    verdicts: Counter[str] = Counter()
    by_qt: dict[str, Counter[str]] = {}
    correct = 0
    total = 0
    t_all = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f:
        for i, conv in enumerate(sampled, start=1):
            print(f"\n[{i}/{len(sampled)}] qid={conv.question_id} qt={conv.question_type}",
                  flush=True)
            with tempfile.TemporaryDirectory(prefix="lme_rerank_") as td:
                db = str(Path(td) / "mem.db")
                mem = vstash.Memory(db=db)
                _override_inference_config(mem, "cerebras", "llama3.1-8b")
                t0 = time.perf_counter()
                n_ing = _ingest(mem, conv, "default", granularity=args.granularity)
                ingest_s = time.perf_counter() - t0

                t_q = time.perf_counter()
                # 1) retrieve k=30
                search_kwargs = {
                    "top_k": args.top_k_retrieve,
                    "collection": "default",
                }
                if args.vec_weight is not None:
                    search_kwargs["vec_weight"] = args.vec_weight
                if args.fts_weight is not None:
                    search_kwargs["fts_weight"] = args.fts_weight
                try:
                    with _alarm(60):
                        hits = mem.search(conv.question, **search_kwargs)
                except _TimeoutError as exc:
                    f.write(json.dumps({
                        "qid": conv.question_id,
                        "question": conv.question,
                        "error": f"search_timeout: {exc}",
                    }) + "\n")
                    f.flush()
                    verdicts["error"] += 1
                    total += 1
                    mem.close()
                    continue

                # 2) cross-encoder rerank
                if hits:
                    pairs = [(conv.question, h.text) for h in hits]
                    scores = reranker.predict(pairs)
                    order = sorted(
                        range(len(hits)), key=lambda j: -float(scores[j])
                    )
                    top_final = [hits[j] for j in order[:args.top_k_final]]
                else:
                    top_final = hits

                # 3) vstash.chat.ask with reranked chunks
                try:
                    with _alarm(120):
                        answer = _vstash_chat.ask(
                            conv.question, top_final, mem._cfg,
                        )
                    err = None
                except _TimeoutError as exc:
                    answer, err = "", f"ask_timeout: {exc}"
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    answer, err = "", f"{type(exc).__name__}: {exc}"

                # 4) oracle
                gt = conv.answer if isinstance(conv.answer, str) else str(conv.answer)
                if err is None:
                    try:
                        with _alarm(60):
                            if args.oracle == "cerebras":
                                o = oracle.score(conv.question, gt, answer)
                            else:
                                o = oracle_score(oracle, conv.question, gt, answer)
                    except _TimeoutError as exc:
                        o = {"verdict": "neutral",
                             "rationale": f"oracle_timeout: {exc}"}
                else:
                    o = {"verdict": "neutral", "rationale": err}

                v = o.get("verdict") or "error"
                verdicts[v] += 1
                qt = conv.question_type or "unknown"
                by_qt.setdefault(qt, Counter())[v] += 1
                total += 1
                if v in ("supports", "partial"):
                    correct += 1

                row = {
                    "qid": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt,
                    "n_ingested": n_ing,
                    "vstash_answer": answer,
                    "oracle": o,
                    "wall_s_total": time.perf_counter() - t_q,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                mem.close()
                print(
                    f"  -> verdict={v} correct={correct}/{total}="
                    f"{correct/total*100:.1f}%",
                    flush=True,
                )

    wall_total = time.perf_counter() - t_all
    print(f"\n=== SUMMARY (LME rerank {args.top_k_retrieve}->{args.top_k_final}) ===",
          flush=True)
    print(f"  wall total   : {wall_total:.1f}s ({wall_total/60:.1f} min)",
          flush=True)
    print(f"  verdicts     : {dict(verdicts)}", flush=True)
    print(f"  correct      : {correct}/{total} = "
          f"{correct/max(1,total)*100:.1f}%", flush=True)
    print(f"  trust_score  : "
          f"{(correct - verdicts['contradicts'])/max(1,total)*100:+.1f}%",
          flush=True)
    print("\n  per question_type:", flush=True)
    for qt in sorted(by_qt):
        cnt = by_qt[qt]
        n = sum(cnt.values())
        c = cnt["supports"] + cnt["partial"]
        print(f"    {qt:<28} {c}/{n} = {c/max(1,n)*100:5.1f}%  v={dict(cnt)}",
              flush=True)
    print(f"[out] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
