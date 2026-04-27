"""LoCoMo runner with cross-encoder reranking.

Retrieves k=30 candidates from vstash hybrid search, reranks
them with a cross-encoder, takes top-8, and feeds those chunks
to vstash.chat.ask (which uses the same default SYSTEM_PROMPT
as mem.ask). Same builder + oracle as runner_phase2.

Hypothesis: the 47% retrieval miss bucket from miss-analysis
2026-04-25 is partially recoverable -- evidence sessions are
in top-30 but not top-8, and a cross-encoder can promote them
to top-8 by scoring query-chunk relevance directly.
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

from experiments.retrieval.locomo.runner import (  # noqa: E402
    CATEGORY_NAMES,
    Conversation,
    QAPair,
    load_locomo,
)
from experiments.retrieval.locomo.runner_phase2 import (  # noqa: E402
    _format_turn,
    _override_inference_config,
)
from experiments.retrieval.longmemeval.mode_a_eval import (  # noqa: E402
    _oracle_client,
    _parse_oracle_json,
    oracle_score,
)


# Cerebras-based oracle fallback. Same prompt + parsing as the Gemini
# oracle in mode_a_eval, but uses Cerebras llama3.1-8b instead. Used
# when Gemini quota is exhausted (e.g. monthly spending limit hit).
# Builder is also llama3.1-8b in our setup, so be aware that oracle
# and Builder share the same model -- this is a self-evaluation bias
# we accept temporarily; a stronger model (gemini-2.5, gpt-4o) is
# preferred when quota is available.
_CEREBRAS_ORACLE_PROMPT = """You are a strict grader scoring answer quality.

Question:
{question}

Ground-truth answer:
{ground_truth}

Candidate answer:
{candidate}

Score the candidate against the ground truth with ONE of:
- "supports"     : the candidate answers the question and agrees with the ground truth
- "partial"      : partially correct (right on the main point, imprecise on a detail)
- "contradicts"  : the candidate answers the question but disagrees with the ground truth
- "neutral"      : the candidate does not actually answer the question (refusal, off-topic, empty)

Respond ONLY with a JSON object:
{{"verdict": "supports"|"partial"|"contradicts"|"neutral", "rationale": "<one sentence>"}}
No prose outside the JSON. No markdown fences."""


class _CerebrasOracle:
    def __init__(self, model: str = "llama3.1-8b") -> None:
        from cerebras.cloud.sdk import Cerebras
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise SystemExit("CEREBRAS_API_KEY required")
        self._client = Cerebras(api_key=api_key)
        self._model = model
        self._calls = 0

    def score(self, question: str, ground_truth: str, candidate: str) -> dict:
        prompt = _CEREBRAS_ORACLE_PROMPT.format(
            question=question[:1500],
            ground_truth=ground_truth[:2000],
            candidate=candidate[:2000],
        )
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            self._calls += 1
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            return {
                "verdict": "neutral",
                "rationale": f"oracle_error: {exc}",
                "wall_s": time.perf_counter() - t0,
            }
        parsed = _parse_oracle_json(raw)
        parsed["wall_s"] = time.perf_counter() - t0
        return parsed


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


def _ingest_per_session(mem: vstash.Memory, conv: Conversation, collection: str) -> int:
    n = 0
    for session in conv.sessions:
        sid = f"session_{session.index}"
        lines = [_format_turn(t) for t in session.turns]
        text = f"[{session.date_time}]\n" + "\n".join(lines)
        mem.remember(text, title=f"{conv.sample_id}::{sid}", collection=collection)
        n += 1
    return n


def _build_reranker(model_name: str):
    """Lazy-load the cross-encoder. Loaded once at startup, reused
    across questions.
    """
    from sentence_transformers import CrossEncoder
    print(f"[rerank] loading {model_name}...", flush=True)
    t0 = time.perf_counter()
    ce = CrossEncoder(model_name)
    print(f"[rerank] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return ce


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data", type=Path,
        default=Path(__file__).parent / "data" / "locomo10.json",
    )
    ap.add_argument("--top-k-retrieve", type=int, default=30,
                    help="Candidates fed to the reranker.")
    ap.add_argument("--top-k-final", type=int, default=8,
                    help="Reranked candidates passed to the Builder.")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--n-per-cat", type=int, default=None)
    ap.add_argument("--max-convs", type=int, default=None)
    ap.add_argument("--max-qa-per-conv", type=int, default=None)
    ap.add_argument(
        "--rerank-model", default="mixedbread-ai/mxbai-rerank-base-v1",
        help="HuggingFace cross-encoder model id for reranking.",
    )
    ap.add_argument(
        "--oracle", choices=["gemini", "cerebras"], default="gemini",
        help="Oracle backend. 'cerebras' is the fallback when Gemini "
             "monthly quota is exhausted -- carries self-evaluation bias "
             "since the Builder is also llama3.1-8b.",
    )
    ap.add_argument(
        "--vec-weight", type=float, default=None,
        help="Pass-through to vstash.search hybrid weights.",
    )
    ap.add_argument("--fts-weight", type=float, default=None)
    ap.add_argument("--tag", default="rerank")
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "phase2_runs",
    )
    args = ap.parse_args()

    convs = load_locomo(args.data)
    if args.max_convs is not None:
        convs = convs[:args.max_convs]

    def _qas_for_conv(conv: Conversation) -> list[QAPair]:
        qas = list(conv.qa_pairs)
        if args.max_qa_per_conv is not None:
            qas = qas[:args.max_qa_per_conv]
        return qas

    selected_keys: set[tuple[str, str]] | None = None
    if args.n_per_cat is not None:
        import random as _random
        rng = _random.Random(args.seed)
        by_cat: dict[int, list[tuple[str, QAPair]]] = {}
        for conv in convs:
            for qa in _qas_for_conv(conv):
                by_cat.setdefault(qa.category, []).append((conv.sample_id, qa))
        selected_keys = set()
        for cat, pool in sorted(by_cat.items()):
            rng.shuffle(pool)
            for sid, qa in pool[:args.n_per_cat]:
                selected_keys.add((sid, qa.question))
            print(
                f"[sample] cat={cat} ({CATEGORY_NAMES.get(cat, cat)}): "
                f"pool={len(pool)} took={min(len(pool), args.n_per_cat)}",
                flush=True,
            )

    def _selected(conv: Conversation) -> list[QAPair]:
        qas = _qas_for_conv(conv)
        if selected_keys is not None:
            qas = [q for q in qas if (conv.sample_id, q.question) in selected_keys]
        return qas

    total_qa = sum(len(_selected(c)) for c in convs)
    print(
        f"[locomo-rerank] retrieve_k={args.top_k_retrieve} "
        f"final_k={args.top_k_final} seed={args.seed} "
        f"convs={len(convs)} total_qa={total_qa} model={args.rerank_model}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / (
        f"locomo_rerank_seed{args.seed}_"
        f"k{args.top_k_retrieve}to{args.top_k_final}_"
        f"{len(convs)}convs_{total_qa}qa_{args.tag}_{ts}.jsonl"
    )
    print(f"[out] {out_path}", flush=True)

    reranker = _build_reranker(args.rerank_model)
    if args.oracle == "cerebras":
        oracle = _CerebrasOracle()
        print(f"[oracle] using Cerebras llama3.1-8b (Gemini quota fallback)",
              flush=True)
    else:
        oracle = _oracle_client()
        print(f"[oracle] using Gemini (mode_a_eval default)", flush=True)

    verdicts: Counter[str] = Counter()
    by_cat_v: dict[str, Counter[str]] = {}
    correct = 0
    total = 0
    errors = 0
    t_all = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f:
        for ci, conv in enumerate(convs, start=1):
            qas = _selected(conv)
            if not qas:
                continue
            print(
                f"\n[conv {ci}/{len(convs)}] {conv.sample_id} qa={len(qas)} "
                f"sessions={len(conv.sessions)}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix="locomo_rer_") as td:
                db = str(Path(td) / "mem.db")
                mem = vstash.Memory(db=db)
                _override_inference_config(mem, "cerebras", "llama3.1-8b")
                t0 = time.perf_counter()
                n_ing = _ingest_per_session(mem, conv, "default")
                ingest_s = time.perf_counter() - t0
                print(f"  ingested {n_ing} sessions in {ingest_s:.1f}s", flush=True)
                try:
                    for j, qa in enumerate(qas, start=1):
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
                                hits = mem.search(qa.question, **search_kwargs)
                        except _TimeoutError as exc:
                            f.write(json.dumps({
                                "sample_id": conv.sample_id,
                                "category_name": qa.category_name,
                                "question": qa.question,
                                "ground_truth": qa.answer,
                                "error": f"search_timeout: {exc}",
                            }, ensure_ascii=False) + "\n")
                            f.flush()
                            errors += 1
                            verdicts["error"] += 1
                            total += 1
                            continue

                        # 2) cross-encoder rerank
                        if hits:
                            t_rer = time.perf_counter()
                            pairs = [(qa.question, h.text) for h in hits]
                            scores = reranker.predict(pairs)
                            # sort hits by reranker score desc
                            order = sorted(
                                range(len(hits)), key=lambda i: -float(scores[i])
                            )
                            top_final = [hits[i] for i in order[:args.top_k_final]]
                            rerank_s = time.perf_counter() - t_rer
                        else:
                            top_final = hits
                            rerank_s = 0.0

                        # 3) vstash.chat.ask with reranked chunks
                        try:
                            with _alarm(120):
                                answer = _vstash_chat.ask(
                                    qa.question, top_final, mem._cfg,
                                )
                            err = None
                        except _TimeoutError as exc:
                            answer, err = "", f"ask_timeout: {exc}"
                        except Exception as exc:  # noqa: BLE001
                            traceback.print_exc()
                            answer, err = "", f"{type(exc).__name__}: {exc}"

                        # 4) oracle
                        if err is None:
                            try:
                                with _alarm(60):
                                    if args.oracle == "cerebras":
                                        o = oracle.score(
                                            qa.question, qa.answer, answer,
                                        )
                                    else:
                                        o = oracle_score(
                                            oracle, qa.question, qa.answer,
                                            answer,
                                        )
                            except _TimeoutError as exc:
                                o = {"verdict": "neutral",
                                     "rationale": f"oracle_timeout: {exc}"}
                        else:
                            o = {"verdict": "neutral", "rationale": err}

                        v = o.get("verdict") or "error"
                        if v == "error" or err is not None:
                            errors += 1
                        verdicts[v] += 1
                        by_cat_v.setdefault(qa.category_name, Counter())[v] += 1
                        total += 1
                        if v in ("supports", "partial"):
                            correct += 1

                        row = {
                            "sample_id": conv.sample_id,
                            "category": qa.category,
                            "category_name": qa.category_name,
                            "question": qa.question,
                            "ground_truth": qa.answer,
                            "evidence": qa.evidence,
                            "answer": answer,
                            "oracle": o,
                            "wall_s_rerank": rerank_s,
                            "wall_s_total": time.perf_counter() - t_q,
                            "_n_retrieved": len(hits),
                            "_n_final": len(top_final),
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        if j % 25 == 0 or j == len(qas):
                            print(
                                f"    [{j}/{len(qas)}] correct="
                                f"{correct}/{total} = "
                                f"{correct/max(1,total)*100:.1f}%",
                                flush=True,
                            )
                finally:
                    mem.close()

    wall_total = time.perf_counter() - t_all
    print(f"\n=== SUMMARY (rerank {args.top_k_retrieve}->{args.top_k_final}) ===",
          flush=True)
    print(f"  wall total   : {wall_total:.1f}s ({wall_total/60:.1f} min)",
          flush=True)
    print(f"  verdicts     : {dict(verdicts)}", flush=True)
    print(
        f"  correct      : {correct}/{total} = "
        f"{correct/max(1,total)*100:.1f}%",
        flush=True,
    )
    print(
        f"  trust_score  : "
        f"{(correct - verdicts['contradicts'])/max(1,total)*100:+.1f}%",
        flush=True,
    )
    if errors:
        print(f"  errors       : {errors}/{total}", flush=True)
    print("\n  per category:", flush=True)
    for cat_name in sorted(by_cat_v):
        cnt = by_cat_v[cat_name]
        n = sum(cnt.values())
        c = cnt["supports"] + cnt["partial"]
        print(
            f"    {cat_name:<14} {c}/{n} = {c/max(1,n)*100:5.1f}%  "
            f"verdicts={dict(cnt)}",
            flush=True,
        )
    print(f"[out] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
