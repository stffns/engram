"""LoCoMo direct vs vstash, no intermediates.

Tests the hypothesis (Jay 2026-04-25 EOD): the bad LoCoMo numbers
come from our per-turn ingest -- splitting the dialogue into one
``remember`` per turn destroys conversational adjacency, so vstash
sees fragments instead of dialogue. Letting vstash chunk a larger
blob may preserve the context the Builder needs.

Three variants of ingest, identical Builder + oracle:

- ``per-turn``     : one remember per turn (current Phase 2 default)
- ``per-session``  : one remember per session (session date header
                     + concatenated turns); vstash chunks if needed
- ``per-conv``     : one remember per conversation (all sessions
                     joined); vstash chunks the entire dialogue
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


class _TimeoutError(Exception):
    pass


@contextmanager
def _alarm(seconds: int):
    """SIGALRM-based wall-clock timeout. Only works on the main
    thread on Unix; we use it to break out of an SSL read that has
    no built-in timeout (the failure mode that hung the first run).
    """
    def _handler(signum, frame):
        raise _TimeoutError(f"timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

os.environ.setdefault("VSTASH_BACKEND", "cerebras")
os.environ.setdefault("VSTASH_MODEL", "llama3.1-8b")

import vstash  # noqa: E402

from experiments.retrieval.locomo.runner import (  # noqa: E402
    CATEGORY_NAMES,
    Conversation,
    QAPair,
    load_locomo,
)
from experiments.retrieval.longmemeval.mode_a_eval import (  # noqa: E402
    _oracle_client,
    oracle_score,
)


def _override_inference_config(mem: vstash.Memory, backend: str, model: str) -> None:
    from vstash.config import InferenceConfig

    old = mem._cfg
    new_inference = InferenceConfig(backend=backend, model=model)
    new_cfg = old.model_copy(update={"inference": new_inference})
    object.__setattr__(mem, "_cfg", new_cfg)


def _ingest(mem: vstash.Memory, conv: Conversation, granularity: str) -> int:
    n = 0
    if granularity == "per-turn":
        for session in conv.sessions:
            sid = f"session_{session.index}"
            for i, turn in enumerate(session.turns):
                mem.remember(
                    f"{turn.speaker}: {turn.text}",
                    title=f"{conv.sample_id}::{sid}::{i}",
                )
                n += 1
    elif granularity == "per-session":
        for session in conv.sessions:
            sid = f"session_{session.index}"
            lines = [f"{t.speaker}: {t.text}" for t in session.turns]
            text = f"[{session.date_time}]\n" + "\n".join(lines)
            mem.remember(text, title=f"{conv.sample_id}::{sid}")
            n += 1
    elif granularity == "per-conv":
        parts = []
        for session in conv.sessions:
            lines = [f"{t.speaker}: {t.text}" for t in session.turns]
            parts.append(f"[{session.date_time}]\n" + "\n".join(lines))
        full = "\n\n".join(parts)
        mem.remember(full, title=conv.sample_id)
        n = 1
    else:
        raise ValueError(granularity)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--granularity",
        choices=["per-turn", "per-session", "per-conv"],
        required=True,
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "locomo10.json",
    )
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--n-per-cat", type=int, default=None)
    ap.add_argument("--max-convs", type=int, default=None)
    ap.add_argument("--max-qa-per-conv", type=int, default=None)
    ap.add_argument("--tag", default="minimal")
    ap.add_argument(
        "--out",
        type=Path,
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

    def _selected_qas(conv: Conversation) -> list[QAPair]:
        qas = _qas_for_conv(conv)
        if selected_keys is not None:
            qas = [q for q in qas if (conv.sample_id, q.question) in selected_keys]
        return qas

    total_qa = sum(len(_selected_qas(c)) for c in convs)
    print(
        f"[locomo-min] granularity={args.granularity} top_k={args.top_k} "
        f"seed={args.seed} convs={len(convs)} total_qa={total_qa}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / (
        f"locomo_minimal_{args.granularity}_seed{args.seed}_"
        f"{len(convs)}convs_{total_qa}qa_{args.tag}_{ts}.jsonl"
    )
    print(f"[out] {out_path}", flush=True)

    oracle = _oracle_client()
    verdicts: Counter[str] = Counter()
    by_cat_v: dict[str, Counter[str]] = {}
    correct = 0
    total = 0
    t_all = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f:
        for ci, conv in enumerate(convs, start=1):
            qas = _selected_qas(conv)
            if not qas:
                continue
            print(
                f"\n[conv {ci}/{len(convs)}] {conv.sample_id} "
                f"qa={len(qas)} sessions={len(conv.sessions)}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix="locomo_min_") as td:
                db = str(Path(td) / "mem.db")
                mem = vstash.Memory(db=db)
                _override_inference_config(mem, "cerebras", "llama3.1-8b")
                t0 = time.perf_counter()
                n_items = _ingest(mem, conv, args.granularity)
                ingest_s = time.perf_counter() - t0
                print(f"  ingested {n_items} items in {ingest_s:.1f}s", flush=True)
                try:
                    for j, qa in enumerate(qas, start=1):
                        t0 = time.perf_counter()
                        try:
                            with _alarm(120):
                                ans = mem.ask(qa.question, top_k=args.top_k)
                            err = None
                        except _TimeoutError as exc:
                            ans = ""
                            err = f"timeout: {exc}"
                            print(f"    [{j}] ASK TIMEOUT", flush=True)
                        except Exception as exc:  # noqa: BLE001
                            traceback.print_exc()
                            ans = ""
                            err = f"{type(exc).__name__}: {exc}"
                        ask_s = time.perf_counter() - t0

                        if err is None:
                            try:
                                with _alarm(60):
                                    o = oracle_score(
                                        oracle, qa.question, qa.answer, ans,
                                    )
                            except _TimeoutError as exc:
                                o = {
                                    "verdict": "neutral",
                                    "rationale": f"oracle_timeout: {exc}",
                                }
                                print(f"    [{j}] ORACLE TIMEOUT", flush=True)
                        else:
                            o = {"verdict": "neutral", "rationale": err}

                        v = o.get("verdict") or "error"
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
                            "answer": ans,
                            "oracle": o,
                            "wall_s_ask": ask_s,
                            "_granularity": args.granularity,
                            "_n_ingested": n_items,
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        if j % 25 == 0 or j == len(qas):
                            print(
                                f"    [{j}/{len(qas)}] correct="
                                f"{correct}/{total} = {correct/max(1,total)*100:.1f}%",
                                flush=True,
                            )
                finally:
                    mem.close()

    wall_total = time.perf_counter() - t_all
    print(f"\n=== SUMMARY granularity={args.granularity} ===", flush=True)
    print(f"  wall total   : {wall_total:.1f}s ({wall_total/60:.1f} min)", flush=True)
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
