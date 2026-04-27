"""LoCoMo end-to-end runner -- Phase 2 stack.

Mirrors experiments/retrieval/longmemeval/run_vstash_ask.py but on
the LoCoMo dataset, so we can validate cross-benchmark whether the
Phase 2 findings (k=8, default vstash.ask prompt, llama3.1-8b
builder) transfer or were LongMemEval-specific.

Stack:
  - Builder: Cerebras llama3.1-8b via vstash.Memory.ask() (default
    SYSTEM_PROMPT, the Pareto-optimal prompt-lever winner)
  - Oracle: Gemini 2.5 Flash via mode_a_eval.oracle_score
  - Retrieval: top_k = 8 (Phase 2 default)
  - Ingest: per-turn, title format ``qid::sid::turn_idx`` matching
    the LongMemEval pattern in run_vstash_ask._ingest

Configs (--config flag):
  - vstash-raw   : ingest direct to vstash.Memory (control,
    bypasses all merken write decisions). Matches run_vstash_ask.
  - merken-recall: ingest via merken.Memory with AlwaysWrite +
    NeverConsolidate + NeverForget. Adds the audit trail and
    primitive plumbing without filtering anything out.
  - merken-v7    : ingest via merken.Memory with the Heuristic
    -> NanoGPT-v7 chained write decider. The write filter MAY drop
    turns. Tests E2E whether v7's 86% filter recall is harmful
    or helpful for downstream answer correctness.

In every config the QUESTION path is identical: open
``vstash.Memory(db=...)`` against the post-ingest db, override
inference to Cerebras llama3.1-8b, and call ``.ask()``. So the
delta is attributable to the write side.

Optimization: each conversation's haystack is ingested once and
reused for all its QA pairs (10 ingests vs 1986).

Usage::

    python3 experiments/retrieval/locomo/runner_phase2.py \
      --config vstash-raw --seed 44 --top-k 8 --max-convs 1 --tag smoke

    python3 experiments/retrieval/locomo/runner_phase2.py \
      --config merken-v7 --seed 44 --top-k 8 --n-per-cat 40 --tag merken_v7

Set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and ``CEREBRAS_API_KEY``
in the environment.
"""
# ruff: noqa: I001, E402

from __future__ import annotations

# torch first for fastembed/vstash MPS init order on macOS.
import torch  # noqa: F401

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

# Force Cerebras backend BEFORE vstash imports so its config picks it up.
os.environ.setdefault("VSTASH_BACKEND", "cerebras")
os.environ.setdefault("VSTASH_MODEL", "llama3.1-8b")

import vstash  # noqa: E402
import vstash.chat as _vstash_chat  # noqa: E402

from experiments.retrieval.locomo.runner import (  # noqa: E402
    CATEGORY_NAMES,
    Conversation,
    QAPair,
    Session,
    Turn,
    load_locomo,
)
from experiments.retrieval.longmemeval.mode_a_eval import (  # noqa: E402
    _oracle_client,
    oracle_score,
)


# Hash of the unmodified vstash.chat.SYSTEM_PROMPT captured 2026-04-25.
# If a prior session ran run_vstash_ask.py with --prompt-variant hybrid,
# the module global is mutated and persists; we hard-fail in that case
# so the LoCoMo numbers stay comparable to LongMemEval Phase 2 default.
_DEFAULT_SYSTEM_PROMPT_LEN = 574
_DEFAULT_SYSTEM_PROMPT_HEAD = (
    "You are a precise document assistant. Answer questions based strictly "
    "on the provided context."
)

# Prompt fix v1 (2026-04-25): removed default rule 4 ("not in X but
# in Y") AND added a hard "respond exactly 'not enough information
# in memory'" rule. Result on seed=44 N=201: -8.4pp correct,
# +4.0pp trust. Multi_hop collapsed -27.5pp because the Builder
# stopped aggregating across chunks. Kept here for record.
FIX_V1_SYSTEM_PROMPT = """You are a precise document assistant. Answer questions based strictly on the provided context.

Rules:
- Answer the question directly using any relevant chunk in the context. If multiple chunks contain relevant facts, combine them.
- Cite the source title in brackets after each specific fact (use the document title shown in brackets in the context).
- If no chunk in the context contains the answer, respond exactly: "not enough information in memory". Do not speculate.
- Do NOT preface answers with "this is not in [X] but in [Y]" -- just answer using whatever chunk has the information.
- For code questions, provide working code examples from the context."""

# Prompt fix v2 (2026-04-25): minimal change -- the default minus
# rule 4 only. No new restrictive rule. Tests whether removing the
# self-contradiction template alone is enough, without forcing the
# "not enough info" pattern that hurt multi_hop in v1.
FIX_V2_SYSTEM_PROMPT = """You are a precise document assistant. Answer questions based strictly on the provided context.

Rules:
- Answer only from the context. Do not invent information.
- If the context doesn't contain the answer, say so clearly.
- Always cite which source document each fact comes from (use the document title shown in brackets).
- For code questions, provide working code examples from the context."""

FIX_SYSTEM_PROMPT = FIX_V2_SYSTEM_PROMPT  # alias for backwards compat


def _assert_default_prompt() -> None:
    sp = _vstash_chat.SYSTEM_PROMPT
    if not sp.startswith(_DEFAULT_SYSTEM_PROMPT_HEAD):
        raise SystemExit(
            f"vstash.chat.SYSTEM_PROMPT has been mutated "
            f"(len={len(sp)}, head={sp[:80]!r}). "
            f"Restart the Python process or fix the mutation -- this run "
            f"must use the default trust-first prompt to be comparable to "
            f"LongMemEval Phase 2 numbers."
        )
    print(
        f"[prompt] vstash.chat.SYSTEM_PROMPT default OK "
        f"(len={len(sp)}, expected~{_DEFAULT_SYSTEM_PROMPT_LEN})",
        flush=True,
    )


def _apply_prompt_variant(variant: str) -> None:
    if variant == "default":
        _assert_default_prompt()
        return
    if variant == "fix":  # alias of fix-v2 (current best minimal fix)
        variant = "fix-v2"
    if variant == "fix-v1":
        _assert_default_prompt()
        _vstash_chat.SYSTEM_PROMPT = FIX_V1_SYSTEM_PROMPT
        print(
            f"[prompt] APPLIED fix-v1 variant "
            f"(len={len(FIX_V1_SYSTEM_PROMPT)})",
            flush=True,
        )
        return
    if variant == "fix-v2":
        _assert_default_prompt()
        _vstash_chat.SYSTEM_PROMPT = FIX_V2_SYSTEM_PROMPT
        print(
            f"[prompt] APPLIED fix-v2 variant "
            f"(len={len(FIX_V2_SYSTEM_PROMPT)})",
            flush=True,
        )
        return
    raise ValueError(f"unknown prompt variant: {variant}")


# Per-turn formatting kept aligned with the LongMemEval Phase 2
# ingest path. LoCoMo turns expose ``speaker`` (a personal name)
# rather than ``role`` (user/assistant), so we cannot reuse
# longmemeval.runner._format_turn directly. Mirror the same
# "speaker: text" shape, no special-token stripping needed because
# LoCoMo conversations are already plain text.
def _format_turn(turn: Turn) -> str:
    return f"{turn.speaker}: {turn.text}"


def _override_inference_config(mem: vstash.Memory, backend: str, model: str) -> None:
    """Override the inference backend/model on this Memory's frozen
    config without rewriting vstash.toml. Mirrors run_vstash_ask.
    """
    from vstash.config import InferenceConfig

    old = mem._cfg
    new_inference = InferenceConfig(backend=backend, model=model)
    new_cfg = old.model_copy(update={"inference": new_inference})
    object.__setattr__(mem, "_cfg", new_cfg)


def _v7_chained_decider():
    """Build a ChainedWriteDecider(Heuristic, nanoGPT v7).

    Mirrors locomo/runner.py's _v7_chained_decider so the v7 we test
    here is exactly the same artifact validated in earlier filter-recall
    experiments. Calibrator OFF by default -- v7 was deemed graduated
    in shadow mode without it (see CLAUDE.md).
    """
    from merken.classifiers.nanogpt import NanoGPTWriteDecider
    from merken.policies.should_remember import (
        ChainedWriteDecider,
        HeuristicWriteDecider,
    )

    nanogpt_dir = os.environ.get("NANOGPT_REPO")
    if not nanogpt_dir:
        raise SystemExit(
            "[runner_phase2] NANOGPT_REPO env var must point to the local "
            "nanoGPT checkout (containing out-merken-bpe-v7/ckpt.pt and "
            "data/merken_bpe_v7/meta.pkl) when --use-v7 is set."
        )
    ckpt = f"{nanogpt_dir}/out-merken-bpe-v7/ckpt.pt"
    meta = f"{nanogpt_dir}/data/merken_bpe_v7/meta.pkl"
    classifier = NanoGPTWriteDecider(ckpt, meta, calibrator=None)
    return ChainedWriteDecider(HeuristicWriteDecider(), classifier)


def _conv_items(conv: Conversation, granularity: str):
    """Yield (text, title) pairs for ingest at the given granularity.

    LoCoMo's natural unit is per-session (turns ~130 chars dialogue,
    sessions 2.8K chars). Per-turn fragments below the grain that
    contains the answer; per-session preserves the [date_time]
    header adjacent to dialogue. See RESULTS_phase2.md
    "Re-baseline LME at per-session: NEGATIVE result" for why
    LME goes the other way.
    """
    if granularity == "per-turn":
        for session in conv.sessions:
            sid = f"session_{session.index}"
            for i, turn in enumerate(session.turns):
                yield _format_turn(turn), f"{conv.sample_id}::{sid}::{i}"
    elif granularity == "per-session":
        for session in conv.sessions:
            sid = f"session_{session.index}"
            lines = [_format_turn(t) for t in session.turns]
            text = f"[{session.date_time}]\n" + "\n".join(lines)
            yield text, f"{conv.sample_id}::{sid}"
    else:
        raise ValueError(f"unknown granularity: {granularity}")


def _ingest_via_vstash(
    db_path: str, conv: Conversation, collection: str,
    *, granularity: str,
) -> tuple[int, int]:
    """vstash-raw config: ingest items at the given granularity
    directly via vstash.Memory. Returns (n_attempted, n_written).
    vstash never refuses, so they match.
    """
    mem = vstash.Memory(db=db_path)
    try:
        n = 0
        for text, title in _conv_items(conv, granularity):
            mem.remember(text, title=title, collection=collection)
            n += 1
        return n, n
    finally:
        mem.close()


def _build_write_decider(config: str):
    """Build the write decider once at startup so:
       - v7 ckpt-load failure is loud and immediate, not deferred to
         the first conversation,
       - the v7 NanoGPT model is loaded once and reused across all
         conversations rather than rebuilt 10x.
    """
    if config == "vstash-raw":
        return None
    if config == "merken-recall":
        from merken import AlwaysWrite
        return AlwaysWrite()
    if config == "merken-v7":
        return _v7_chained_decider()
    raise ValueError(f"unknown --config: {config}")


def _ingest_via_merken(
    db_path: str, conv: Conversation, collection: str, *,
    write_decider, granularity: str,
) -> tuple[int, int]:
    """merken-* configs: ingest via merken.Memory at the given
    granularity so the write decision tree is exercised on the same
    items vstash-raw would see.

    Returns (n_attempted, n_written). When write_decider rejects, the
    item is NOT persisted -- so the downstream ``vstash.ask`` retrieval
    pool reflects the filter's selectivity.
    """
    from merken import (
        Memory,
        NeverConsolidate,
        NeverForget,
    )

    mem = Memory(
        project=f"locomo_{conv.sample_id}",
        db=db_path,
        collection=collection,
        write_decider=write_decider,
        consolidate_decider=NeverConsolidate(),
        forget_decider=NeverForget(),
    )
    try:
        n_attempted = 0
        n_written = 0
        for text, title in _conv_items(conv, granularity):
            r = mem.remember(text, title=title)
            n_attempted += 1
            if r.written:
                n_written += 1
        return n_attempted, n_written
    finally:
        mem.close()


def _ingest_conversation(
    db_path: str, conv: Conversation, collection: str, *,
    config: str, write_decider, granularity: str,
) -> tuple[int, int]:
    """Dispatch to the per-config ingest path. ``write_decider`` is
    pre-built (see ``_build_write_decider``) so the model is shared
    across conversations.
    """
    if config == "vstash-raw":
        return _ingest_via_vstash(
            db_path, conv, collection, granularity=granularity,
        )
    return _ingest_via_merken(
        db_path, conv, collection,
        write_decider=write_decider, granularity=granularity,
    )


def run_qa(
    mem: vstash.Memory,
    conv: Conversation,
    qa: QAPair,
    *,
    top_k: int,
    oracle,
    collection: str,
    vec_weight: float | None = None,
    fts_weight: float | None = None,
    retrieval_mode: str | None = None,
) -> dict:
    """Ask the question, judge the answer. Errors land as a row
    rather than crashing the run, so a single API hiccup does not
    abort 1986 questions of work.
    """
    t_q = time.perf_counter()

    t0 = time.perf_counter()
    ask_kwargs = {"top_k": top_k, "collection": collection}
    if vec_weight is not None:
        ask_kwargs["vec_weight"] = vec_weight
    if fts_weight is not None:
        ask_kwargs["fts_weight"] = fts_weight
    if retrieval_mode is not None:
        ask_kwargs["retrieval_mode"] = retrieval_mode
    try:
        answer = mem.ask(qa.question, **ask_kwargs)
    except Exception as exc:  # noqa: BLE001 -- per-question fail-soft
        traceback.print_exc()
        return {
            "sample_id": conv.sample_id,
            "question": qa.question,
            "category": qa.category,
            "category_name": qa.category_name,
            "ground_truth": qa.answer,
            "evidence": qa.evidence,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": time.perf_counter() - t_q,
        }
    # Reasoning models (e.g. gpt-oss-120b) sometimes burn the token
    # budget on hidden reasoning before producing visible content,
    # leaving message.content=None. Treat as a refusal row instead
    # of crashing the oracle on a None candidate.
    if answer is None:
        return {
            "sample_id": conv.sample_id,
            "question": qa.question,
            "category": qa.category,
            "category_name": qa.category_name,
            "ground_truth": qa.answer,
            "evidence": qa.evidence,
            "error": "no_content (model returned None message.content)",
            "wall_s": time.perf_counter() - t_q,
        }
    ask_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    o = oracle_score(oracle, qa.question, qa.answer, answer)
    oracle_s = time.perf_counter() - t0

    return {
        "sample_id": conv.sample_id,
        "category": qa.category,
        "category_name": qa.category_name,
        "question": qa.question,
        "ground_truth": qa.answer,
        "evidence": qa.evidence,
        "answer": answer,
        "oracle": o,
        "wall_s_ask": ask_s,
        "wall_s_oracle": oracle_s,
        "wall_s_total": time.perf_counter() - t_q,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "locomo10.json",
    )
    ap.add_argument("--seed", type=int, default=44,
                    help="Tag only -- LoCoMo full has no sampling. "
                         "Re-runs with the same seed will still differ "
                         "because Cerebras inference is non-deterministic.")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--backend", default=os.environ.get("VSTASH_BACKEND", "cerebras"))
    ap.add_argument("--model", default=os.environ.get("VSTASH_MODEL", "llama3.1-8b"))
    ap.add_argument("--max-convs", type=int, default=None,
                    help="Limit to first N conversations (smoke test).")
    ap.add_argument("--max-qa-per-conv", type=int, default=None,
                    help="Limit QA pairs per conversation (smoke test).")
    ap.add_argument("--categories", default=None,
                    help="Comma-separated category IDs (1-5). Default = all.")
    ap.add_argument("--n-per-cat", type=int, default=None,
                    help="Stratified sample: take N QAs per category, "
                         "sampled with --seed (and seed-shuffled within "
                         "each cat). Mirrors Phase 2 LME methodology where "
                         "different seeds yield different samples + builder "
                         "runs. Picks across all conversations.")
    ap.add_argument(
        "--config",
        choices=["vstash-raw", "merken-recall", "merken-v7"],
        default="vstash-raw",
        help="Ingest path. vstash-raw bypasses merken (Phase 2 baseline). "
             "merken-recall exercises primitives without filtering. "
             "merken-v7 applies the v7 write filter end-to-end.",
    )
    ap.add_argument(
        "--granularity",
        choices=["per-turn", "per-session"],
        default="per-session",
        help="Ingest unit. per-session is LoCoMo's natural unit "
             "(2.8K char sessions, dialogue-grain turns). per-turn "
             "is legacy and underperforms by ~13pp on LoCoMo.",
    )
    ap.add_argument(
        "--prompt-variant",
        choices=["default", "fix", "fix-v1", "fix-v2"],
        default="default",
        help="Builder system prompt. 'default' is vstash.chat.SYSTEM_PROMPT. "
             "'fix-v1' removes rule 4 AND adds a hard 'not enough info' "
             "rule (tested 2026-04-25, hurt multi_hop -27pp). "
             "'fix-v2' = 'fix' = default minus rule 4 only, minimal change.",
    )
    ap.add_argument(
        "--vec-weight", type=float, default=None,
        help="vstash hybrid retrieval vec weight. Default ~0.86.",
    )
    ap.add_argument(
        "--fts-weight", type=float, default=None,
        help="vstash hybrid retrieval fts weight. Default ~0.14.",
    )
    ap.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vec_only", "fts_only"],
        default=None,
    )
    ap.add_argument("--tag", default="phase2")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "phase2_runs",
    )
    args = ap.parse_args()

    _apply_prompt_variant(args.prompt_variant)
    if args.backend != "cerebras" or args.model != "llama3.1-8b":
        print(
            f"[warn] backend={args.backend} model={args.model} -- LoCoMo "
            f"results will NOT be comparable to LongMemEval Phase 2 "
            f"(cerebras/llama3.1-8b). Continuing anyway.",
            flush=True,
        )

    convs = load_locomo(args.data)
    if args.max_convs is not None:
        convs = convs[:args.max_convs]

    cat_filter: set[int] | None = None
    if args.categories:
        cat_filter = {int(c.strip()) for c in args.categories.split(",") if c.strip()}

    def _filter_qas(conv: Conversation) -> list[QAPair]:
        qas = conv.qa_pairs
        if cat_filter is not None:
            qas = [q for q in qas if q.category in cat_filter]
        if args.max_qa_per_conv is not None:
            qas = qas[:args.max_qa_per_conv]
        return qas

    # Stratified sampling: select up to --n-per-cat QAs per category
    # across all conversations, deterministic given --seed. Returns
    # the set of (sample_id, question) tuples we will evaluate; the
    # main loop then iterates conversations as before but only runs
    # the QAs in the selected set.
    selected_keys: set[tuple[str, str]] | None = None
    if args.n_per_cat is not None:
        import random as _random
        rng = _random.Random(args.seed)
        by_cat_pool: dict[int, list[tuple[str, QAPair]]] = {}
        for conv in convs:
            for qa in _filter_qas(conv):
                by_cat_pool.setdefault(qa.category, []).append((conv.sample_id, qa))
        selected_keys = set()
        for cat, pool in sorted(by_cat_pool.items()):
            rng.shuffle(pool)
            taken = pool[:args.n_per_cat]
            for sid, qa in taken:
                selected_keys.add((sid, qa.question))
            print(
                f"[sample] cat={cat} ({CATEGORY_NAMES.get(cat, cat)}): "
                f"pool={len(pool)} took={len(taken)}",
                flush=True,
            )

    def _qas_for_conv(conv: Conversation) -> list[QAPair]:
        qas = _filter_qas(conv)
        if selected_keys is not None:
            qas = [q for q in qas if (conv.sample_id, q.question) in selected_keys]
        return qas

    total_qa = sum(len(_qas_for_conv(conv)) for conv in convs)

    # Build the write decider ONCE at startup so the v7 model is
    # shared across conversations (10x speedup on full LoCoMo) and
    # any ckpt-load failure surfaces immediately rather than after
    # the first conversation finishes ingesting.
    write_decider = _build_write_decider(args.config)
    print(
        f"[locomo-phase2] config={args.config} granularity={args.granularity} "
        f"backend={args.backend} model={args.model} top_k={args.top_k} "
        f"seed={args.seed} convs={len(convs)} total_qa={total_qa}",
        flush=True,
    )
    if write_decider is not None:
        print(f"[write_decider] {type(write_decider).__name__}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # `convbatched` flags that each conv reuses one Memory across its
    # ~190 QA pairs (vs run_vstash_ask which builds a fresh Memory per
    # question). Methodologically a footnote, not a divergence: vstash
    # does not cache query results, so cross-question retrieval state
    # is invariant. Filename makes that explicit so a future per-Q run
    # is not silently compared to this one.
    out_path = args.out / (
        f"locomo_phase2_{args.config}_{args.granularity}_"
        f"prompt-{args.prompt_variant}_seed{args.seed}_"
        f"{len(convs)}convs_{total_qa}qa_convbatched_"
        f"{args.tag}_{ts}.jsonl"
    )
    print(f"[out] {out_path}", flush=True)

    oracle = _oracle_client()

    from experiments.retrieval.oracle_health import (
        OracleHealthError,
        OracleHealthGuard,
    )
    health = OracleHealthGuard()
    try:
        health.preflight(lambda q, gt, ans: oracle_score(oracle, q, gt, ans))
        print(f"[oracle] pre-flight OK", flush=True)
    except OracleHealthError as exc:
        print(f"[oracle] PRE-FLIGHT FAILED: {exc}", flush=True)
        return 2

    # Aggregate counters. `error` is a fifth verdict bucket emitted by
    # run_qa() when mem.ask raises; the row is still written so the run
    # does not abort on a single API hiccup. errors count toward the
    # `total` denominator (so an outage does NOT inflate correct_rate)
    # and are reported as a separate line in the summary.
    verdicts: Counter[str] = Counter()
    correct = 0
    total = 0
    errors = 0
    by_cat: dict[str, Counter[str]] = {}
    t_all = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f:
        for conv_idx, conv in enumerate(convs, start=1):
            qas = _qas_for_conv(conv)
            if not qas:
                continue

            print(
                f"\n[conv {conv_idx}/{len(convs)}] {conv.sample_id} "
                f"sessions={len(conv.sessions)} qa_pairs={len(qas)}",
                flush=True,
            )

            with tempfile.TemporaryDirectory(prefix=f"locomo_p2_{conv.sample_id}_") as td:
                db_path = str(Path(td) / "mem.db")
                collection = "default"

                # 1) Ingest. For merken configs this opens a separate
                # merken.Memory and closes it; for vstash-raw it opens
                # vstash.Memory directly. Either way the post-ingest
                # state lives in the same SQLite file.
                t0 = time.perf_counter()
                n_attempted, n_written = _ingest_conversation(
                    db_path, conv, collection,
                    config=args.config,
                    write_decider=write_decider,
                    granularity=args.granularity,
                )
                ingest_s = time.perf_counter() - t0
                drop_pct = (
                    (n_attempted - n_written) / n_attempted * 100
                    if n_attempted else 0.0
                )
                print(
                    f"  ingested {n_written}/{n_attempted} turns "
                    f"({drop_pct:.1f}% dropped) in {ingest_s:.1f}s",
                    flush=True,
                )

                # 2) Open vstash.Memory against the post-ingest db
                # for the .ask() loop. Same builder + prompt across
                # all configs so the answer-correctness delta is
                # attributable to the write side.
                mem = vstash.Memory(db=db_path)
                _override_inference_config(mem, args.backend, args.model)

                try:
                    for j, qa in enumerate(qas, start=1):
                        row = run_qa(
                            mem, conv, qa,
                            top_k=args.top_k,
                            oracle=oracle,
                            collection=collection,
                            vec_weight=args.vec_weight,
                            fts_weight=args.fts_weight,
                            retrieval_mode=args.retrieval_mode,
                        )
                        row["_ingest_n_attempted"] = n_attempted
                        row["_ingest_n_written"] = n_written
                        row["_config"] = args.config
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()

                        if "error" in row:
                            v = "error"
                            errors += 1
                        else:
                            v = (row.get("oracle") or {}).get("verdict") or "error"
                            if v == "error":
                                errors += 1
                            try:
                                health.record(row.get("oracle"))
                            except OracleHealthError as exc:
                                print(f"\n[oracle] HEALTH ABORT: {exc}", flush=True)
                                return 2
                        verdicts[v] += 1
                        total += 1
                        if v in ("supports", "partial"):
                            correct += 1
                        by_cat.setdefault(qa.category_name, Counter())[v] += 1

                        if j % 25 == 0 or j == len(qas):
                            print(
                                f"    [{j}/{len(qas)}] running correct="
                                f"{correct}/{total} = "
                                f"{correct/max(1,total)*100:.1f}%",
                                flush=True,
                            )
                finally:
                    mem.close()

    wall_total = time.perf_counter() - t_all
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  wall total   : {wall_total:.1f}s ({wall_total/60:.1f} min)", flush=True)
    print(f"  verdicts     : {dict(verdicts)}", flush=True)
    if not health.summary_safe:
        print(f"\n  {health.warning()}\n", flush=True)
        print(
            f"  correct      : SUPPRESSED ({health.errors}/{health.total} oracle errors)",
            flush=True,
        )
        print(f"  trust_score  : SUPPRESSED", flush=True)
    else:
        print(
            f"  correct      : {correct}/{total} = "
            f"{correct/max(1,total)*100:.1f}%",
            flush=True,
        )
        # trust_score = (correct - contradicts) / total. Errors land in
        # the denominator but neither the numerator nor the contradicts
        # subtraction -- they shrink the rate without flipping its sign,
        # which is the right shape for "no answer due to outage".
        print(
            f"  trust_score  : "
            f"{(correct - verdicts['contradicts'])/max(1,total)*100:+.1f}%",
            flush=True,
        )
    if errors:
        print(
            f"  errors       : {errors}/{total} "
            f"({errors/max(1,total)*100:.1f}%) -- counted as wrong, "
            f"not as contradicts",
            flush=True,
        )
    if not health.summary_safe:
        print(
            "\n  per category : SUPPRESSED -- rejudge before reading per-shape numbers",
            flush=True,
        )
    else:
        print("\n  per category:", flush=True)
        for cat_name in sorted(by_cat):
            cnt = by_cat[cat_name]
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
