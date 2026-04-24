"""merken full-pipeline runner for LongMemEval Step 3.

Arm 1 (pipeline+RAG): AlwaysWrite ingest of haystack turns ->
per-session brief_v1 via Cerebras -> recall top-k briefs (semantic
layer, filtered to method:brief_v1) + top-k episodic turns -> Builder.

Arm 3 (pipeline+Mode C) is deferred: plug the same brief+episodic
retrieval into mode_c_demo.run_mode_c (not implemented here; this
runner does arm 1 only).

Output: one JSONL row per question with {qid, question, answer,
ground_truth, briefs_used, excerpts_used, oracle_verdict, wall_s,
tokens}. Summary table printed at end.

Baseline RAG-k3 (63.3% on seed=44 N=30) lives in mode_c_runs_v10/;
compare the new arm1 correct_rate to that number.

Cost (N=30): ~1380 Cerebras brief calls (~$2.5) + 30 Builder calls
+ 30 Gemini oracle calls. Budget ~$3 + $0.30 oracle. Wall ~15 min.
"""

from __future__ import annotations

# ruff: noqa: E402
import torch  # noqa: F401  (merken "Mistake #10" -- import torch first)

import argparse
import json
import random
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))

import vstash  # noqa: E402

from experiments.retrieval.longmemeval.dataset import (  # noqa: E402
    Conversation,
    load_longmemeval,
)
from experiments.retrieval.longmemeval.mode_a_eval import (  # noqa: E402
    _cite_footer_from_rag,
    _oracle_client,
    oracle_score,
)
from experiments.retrieval.longmemeval.runner import (  # noqa: E402
    _format_turn as _format_turn_from_runner,
)
from experiments.midloop_concept.medlocal.cerebras_midloop import (  # noqa: E402
    BUILDER,
    MAX_TOKENS_DRAFT,
    cerebras_chat,
)
from merken.consolidation import generate_briefs  # noqa: E402


# ---------------------------------------------------------------- constants

BRIEF_MODEL = "qwen-3-235b-a22b-instruct-2507"
BRIEF_MAX_COMPLETION_TOKENS = 1200
BRIEF_TEMPERATURE = 0.0
BRIEF_WORKERS = 4  # concurrent Cerebras calls for per-session briefs

RAG_TOP_K_EPISODIC = 3
RAG_TOP_K_BRIEFS = 3
RAG_MAX_BRIEF_CHARS = 2000   # cap per brief in the prompt
RAG_MAX_EXCERPT_CHARS = 800  # cap per episodic excerpt in the prompt

# Training-data capture: when set, every synthesized brief is appended
# to the given JSONL file so that expensive Cerebras runs double as
# teacher-signal collection for a future local brief-synth model.
# The handle is opened once in main() and kept open; a lock protects
# multi-thread writes (today the writer is single-threaded but leaving
# the lock here hardens against future refactors).
BRIEF_DUMP_PATH: "Path | None" = None
BRIEF_DUMP_HANDLE = None  # set in main() if --dump-briefs-to is used
BRIEF_DUMP_RUN_ID: str | None = None
_BRIEF_DUMP_LOCK = threading.Lock()

BUILDER_SYSTEM = (
    "Answer the user question using the provided context. "
    "Quote specific numbers or names from the context when they "
    "appear. Keep the answer under 150 words. Do not hedge."
)

# Pipeline-aware variant. The baseline prompt above treats all
# context as a single raw-chunk pool; applied to the pipeline it
# leaves the Builder guessing how briefs relate to excerpts.
# Jay's 2026-04-24 observation: the pipeline input shape is
# fundamentally different (briefs = topic summaries, excerpts =
# raw turns) and the Builder should be told what each is for.
# The rules below do NOT encode LongMemEval-specific knowledge;
# they describe the two context types and the default policy for
# combining them.
PIPELINE_BUILDER_SYSTEM = (
    "Context comes in two kinds:\n\n"
    "- BRIEFS: LLM-generated topic summaries from prior "
    "sessions. Format: `## <topic>` header, `**As of:**` date, "
    "bullets with key facts.\n"
    "- EXCERPTS: raw conversational turns from prior sessions.\n\n"
    "Answer the user's question using whatever combination of "
    "briefs and excerpts contains the answer. Quote specific "
    "values (numbers, names, titles, dates) verbatim when the "
    "context contains them. If the answer requires combining "
    "facts across briefs and/or excerpts (e.g. computing a "
    "ratio, ordering by date, enumerating items), do so and "
    "state the result directly. If neither briefs nor excerpts "
    "contain the answer, respond 'not enough information'.\n\n"
    "Respond in plain prose. Do NOT reproduce the brief "
    "`## topic` / `**As of:**` format in your answer. Keep the "
    "answer under 150 words."
)

# Temperature 0.0 is load-bearing. At 0.3 (cerebras_chat default)
# the same (question, context) flips verdict 16-18% across runs
# (Jay 2026-04-21 N=50 study). Pin to 0.0 so arm-vs-baseline
# deltas survive replication.
BUILDER_TEMPERATURE = 0.0

# Reproduce the _BRIEF_PROMPT_TEMPLATE fingerprint check bypass we
# need since generate_briefs is called per-session (not the
# production whole-corpus call). Tag briefs with session id so the
# retrieval filter is easy to express and audit.
BRIEF_TAG = "method:brief_v1"


# ---------------------------------------------------------------- Cerebras brief synth

def _cerebras_brief_synth(prompts: list[str]) -> str:
    """SynthesizeFn for generate_briefs: (list[str]) -> str.

    Uses the production brief_v1 prompt (prompts[0] is the full
    template rendered by generate_briefs). Retries transient 5xx/429
    with backoff; other errors bubble.
    """
    from cerebras.cloud.sdk import APIStatusError, Cerebras

    client = Cerebras()
    prompt = prompts[0]
    backoffs = [2, 4, 8]
    for attempt in range(len(backoffs) + 1):
        try:
            resp = client.chat.completions.create(
                model=BRIEF_MODEL,
                messages=[
                    {"role": "system", "content": "You synthesize temporal briefs from event streams."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=BRIEF_MAX_COMPLETION_TOKENS,
                temperature=BRIEF_TEMPERATURE,
            )
            break
        except APIStatusError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = isinstance(status, int) and (status >= 500 or status == 429)
            if not transient or attempt == len(backoffs):
                raise
            time.sleep(backoffs[attempt])
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------- ingestion

def _ingest_turns(mem: vstash.Memory, conv: Conversation, collection: str) -> int:
    """AlwaysWrite ingestion: every haystack turn -> one episodic doc.

    Title matches mode_a_eval._ingest so hits can be traced to
    source session by the same downstream tooling.
    """
    n = 0
    for sid, turns in conv.haystack_sessions.items():
        for i, turn in enumerate(turns):
            mem.remember(
                _format_turn_from_runner(turn),
                title=f"{conv.question_id}::{sid}::{i}",
                collection=collection,
                layer="episodic",
            )
            n += 1
    return n


def _per_session_briefs(
    conv: Conversation,
    mem: vstash.Memory,
    collection: str,
    today: str,
) -> tuple[int, int, list[dict]]:
    """Run brief_v1 once per session, write every brief to the
    semantic layer with tag ``method:brief_v1,session:<sid>``.

    Returns ``(n_calls, n_briefs, per_session_meta)`` where the
    metadata list keeps per-session brief counts for the audit row.
    """
    sessions = list(conv.haystack_sessions.items())
    per_session_meta: list[dict] = []

    def _one(sid: str, turns) -> tuple[str, list[str], float, list[tuple[str, str]]]:
        # Also return events so the caller can emit training-data rows
        # containing the input text the teacher actually saw.
        events = [(f"{sid}:{i}", f"[{t.role}] {t.content}") for i, t in enumerate(turns)]
        t0 = time.perf_counter()
        briefs = generate_briefs(events, _cerebras_brief_synth, today=today)
        return sid, briefs, time.perf_counter() - t0, events

    n_calls = 0
    n_briefs = 0
    with ThreadPoolExecutor(max_workers=BRIEF_WORKERS) as ex:
        futures = [ex.submit(_one, sid, turns) for sid, turns in sessions]
        for fut in futures:
            # Preserve whatever partial work completed if a single
            # session raises (Cerebras 5xx past retry, etc.); a
            # $10 all-qids run must not discard an entire qid's
            # earlier sessions just because the 40th session fails.
            try:
                sid, briefs, dt, events = fut.result()
            except Exception as exc:  # noqa: BLE001 -- deliberate
                if BRIEF_DUMP_HANDLE is not None:
                    err_row = {
                        "run_id": BRIEF_DUMP_RUN_ID,
                        "qid": conv.question_id,
                        "sid": None,
                        "today": today,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    line = json.dumps(err_row, ensure_ascii=False) + "\n"
                    with _BRIEF_DUMP_LOCK:
                        BRIEF_DUMP_HANDLE.write(line)
                        BRIEF_DUMP_HANDLE.flush()
                raise
            n_calls += 1
            for bi, brief in enumerate(briefs):
                mem.remember(
                    brief,
                    title=f"brief::{conv.question_id}::{sid}::{bi}",
                    collection=collection,
                    layer="semantic",
                    tags=f"{BRIEF_TAG},session:{sid}",
                )
                n_briefs += 1
            per_session_meta.append({
                "sid": sid,
                "n_briefs": len(briefs),
                "dt_s": dt,
                "is_answer": sid in conv.answer_session_ids,
            })
            if BRIEF_DUMP_HANDLE is not None:
                row = {
                    "run_id": BRIEF_DUMP_RUN_ID,
                    "qid": conv.question_id,
                    "sid": sid,
                    "is_answer_session": sid in conv.answer_session_ids,
                    "today": today,
                    "input_events": [{"id": eid, "text": etxt} for eid, etxt in events],
                    "briefs": briefs,
                    "teacher_model": BRIEF_MODEL,
                    "teacher_temperature": BRIEF_TEMPERATURE,
                    "teacher_max_completion_tokens": BRIEF_MAX_COMPLETION_TOKENS,
                }
                line = json.dumps(row, ensure_ascii=False) + "\n"
                with _BRIEF_DUMP_LOCK:
                    BRIEF_DUMP_HANDLE.write(line)
                    BRIEF_DUMP_HANDLE.flush()
    return n_calls, n_briefs, per_session_meta


# ---------------------------------------------------------------- retrieval

def _filter_brief_hits(hits, top_k: int) -> list:
    out = []
    for h in hits:
        tags = getattr(h, "tags", "") or ""
        if BRIEF_TAG in tags:
            out.append(h)
            if len(out) >= top_k:
                break
    return out


def _dual_episodic(mem: vstash.Memory, question: str, collection: str, top_k: int) -> list:
    """Layer-scoped dual retrieval.

    Mirrors the four-pool interleave in
    ``cerebras_midloop.retrieve(retrieval_mode='dual')`` but pins
    ``layer='episodic'`` so briefs in the semantic layer cannot leak
    into the episodic pool. This matches the baseline RAG-k3 regime
    exactly (grid-rag_k3_seed_robustness_n30_seed44.jsonl uses
    retrieval_mode='dual', top_k=3) minus the layer filter, which
    only matters in pipelines where the mem also has briefs.

    ``q_only`` is the stable head of the query before the first
    blank line. Baseline used it to sidestep Builder-draft synonym
    drift -- keeping identical here for parity even though this
    pipeline uses the raw question.
    """
    fts_k = top_k * 3
    q_only = question.split("\n", 1)[0].strip() or question
    hybrid = mem.search(question, top_k=top_k, collection=collection, layer="episodic")
    fts = mem.search(
        question, top_k=fts_k, collection=collection, layer="episodic",
        retrieval_mode="fts_only",
    )
    fts_q = (
        mem.search(
            q_only, top_k=fts_k, collection=collection, layer="episodic",
            retrieval_mode="fts_only",
        )
        if q_only != question
        else []
    )
    vec = mem.search(
        question, top_k=top_k, collection=collection, layer="episodic",
        retrieval_mode="vec_only",
    )
    pools = [hybrid, fts, fts_q, vec]
    merged: list = []
    seen_ids: set = set()
    # Interleave by position, then drain the tail of any longer pool.
    for row in zip(*pools, strict=False):
        for h in row:
            key = (getattr(h, "title", None), (h.text or "")[:120])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            merged.append(h)
    for p in pools:
        for h in p:
            key = (getattr(h, "title", None), (h.text or "")[:120])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            merged.append(h)
    return merged[:top_k]


def _retrieve(
    mem: vstash.Memory, question: str, collection: str
) -> tuple[list, list, int]:
    """Return ``(brief_hits, episodic_hits, brief_pool_size)``.

    Episodic retrieval uses the baseline RAG-k3 dual-search regime
    (scoped to layer='episodic') so the arm-vs-baseline delta
    isolates the brief contribution. Brief retrieval is a separate
    semantic-layer search filtered to method:brief_v1.

    ``brief_pool_size`` is the size of the semantic-layer over-fetch
    before filtering. Lets us detect silent brief-retrieval collapse
    post-hoc.
    """
    brief_pool = mem.search(
        question,
        top_k=RAG_TOP_K_BRIEFS * 3,
        collection=collection,
        layer="semantic",
    )
    briefs = _filter_brief_hits(brief_pool, RAG_TOP_K_BRIEFS)
    episodic = _dual_episodic(mem, question, collection, RAG_TOP_K_EPISODIC)
    return briefs, episodic, len(brief_pool)


def _build_prompt(question: str, briefs: list, excerpts: list) -> str:
    brief_blocks = []
    for i, b in enumerate(briefs):
        text = (b.text or "").strip()
        if len(text) > RAG_MAX_BRIEF_CHARS:
            text = text[:RAG_MAX_BRIEF_CHARS] + "..."
        brief_blocks.append(f"[brief {i}]\n{text}")
    excerpt_blocks = []
    for i, e in enumerate(excerpts):
        src = getattr(e, "title", "memory") or "memory"
        text = (e.text or "").strip()
        if len(text) > RAG_MAX_EXCERPT_CHARS:
            text = text[:RAG_MAX_EXCERPT_CHARS] + "..."
        excerpt_blocks.append(f"[excerpt {i} | source={src}]\n{text}")
    parts = []
    if brief_blocks:
        parts.append("Context (curated briefs from prior sessions):\n\n" + "\n\n---\n\n".join(brief_blocks))
    if excerpt_blocks:
        parts.append("Context (raw events):\n\n" + "\n\n---\n\n".join(excerpt_blocks))
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def _call_builder(user_prompt: str, system: str) -> dict:
    t0 = time.perf_counter()
    answer, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        MAX_TOKENS_DRAFT,
        temperature=BUILDER_TEMPERATURE,
    )
    return {
        "answer": answer,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------- per-question

def run_one_question(
    conv: Conversation,
    oracle_client,
    out_stream,
    today: str,
    builder_system: str,
) -> dict:
    """Process one question end-to-end. Writes its JSONL row and
    returns the row dict for aggregation. ``today`` is pinned by
    the caller so every brief gets the same ``**As of:**`` even
    if the run straddles midnight UTC.
    """
    t_q0 = time.perf_counter()

    # Fresh vstash db per question -- isolate haystacks.
    with tempfile.TemporaryDirectory(prefix="merken_pipeline_") as td:
        db_path = Path(td) / "mem.db"
        mem = vstash.Memory(db=str(db_path))
        collection = "default"

        # 1. Ingest
        t0 = time.perf_counter()
        n_ingested = _ingest_turns(mem, conv, collection)
        ingest_s = time.perf_counter() - t0

        # 2. Per-session briefs
        t0 = time.perf_counter()
        n_brief_calls, n_briefs, brief_meta = _per_session_briefs(
            conv, mem, collection, today
        )
        brief_s = time.perf_counter() - t0

        # 3. Retrieval
        t0 = time.perf_counter()
        brief_hits, episodic_hits, brief_pool_size = _retrieve(
            mem, conv.question, collection
        )
        retrieval_s = time.perf_counter() - t0

        # 4. Builder
        prompt = _build_prompt(conv.question, brief_hits, episodic_hits)
        b = _call_builder(prompt, builder_system)

        # 4b. Append cite_footer for parity with baseline RAG-k3 grid
        # (grid-rag_k3_seed_robustness_n30_seed44.jsonl had
        # cite_footer=True). Builder under rag_baseline system does
        # not emit `[excerpt N]` tags so the footer collapses to
        # "[no excerpt citations in answer]" in ~all cases, but
        # keeping it appended preserves byte-comparable answer
        # shape with the baseline.
        episodic_as_excerpts = [
            {
                "source_id": getattr(h, "title", "memory") or "memory",
                "text": h.text or "",
            }
            for h in episodic_hits
        ]
        footer, grounded, cited_ids = _cite_footer_from_rag(
            b["answer"], episodic_as_excerpts
        )
        b["answer"] = b["answer"] + footer
        b["grounded"] = grounded
        b["cited_excerpt_ids"] = cited_ids

    # 5. Oracle
    # LongMemEval sometimes stores numeric answers as int (e.g. qid
    # 80ec1f4f answer=2, c4a1ceb8=3, 370a8ff4=15 in seed=44). The
    # shared oracle_score slices ground_truth[:2000], which crashes
    # on ints. Coerce here; we don't touch mode_a_eval.oracle_score
    # because that helper is shared with existing runs.
    ground_truth_str = str(conv.answer) if not isinstance(conv.answer, str) else conv.answer
    t_oracle = time.perf_counter()
    oracle = oracle_score(oracle_client, conv.question, ground_truth_str, b["answer"])
    oracle_s = time.perf_counter() - t_oracle

    row = {
        "qid": conv.question_id,
        "question_type": conv.question_type,
        "question": conv.question,
        "ground_truth": conv.answer,
        "answer_session_ids": conv.answer_session_ids,
        "n_haystack_sessions": conv.n_sessions,
        "n_haystack_turns": conv.n_turns,
        "n_ingested": n_ingested,
        "n_brief_calls": n_brief_calls,
        "n_briefs": n_briefs,
        "brief_per_session": brief_meta,
        "brief_pool_size": brief_pool_size,
        "n_brief_hits": len(brief_hits),
        "n_episodic_hits": len(episodic_hits),
        "brief_hits": [
            {
                "title": getattr(h, "title", None),
                "tags": getattr(h, "tags", None),
                "score": getattr(h, "score", None),
                "text": (h.text or "")[:400],
            }
            for h in brief_hits
        ],
        "episodic_hits": [
            {
                "title": getattr(h, "title", None),
                "score": getattr(h, "score", None),
                "text": (h.text or "")[:400],
            }
            for h in episodic_hits
        ],
        "builder_answer": b["answer"],
        "builder_tokens": b["total_tokens"],
        "builder_wall_s": b["wall_s"],
        "oracle": oracle,
        "wall_s_ingest": ingest_s,
        "wall_s_briefs": brief_s,
        "wall_s_retrieval": retrieval_s,
        "wall_s_oracle": oracle_s,
        "wall_s_total": time.perf_counter() - t_q0,
    }
    out_stream.write(json.dumps(row) + "\n")
    out_stream.flush()
    return row


# ---------------------------------------------------------------- driver

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="longmemeval_s")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/retrieval/longmemeval/pipeline_runs"),
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix to tag this run's JSONL (e.g. 'smoke', 'n30').",
    )
    parser.add_argument(
        "--only-qids",
        default=None,
        help="Comma-separated qids; if set, restricts to these (ignores --n/--seed sample).",
    )
    parser.add_argument(
        "--builder-system",
        choices=["baseline", "pipeline"],
        default="pipeline",
        help=(
            "Which Builder system prompt to use. 'baseline' = same as "
            "RAG-k3 (treats all context as one pool). 'pipeline' = "
            "describes briefs vs excerpts with a combining policy. "
            "Default 'pipeline' because the input shape differs from "
            "baseline RAG-k3."
        ),
    )
    parser.add_argument(
        "--dump-briefs-to",
        type=Path,
        default=None,
        help=(
            "Append every synthesized brief to this JSONL alongside "
            "the session-turn input the teacher saw. Enables the run "
            "to double as training-data collection for a local "
            "brief-synth replacement (see notes)."
        ),
    )
    args = parser.parse_args()

    if args.dump_briefs_to is not None:
        args.dump_briefs_to.parent.mkdir(parents=True, exist_ok=True)
        # Append-only across runs; also stamp provenance (run_id) so
        # dedup by (run_id, qid, sid) is trivial and resume-safe.
        import uuid
        global BRIEF_DUMP_PATH, BRIEF_DUMP_HANDLE, BRIEF_DUMP_RUN_ID
        BRIEF_DUMP_PATH = args.dump_briefs_to
        BRIEF_DUMP_RUN_ID = uuid.uuid4().hex[:12]
        # Single open handle for the life of main() -- avoids
        # O(sessions) open/close thrash and leaves one writer per
        # process (the in-process lock is the sole serializer).
        BRIEF_DUMP_HANDLE = open(BRIEF_DUMP_PATH, "a", buffering=1)
        print(
            f"[dump-briefs] appending to {BRIEF_DUMP_PATH} "
            f"run_id={BRIEF_DUMP_RUN_ID}",
            flush=True,
        )

    convs = load_longmemeval(args.subset)
    if args.only_qids:
        wanted = [q.strip() for q in args.only_qids.split(",") if q.strip()]
        index = {c.question_id: c for c in convs}
        missing = [q for q in wanted if q not in index]
        if missing:
            raise SystemExit(f"qid(s) not in subset: {missing}")
        sampled = [index[q] for q in wanted]
        print(f"[dataset] explicit qids: {[c.question_id for c in sampled]}", flush=True)
    else:
        rnd = random.Random(args.seed)
        sampled = rnd.sample(convs, min(args.n, len(convs)))
        print(
            f"[dataset] seed={args.seed} n={len(sampled)} of {len(convs)}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = args.out / f"pipeline_rag_seed{args.seed}_n{len(sampled)}{tag}_{ts}.jsonl"
    print(f"[out] {out_path}", flush=True)

    builder_system = (
        BUILDER_SYSTEM if args.builder_system == "baseline" else PIPELINE_BUILDER_SYSTEM
    )
    print(f"[builder_system] {args.builder_system}", flush=True)

    oracle_client = _oracle_client()
    today = datetime.now(timezone.utc).date().isoformat()
    print(f"[pinned] today={today}", flush=True)

    t_all = time.perf_counter()
    rows = []
    consec_errors = 0
    with out_path.open("w") as out_stream:
        for i, conv in enumerate(sampled):
            print(
                f"\n[{i+1}/{len(sampled)}] qid={conv.question_id} "
                f"type={conv.question_type} n_turns={conv.n_turns}",
                flush=True,
            )
            try:
                row = run_one_question(
                    conv, oracle_client, out_stream, today, builder_system
                )
                consec_errors = 0
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                row = {
                    "qid": conv.question_id,
                    "question_type": conv.question_type,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                out_stream.write(json.dumps(row) + "\n")
                out_stream.flush()
                rows.append(row)
                consec_errors += 1
                if consec_errors >= 2:
                    print(
                        f"\n[abort] 2 consecutive question errors; stopping to "
                        f"preserve budget. Fix the underlying issue and re-run.",
                        flush=True,
                    )
                    break
                continue
            rows.append(row)
            print(
                f"  briefs={row['n_briefs']} pool={row['brief_pool_size']} "
                f"brief_hits={row['n_brief_hits']} "
                f"episodic_hits={row['n_episodic_hits']} "
                f"oracle={row['oracle'].get('verdict')} "
                f"wall={row['wall_s_total']:.1f}s",
                flush=True,
            )

    total_wall = time.perf_counter() - t_all

    # Summary
    ok = [r for r in rows if "oracle" in r]
    n_supports = sum(1 for r in ok if r["oracle"].get("verdict") == "supports")
    n_partial = sum(1 for r in ok if r["oracle"].get("verdict") == "partial")
    n_contra = sum(1 for r in ok if r["oracle"].get("verdict") == "contradicts")
    n_neutral = sum(1 for r in ok if r["oracle"].get("verdict") == "neutral")
    n_error = sum(1 for r in rows if "error" in r)
    correct = n_supports + n_partial
    denom = len(ok) if ok else 1
    print("\n=== summary ===", flush=True)
    print(
        f"completed     : {len(ok)}/{len(rows)}  errors={n_error}",
        flush=True,
    )
    print(
        f"supports      : {n_supports}/{len(ok)}  "
        f"partial={n_partial}  contradicts={n_contra}  neutral={n_neutral}",
        flush=True,
    )
    print(
        f"correct rate  : {correct}/{len(ok)} = {correct/denom*100:.1f}%",
        flush=True,
    )
    print(f"total wall    : {total_wall:.1f}s", flush=True)
    print(f"[out] {out_path}", flush=True)
    if BRIEF_DUMP_HANDLE is not None:
        BRIEF_DUMP_HANDLE.flush()
        BRIEF_DUMP_HANDLE.close()
        print(f"[dump-briefs] closed {BRIEF_DUMP_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
