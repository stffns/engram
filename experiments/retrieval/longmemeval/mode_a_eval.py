"""Honest Mode A answer-quality eval on LongMemEval.

The existing runner (``runner.py``) measures R@k -- did retrieval
surface a chunk from the answer session. That is one piece of the
Mode A pipeline (``retrieve()``) and tells us whether the substrate
is finding the right chunk, not whether the final user-facing
answer is correct.

This script extends the eval to answer quality across three
conditions per question:

  1. ``control`` -- Builder only, no context. Baseline for
     "what does the small model know unaided?"
  2. ``rag`` -- Builder + top-k chunks from the SAME dual-3
     retrieval Mode A uses, inlined into the user prompt. Baseline
     for "what does the substrate buy us without a Judge?"
  3. ``mode_a`` -- the full Mode A v4 pipeline: draft + dual-3 +
     claim-level Judge + deterministic annotator.

Scoring: Gemini 2.5 Flash as LLM-as-judge on
``(question, ground_truth, candidate)`` triples. The oracle
model is deliberately a different family (Google) from the
Builder/Judge (Cerebras) to avoid intra-family bias.

Metrics per condition:
- ``correct_rate = (supports + partial) / total``
- Mode A extras: ``grounded_rate`` (fraction with verbatim
  ``quoted_evidence``), ``claims_unsupported_rate`` (fraction
  with >= 1 ``contradicts``/``neutral`` sub-claim).
- ``avg_tokens``, ``avg_wall_s``.

Cost profile (N=50):
- ~150 Builder calls (cheap, Cerebras).
- ~50 Judge calls (Mode A only; Cerebras).
- ~150 oracle calls (Gemini 2.5 Flash; cents).
- Ingestion is local vstash (no API cost).
- Budget estimate: ~\\$5-15 total.

Run:
  python -m experiments.retrieval.longmemeval.mode_a_eval \\
      --subset longmemeval_s --n 3 --seed 42    # preview
  python -m experiments.retrieval.longmemeval.mode_a_eval \\
      --subset longmemeval_s --n 50 --seed 42   # full

Output: one JSONL row per (question, condition) plus a printed
summary table. Each row carries everything needed to reconstruct
the decision later -- retrieval pool with signals, Judge
verdict + claims, oracle verdict, wall + tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import vstash

# We mutate the module-level JUDGE_SYSTEM before calling judge_once
# so the Judge sees the "personal" frame that matches the
# conversational LongMemEval haystacks. The alternative of adding
# a new DOMAIN_FRAMES entry is saved for when we actually have a
# different-enough domain.
from experiments.midloop_concept.medlocal import cerebras_midloop as _mod

# Reuse the validated Mode A pipeline parts. cerebras_midloop is the
# authoritative implementation; we import its helpers rather than
# duplicating them so a future change to (say) the Judge prompt
# reaches this eval automatically.
from experiments.midloop_concept.medlocal.cerebras_midloop import (
    BUILDER,
    BUILDER_MODES,
    DOMAIN_FRAMES,
    JUDGE_SYSTEM_TEMPLATE,
    MAX_TOKENS_DRAFT,
    annotate_deterministic,
    cerebras_chat,
    judge_once,
    retrieval_stats,
    retrieve,
)
from experiments.retrieval.longmemeval.dataset import (
    Conversation,
    load_longmemeval,
)

# Reuse runner._format_turn so ingestion is bit-identical to the R@k
# benchmark -- in particular it strips tiktoken special tokens
# (<|endoftext|> etc) that crash vstash's chunk_text on some
# LongMemEval haystacks. Without this shared helper our rows would
# not be comparable to the published R@k numbers AND we would lose
# full questions to ingest crashes (caught by the code-reviewer
# pass before the N=50 run).
from experiments.retrieval.longmemeval.runner import (
    _format_turn as _format_turn_from_runner,
)

ORACLE_MODEL = "gemini-2.5-flash"
TOP_K = 5

# Oracle prompt. Short, strict JSON, zero chain-of-thought so the
# scoring is as close to deterministic as possible. Four-way verdict
# (supports / partial / contradicts / neutral) maps cleanly to
# correctness: supports+partial = correct enough, contradicts+neutral
# = incorrect or evasive.
ORACLE_PROMPT = """You are a strict grader scoring answer quality.

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
No prose outside the JSON. No markdown fences.
"""

# Builder system prompt for the RAG baseline. Keeps the Builder
# confident about the injected context so the RAG-only branch
# actually uses the retrieved chunks rather than hedging them away.
RAG_BUILDER_SYSTEM = (
    "Answer the user question using the provided context. "
    "Quote specific numbers or names from the context when they "
    "appear. Keep the answer under 150 words. Do not hedge."
)


# --------------------------------------------------------------------- oracle


def _oracle_client():
    # Lazy import so `--help` works without google-genai. The repo
    # already uses gemini elsewhere; same env var fallback chain.
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY required")
    return genai.Client(api_key=key)


def _parse_oracle_json(raw: str) -> dict:
    """Same pattern as cerebras_midloop's _extract_json_object --
    tolerate markdown fences and trailing prose by finding the
    outermost {...} and parsing it.
    """
    text = raw.strip()
    m = re.match(r"```(?:json)?\s*", text)
    if m:
        text = text[m.end():]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"verdict": "neutral", "rationale": "oracle_parse_failure", "raw": raw[:200]}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"verdict": "neutral", "rationale": "oracle_parse_failure", "raw": raw[:200]}


def oracle_score(client, question: str, ground_truth: str, candidate: str) -> dict:
    """Return {verdict, rationale, wall_s}. Failures default to
    ``neutral`` so an oracle outage does not silently inflate any
    condition's correct_rate.
    """
    prompt = ORACLE_PROMPT.format(
        question=question[:1500],
        ground_truth=ground_truth[:2000],
        candidate=candidate[:2000],
    )
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(model=ORACLE_MODEL, contents=prompt)
    except Exception as exc:  # noqa: BLE001 -- fail-closed deliberately
        return {
            "verdict": "neutral",
            "rationale": f"oracle_error: {exc}",
            "wall_s": time.perf_counter() - t0,
        }
    parsed = _parse_oracle_json(resp.text or "")
    parsed["wall_s"] = time.perf_counter() - t0
    return parsed


# --------------------------------------------------------------------- ingestion


def _ingest(mem: vstash.Memory, conv: Conversation) -> int:
    """Ingest every turn of every session as a separate memory
    item, using ``runner._format_turn`` so the ingested shape
    (including the special-token strip) matches the R@k benchmark.
    Title encodes session_id so downstream code can trace hits
    back to the answer session. Returns number of items ingested.
    """
    n = 0
    for sid, turns in conv.haystack_sessions.items():
        for i, turn in enumerate(turns):
            mem.remember(
                _format_turn_from_runner(turn),
                title=f"{conv.question_id}::{sid}::{i}",
                collection="default",
            )
            n += 1
    return n


# --------------------------------------------------------------------- conditions


def run_control(question: str) -> dict:
    """Builder-only baseline. No retrieval, no context."""
    t0 = time.perf_counter()
    draft, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": BUILDER_MODES["confident"] or ""},
            {"role": "user", "content": question},
        ],
        MAX_TOKENS_DRAFT,
    )
    return {
        "answer": draft,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }


def run_rag(mem: vstash.Memory, question: str) -> dict:
    """Naive RAG: same dual-3 retrieval Mode A uses, but the
    Builder sees the chunks directly and produces the final
    answer. No Judge, no claim-level. This is what people
    usually mean by "RAG".
    """
    t0 = time.perf_counter()
    excerpts = retrieve(mem, question, top_k=TOP_K, retrieval_mode="dual")
    stats = retrieval_stats(excerpts)
    # Truncate each excerpt so the RAG prompt stays within the
    # Builder's effective context. Same cap Mode A uses in its
    # draft step (ballpark).
    joined = "\n\n---\n\n".join(
        f"[source={e['source_id']}]\n{e['text'][:800]}"
        for e in excerpts
    )
    user = f"Context:\n{joined}\n\nQuestion: {question}"
    answer, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": RAG_BUILDER_SYSTEM},
            {"role": "user", "content": user},
        ],
        MAX_TOKENS_DRAFT,
    )
    return {
        "answer": answer,
        "retrieved": excerpts,
        "retrieval_stats": stats,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }


def run_mode_a(mem: vstash.Memory, question: str) -> dict:
    """Full Mode A v4. Replicates cerebras_midloop.run_one but
    returns the raw components so we can build a uniform audit row
    across conditions. Using the personal domain frame because
    LongMemEval haystacks are conversational project memory, not
    clinical guidelines.
    """
    t0 = time.perf_counter()

    # Builder draft (confident mode, matches the smoke runs).
    b_sys = BUILDER_MODES["confident"]
    draft, b_dt, b_usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": b_sys},
            {"role": "user", "content": question},
        ],
        MAX_TOKENS_DRAFT,
    )

    # Retrieval (3-way dual by default).
    query = f"{question}\n{draft[:400]}"
    excerpts = retrieve(mem, query, top_k=TOP_K, retrieval_mode="dual")
    stats = retrieval_stats(excerpts)

    # Point the Judge at the personal domain frame for this eval.
    # Mutated once globally per call; the Judge prompt is idempotent
    # across runs so setting it every iteration is cheap.
    _mod.JUDGE_SYSTEM = JUDGE_SYSTEM_TEMPLATE.format(
        domain_frame=DOMAIN_FRAMES["personal"]
    )
    judgment, j_dt, j_usage = judge_once(question, draft, excerpts)

    final = annotate_deterministic(draft, judgment, excerpts)

    claims = judgment.get("claims") or []
    bad_claims = [
        c for c in claims
        if isinstance(c, dict) and c.get("verdict") in ("contradicts", "neutral")
    ]
    total_tokens = (b_usage.get("total_tokens", 0) or 0) + (j_usage.get("total_tokens", 0) or 0)
    return {
        "answer": final,
        "draft": draft,
        "retrieved": excerpts,
        "retrieval_stats": stats,
        "judgment": judgment,
        "builder_usage": b_usage,
        "judge_usage": j_usage,
        "draft_s": b_dt,
        "judge_s": j_dt,
        "total_tokens": total_tokens,
        "n_sub_claims": len(claims),
        "n_sub_claims_unsupported": len(bad_claims),
        "_total_s": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------- eval loop


@dataclass
class EvalConfig:
    subset: str
    n: int
    seed: int
    out_path: Path
    top_k: int = TOP_K
    keep_dbs: bool = False


def run_eval(cfg: EvalConfig) -> list[dict]:
    print(f"[config] subset={cfg.subset} n={cfg.n} seed={cfg.seed} out={cfg.out_path}")
    conversations = load_longmemeval(subset=cfg.subset)
    print(f"[dataset] loaded {len(conversations)} conversations")

    rnd = random.Random(cfg.seed)
    sampled = rnd.sample(conversations, min(cfg.n, len(conversations)))
    print(f"[dataset] sampled {len(sampled)} for eval")

    oracle = _oracle_client()
    rows: list[dict] = []

    tmp_root = Path.home() / ".merken" / f"longmemeval_mode_a_{cfg.seed}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = cfg.out_path.open("w")

    try:
        for i, conv in enumerate(sampled):
            # Some LongMemEval answers are numbers (int/float). The
            # Conversation dataclass annotates ``answer: str`` but the
            # loader just passes through whatever is in the JSON.
            # Coerce here so both the debug print and the oracle prompt
            # (via ground_truth[:2000]) work uniformly. The coerced
            # string is what ships to the oracle and into the audit,
            # so the comparison is faithful.
            gt_text = str(conv.answer) if conv.answer is not None else ""
            print(
                f"\n{'='*72}\n[{i+1}/{len(sampled)}] qid={conv.question_id} "
                f"type={conv.question_type}\n{'='*72}"
            )
            print(f"Q: {conv.question[:200]}")
            print(f"GT: {gt_text[:200]}")

            # Fresh DB per question so haystacks don't pollute each
            # other. Project tag is the qid so vstash internal
            # filtering stays predictable.
            db_path = tmp_root / f"{conv.question_id}.db"
            if db_path.exists():
                db_path.unlink()
            mem = vstash.Memory(
                db=str(db_path),
                project=conv.question_id,
                collection="default",
            )
            # Wrap per-question body so a single Cerebras 5xx /
            # Gemini hiccup / ingest crash does NOT abort the
            # remaining rows after we've already paid oracle budget
            # on earlier ones. A failed question writes a stub row
            # with the error so the summary can still account for
            # it and a resume-from-log is possible.
            try:
                t_ingest = time.perf_counter()
                n_ingested = _ingest(mem, conv)
                ingest_s = time.perf_counter() - t_ingest
                print(f"[ingest] {n_ingested} turns in {ingest_s:.1f}s")

                # Three conditions, same question, same mem for the
                # two that need retrieval.
                print("[control]")
                r_control = run_control(conv.question)
                print(f"  answer: {r_control['answer'][:180]!r}")

                print("[rag]")
                r_rag = run_rag(mem, conv.question)
                print(f"  answer: {r_rag['answer'][:180]!r}")

                print("[mode_a]")
                r_mode_a = run_mode_a(mem, conv.question)
                v = r_mode_a["judgment"].get("verdict")
                n_claims = r_mode_a["n_sub_claims"]
                n_bad = r_mode_a["n_sub_claims_unsupported"]
                print(
                    f"  verdict={v} sub_claims={n_claims} unsupported={n_bad}"
                )

                # Oracle each answer against the ground truth.
                print("[oracle]")
                o_control = oracle_score(
                    oracle, conv.question, gt_text, r_control["answer"]
                )
                o_rag = oracle_score(
                    oracle, conv.question, gt_text, r_rag["answer"]
                )
                o_mode_a = oracle_score(
                    oracle, conv.question, gt_text, r_mode_a["answer"]
                )
                print(
                    f"  control={o_control['verdict']:10s} "
                    f"rag={o_rag['verdict']:10s} "
                    f"mode_a={o_mode_a['verdict']:10s}"
                )

                audit = {
                    "audit_id": uuid.uuid4().hex[:12],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt_text,
                    "answer_session_ids": conv.answer_session_ids,
                    "n_sessions_ingested": n_ingested,
                    "ingest_s": ingest_s,
                    "conditions": {
                        "control": {**r_control, "oracle": o_control},
                        "rag": {**r_rag, "oracle": o_rag},
                        "mode_a": {**r_mode_a, "oracle": o_mode_a},
                    },
                }
                fout.write(json.dumps(audit, default=str) + "\n")
                fout.flush()
                rows.append(audit)
            except Exception as exc:  # noqa: BLE001 -- deliberate catch-all
                print(f"[error] question {conv.question_id} failed: {exc!r}")
                err_row = {
                    "audit_id": uuid.uuid4().hex[:12],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt_text,
                    "error": repr(exc),
                }
                fout.write(json.dumps(err_row, default=str) + "\n")
                fout.flush()
                # Do NOT append to rows -- the summary counts
                # completed questions only. The stub row in the log
                # is enough for later debugging.
            finally:
                mem.close()
                if not cfg.keep_dbs and db_path.exists():
                    db_path.unlink()
    finally:
        fout.close()

    return rows


# --------------------------------------------------------------------- summary


def _correct(v: str) -> bool:
    return v in ("supports", "partial")


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        print("no rows -- nothing to summarise")
        return

    conditions = ["control", "rag", "mode_a"]
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  n questions: {n}")
    print()
    print(f"  {'condition':10s}  "
          f"{'correct':>8s}  {'supports':>9s}  {'partial':>8s}  "
          f"{'contradicts':>12s}  {'neutral':>8s}  "
          f"{'avg_tok':>8s}  {'avg_s':>6s}")
    for cond in conditions:
        verdicts = [r["conditions"][cond]["oracle"]["verdict"] for r in rows]
        tok = [int(r["conditions"][cond].get("total_tokens") or 0) for r in rows]
        wall = [float(r["conditions"][cond].get("_total_s") or 0) for r in rows]
        supports = verdicts.count("supports")
        partial = verdicts.count("partial")
        contradicts = verdicts.count("contradicts")
        neutral = verdicts.count("neutral")
        correct = supports + partial
        print(
            f"  {cond:10s}  "
            f"{correct/n*100:6.1f}%  {supports:>9d}  {partial:>8d}  "
            f"{contradicts:>12d}  {neutral:>8d}  "
            f"{sum(tok)//n:>8d}  {sum(wall)/n:>5.1f}s"
        )

    print()
    mode_a_rows = [r["conditions"]["mode_a"] for r in rows]
    grounded = sum(
        1 for r in mode_a_rows
        if (r.get("judgment", {}).get("quoted_evidence") or "").strip()
    )
    with_bad = sum(
        1 for r in mode_a_rows
        if (r.get("n_sub_claims_unsupported") or 0) > 0
    )
    print(
        f"  mode_a grounded (has quoted_evidence): "
        f"{grounded}/{n} ({grounded/n*100:.1f}%)"
    )
    print(
        f"  mode_a claim-level leaks (>=1 unsupported sub-claim): "
        f"{with_bad}/{n} ({with_bad/n*100:.1f}%)"
    )


# --------------------------------------------------------------------- cli


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="longmemeval_s",
                   help="LongMemEval subset (default longmemeval_s)")
    p.add_argument("--n", type=int, default=3,
                   help="number of questions to sample (default 3 for preview)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/retrieval/longmemeval/mode_a_eval_runs"),
        help="directory for audit logs; filename includes n+seed",
    )
    p.add_argument(
        "--keep-dbs", action="store_true",
        help="keep per-question vstash dbs after the run (default: cleanup)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"n{args.n}_seed{args.seed}.jsonl"
    cfg = EvalConfig(
        subset=args.subset,
        n=args.n,
        seed=args.seed,
        out_path=out_path,
        keep_dbs=args.keep_dbs,
    )
    rows = run_eval(cfg)
    summarize(rows)
    print(f"\nlog: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
