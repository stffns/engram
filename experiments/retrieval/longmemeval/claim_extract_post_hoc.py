"""Structured claim extraction post-hoc over a pipeline_runner output.

Alternative to `judge_post_hoc.py`: instead of a generic "is this
answer supported?" pass, we force the Judge to (1) enumerate events
from context, (2) apply a deterministic reasoner for the question's
shape (ordering, frequency, presence), (3) render an answer from
the reasoner output.

The hypothesis: the Judge post-hoc experiment (2026-04-24) showed
that the 3 gains on LongMemEval seed=44 merken-brief_v1 were all
multi-fact temporal reasoning fixes -- exactly the failure class a
structured enumerate-then-reason pass should hit. The single loss
was over-literal inference refusal, which structured extraction
should avoid because it doesn't gate on 'supported': it lists
events and reasons over them.

Unit test target (pre-registered): 3 qids where the generic Judge
already flipped a contradiction to a support.
- gpt4_0a05b494: who met first, jam seller or Australian tourist?
  (GT: jam seller)
- gpt4_213fd887: which event first, volleyball or charity 5K?
  (GT: volleyball league)
- f685340e_abs:  how often play table tennis? (GT: info
  insufficient; baseline confused tennis <-> table tennis)

Success criterion for the unit tests: 3/3 flip to correct, 0/3
break. Anything less means the structured approach doesn't dominate
the generic Judge on the ONE failure class it was designed for
and shouldn't scale. If it does pass, run on all 27 graded rows
and compare net to Judge post-hoc's +2.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

from experiments.midloop_concept.medlocal.cerebras_midloop import (
    JUDGE,
    cerebras_chat,
)
from experiments.retrieval.longmemeval.mode_a_eval import (
    _oracle_client,
    _parse_oracle_json,
    oracle_score,
)

MAX_TOKENS_EXTRACT = 700
EXTRACT_TEMPERATURE = 0.1

ORDERING_PATTERNS = (
    r"\bwhich.*first\b",
    r"\bwho.*first\b",
    r"\bwhat.*first\b",
    r"\bfirst\b.*(?:or|,)",
    r"\bbefore\s+(?:the|a)\b",
    r"\bearlier\b",
)
FREQUENCY_PATTERNS = (
    r"\bhow often\b",
    r"\bhow many times\b",
    r"\bhow frequently\b",
)
PRESENCE_PATTERNS = (
    # "how often do I play X" where X may not exist at all:
    # same regex as frequency. Kept separate for documentation.
)


def classify_question(q: str) -> str:
    """Return 'ordering', 'frequency', or 'other'."""
    lo = q.lower()
    for p in ORDERING_PATTERNS:
        if re.search(p, lo):
            return "ordering"
    for p in FREQUENCY_PATTERNS:
        if re.search(p, lo):
            return "frequency"
    return "other"


# Prompt that forces enumerate-then-reason.
# Note: ground_truth is NEVER in this prompt. The model only sees
# the question + retrieved context (same brief+excerpts the Builder
# saw).
EXTRACT_SYSTEM = (
    "You are a careful analyst. You will see a question and the "
    "retrieved context (briefs and excerpts) that were given to a "
    "RAG system. Your job is to answer by first enumerating the "
    "relevant events/items mentioned in the context, then applying "
    "deterministic reasoning over them.\n\n"
    "Rules:\n"
    "1. List events ONLY from the retrieved context. Do NOT invent "
    "events. If an event the question asks about is not mentioned "
    "at all, note this explicitly in the events list as "
    '{"event": "<thing>", "date": null, "mentioned": false, "source": "not in context"}.\n'
    "2. For temporal references, use ISO dates (YYYY-MM-DD) when the "
    "context gives one, otherwise the natural-language reference "
    "verbatim (e.g. 'two weeks ago', 'last Thursday').\n"
    "3. Apply the reasoner specified in the user message. Do not "
    "paraphrase the question -- follow its literal terms (e.g. "
    "'table tennis' is not 'tennis').\n"
    "4. The final 'answer' field must answer the question directly. "
    "If the context is insufficient or the queried entity isn't "
    "mentioned, say so in plain language (e.g. 'The context does "
    "not mention table tennis').\n\n"
    "Output ONLY a JSON object:\n"
    '{"question_shape": "ordering"|"frequency"|"other",\n'
    ' "events": [{"event": "...", "date": "...", "mentioned": true|false, "source": "..."}],\n'
    ' "reasoning": "<2-3 short sentences explaining the sort/count/match>",\n'
    ' "answer": "<answer under 100 words>"}\n'
    "No prose outside the JSON. No markdown fences."
)

ORDERING_USER = (
    "Question (ordering/temporal shape):\n{question}\n\n"
    "{context}\n\n"
    "Reasoner: list each candidate event with its date. Sort ALL "
    "events by date (earliest first). The earliest event that "
    "matches one of the candidate choices in the question is the "
    "answer. If a choice isn't in the context, mark it mentioned=false."
)
FREQUENCY_USER = (
    "Question (frequency/presence shape):\n{question}\n\n"
    "{context}\n\n"
    "Reasoner: identify the literal subject of 'how often' in the "
    "question. Then enumerate events in the context that match the "
    "subject LITERALLY (e.g. 'table tennis' != 'tennis'). Count "
    "them. If zero literal matches, say the context does not "
    "mention the subject; do NOT substitute a related activity."
)
OTHER_USER = (
    "Question:\n{question}\n\n"
    "{context}\n\n"
    "Reasoner: extract all events or facts in the context relevant "
    "to the question. Answer using only what is enumerated."
)

USER_BY_SHAPE = {
    "ordering": ORDERING_USER,
    "frequency": FREQUENCY_USER,
    "other": OTHER_USER,
}


def _format_hits(hits: list[dict], kind: str, per_hit_max: int) -> str:
    lines = []
    for i, h in enumerate(hits or []):
        title = (h.get("title") or "").strip()
        text = (h.get("text") or "").strip()
        if len(text) > per_hit_max:
            text = text[:per_hit_max] + " ...[truncated]"
        lines.append(f"[{kind} {i}] title={title}\n{text}")
    return "\n\n".join(lines) if lines else f"(no {kind}s)"


def build_context_block(row: dict) -> str:
    briefs = _format_hits(row.get("brief_hits") or [], "brief", 1200)
    excerpts = _format_hits(row.get("episodic_hits") or [], "excerpt", 1500)
    parts = []
    if row.get("brief_hits"):
        parts.append(f"Retrieved briefs:\n{briefs}")
    parts.append(f"Retrieved excerpts:\n{excerpts}")
    return "\n\n".join(parts)


def run_extractor(row: dict) -> dict:
    q = row.get("question") or ""
    shape = classify_question(q)
    context = build_context_block(row)[:9000]
    user_prompt = USER_BY_SHAPE[shape].format(
        question=q[:1500], context=context
    )
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    t0 = time.perf_counter()
    try:
        raw, dt, usage = cerebras_chat(
            JUDGE, messages, MAX_TOKENS_EXTRACT, temperature=EXTRACT_TEMPERATURE
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate fail-visible
        return {
            "shape": shape,
            "events": None,
            "reasoning": f"extractor_error: {exc}",
            "answer": None,
            "wall_s": time.perf_counter() - t0,
            "usage": {},
            "raw": "",
        }
    parsed = _parse_oracle_json(raw)
    if not isinstance(parsed, dict):
        parsed = {}
    out = {
        "shape": shape,
        "events": parsed.get("events"),
        "reasoning": parsed.get("reasoning"),
        "answer": parsed.get("answer"),
        "wall_s": dt,
        "usage": usage,
        "raw": raw[:2000],
    }
    if not out["answer"]:
        out["reasoning"] = (
            f"extract_parse_failure: {out.get('reasoning') or raw[:200]}"
        )
    return out


def _is_correct(verdict: str | None) -> bool:
    return verdict in {"supports", "partial"}


def _classify_flip(baseline_verdict: str | None, regrade_verdict: str | None) -> str:
    if regrade_verdict is None:
        return "noop"
    base_ok = _is_correct(baseline_verdict)
    new_ok = _is_correct(regrade_verdict)
    if base_ok == new_ok:
        return "noop"
    return "gain" if new_ok else "loss"


def process(args: argparse.Namespace) -> int:
    in_path = Path(args.pipeline_run)
    if not in_path.exists():
        print(f"pipeline-run not found: {in_path}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    graded = [r for r in rows if r.get("oracle") and r["oracle"].get("verdict")]

    if args.only_qids:
        wanted = {q.strip() for q in args.only_qids.split(",") if q.strip()}
        target = [r for r in graded if r["qid"] in wanted]
    elif args.scope == "failing":
        target = [r for r in graded if not _is_correct(r["oracle"]["verdict"])]
    elif args.scope == "supports":
        target = [r for r in graded if _is_correct(r["oracle"]["verdict"])]
    else:
        target = graded
    if args.limit is not None:
        target = target[: args.limit]
    target_qids = {r["qid"] for r in target}

    out_path = (
        Path(args.out) if args.out
        else in_path.with_name(in_path.stem + ".claim_extract.jsonl")
    )
    print(
        f"[claim-extract] in={in_path.name} graded={len(graded)} target={len(target)} "
        f"out={out_path.name}",
        flush=True,
    )

    oracle = _oracle_client()
    n_gain = 0
    n_loss = 0
    n_noop = 0
    n_regraded = 0
    n_parse_fail = 0
    shape_counts: dict[str, int] = {}
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for i, row in enumerate(rows):
            base = dict(row)
            if row.get("qid") not in target_qids:
                f.write(json.dumps(base) + "\n")
                continue
            print(
                f"  [{i+1}/{len(rows)}] qid={row.get('qid')} "
                f"baseline={row['oracle']['verdict']}",
                flush=True,
            )
            ext = run_extractor(row)
            base["claim_extract"] = ext
            shape_counts[ext["shape"]] = shape_counts.get(ext["shape"], 0) + 1
            new_ans = (ext.get("answer") or "").strip()
            orig = (row.get("builder_answer") or "").strip()
            regrade = None
            if not new_ans:
                n_parse_fail += 1
            elif new_ans != orig:
                regrade = oracle_score(
                    oracle,
                    row.get("question") or "",
                    str(row.get("ground_truth") or ""),
                    new_ans,
                )
                n_regraded += 1
            base["regrade"] = regrade
            final_verdict = (
                (regrade or {}).get("verdict")
                if regrade
                else row["oracle"]["verdict"]
            )
            flip = _classify_flip(row["oracle"]["verdict"], final_verdict)
            base["flip"] = flip
            base["final_verdict"] = final_verdict
            if flip == "gain":
                n_gain += 1
            elif flip == "loss":
                n_loss += 1
            else:
                n_noop += 1
            f.write(json.dumps(base) + "\n")
            f.flush()
    wall = time.perf_counter() - t0_all
    print(
        f"[claim-extract] done in {wall:.1f}s  regraded={n_regraded}  "
        f"gain={n_gain}  loss={n_loss}  noop={n_noop}  net={n_gain - n_loss:+d}  "
        f"parse_fail={n_parse_fail}  shapes={shape_counts}",
        flush=True,
    )
    summary = {
        "input": str(in_path),
        "scope": args.scope,
        "only_qids": args.only_qids,
        "n_target": len(target),
        "n_regraded": n_regraded,
        "n_parse_fail": n_parse_fail,
        "shape_counts": shape_counts,
        "gain": n_gain,
        "loss": n_loss,
        "noop": n_noop,
        "net": n_gain - n_loss,
        "wall_s": wall,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scope", choices=["failing", "supports", "all"], default="all")
    ap.add_argument("--only-qids", default=None,
                    help="Comma-separated qids; overrides --scope")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    return process(args)


if __name__ == "__main__":
    raise SystemExit(main())
