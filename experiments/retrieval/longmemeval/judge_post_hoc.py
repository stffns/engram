"""Judge post-hoc over a pipeline_runner output.

Takes the per-row JSONL emitted by
``experiments/retrieval/longmemeval/pipeline_runner.py``, runs a
claim-grounded Judge pass over (question, builder_answer,
episodic_hits[3]) that either approves the existing answer or
produces a corrected one grounded in the same 3 chunks, and re-grades
the corrected answer with the same Gemini oracle the baseline used.

Design rules:
- The Judge only sees what the Builder saw. It cannot re-retrieve.
  The hypothesis under test is "reasoning over the same k=3 is the
  bottleneck", not "we need better retrieval".
- Ground truth is NEVER in the Judge prompt. That would leak the
  answer and make the experiment meaningless.
- Judge model is Cerebras qwen-3-235b (same JUDGE constant as
  cerebras_midloop). Oracle is gemini-2.5-flash (same as baseline).
  This mirrors the Mode A v4 family so deltas are comparable.
- Output schema is additive: we keep the original row fields so the
  file is a superset of the baseline.

Decision criteria (pre-registered, in notes/2026-04-23-retrieval-
not-bottleneck.md companion entry). "Correct" = {supports, partial}.
- Gain = baseline not-correct AND final-verdict correct.
- Loss = baseline correct AND final-verdict not-correct.
- Net >= +3 -> strong signal, scale to 3 seeds.
- Net +1..+2 -> marginal, do not invest further without another angle.
- Net <= 0 -> rejected; builder-level reasoning is not the bottleneck.
"""

from __future__ import annotations

import argparse
import json
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

MAX_TOKENS_JUDGE = 600
JUDGE_TEMPERATURE = 0.1

JUDGE_SYSTEM = (
    "You are a strict post-hoc reviewer of a RAG system's answer. "
    "You will see the user's question, the retrieved context the "
    "original answerer saw (briefs and/or excerpts), and the answer "
    "that was produced. Your job:\n\n"
    "1. Decide whether the answer is SUPPORTED by the retrieved "
    "context. An answer is supported when the briefs or excerpts "
    "contain the specific fact the question asks for AND the "
    "answer reflects it. Facts may be spread across multiple "
    "briefs/excerpts -- stitching them is allowed.\n"
    "2. If supported, echo the original answer verbatim.\n"
    "3. If not supported, produce a corrected answer grounded in "
    "the retrieved context. Quote specific numbers, names, and "
    "dates verbatim. If the retrieved context does not contain "
    "the answer, respond exactly 'not found in memory' rather "
    "than guessing.\n"
    "4. Keep corrected answers under 150 words. Do not add "
    "disclaimers. Do not hedge outside of rule 3.\n\n"
    "Output ONLY a JSON object:\n"
    '{"supported": true|false, "corrected_answer": "<string>", '
    '"reasoning": "<one short sentence>"}\n'
    "No prose outside the JSON. No markdown fences."
)

JUDGE_USER_TEMPLATE = """Question:
{question}

{briefs_block}Retrieved excerpts (the original answerer saw these 3):
{excerpts}

Original answer:
{answer}
"""


def _format_hits(hits: list[dict], kind: str, per_hit_max_chars: int) -> str:
    lines = []
    for i, h in enumerate(hits or []):
        title = (h.get("title") or "").strip()
        text = (h.get("text") or "").strip()
        if len(text) > per_hit_max_chars:
            text = text[:per_hit_max_chars] + " ...[truncated]"
        lines.append(f"[{kind} {i}] title={title}\n{text}")
    return "\n\n".join(lines) if lines else f"(no {kind}s)"


def run_judge(row: dict) -> dict:
    """Run the Judge pass over a pipeline row. Return a dict with
    {supported, corrected_answer, reasoning, wall_s, usage, raw}.

    Judge sees the SAME context the Builder saw: brief_hits (if any)
    plus episodic_hits. Never ground_truth.
    """
    briefs = _format_hits(row.get("brief_hits") or [], "brief", 1200)
    excerpts = _format_hits(row.get("episodic_hits") or [], "excerpt", 1500)
    briefs_block = (
        f"Retrieved briefs (temporal summaries, the original answerer also saw these):\n"
        f"{briefs}\n\n"
        if row.get("brief_hits")
        else ""
    )
    user = JUDGE_USER_TEMPLATE.format(
        question=(row.get("question") or "")[:1500],
        briefs_block=briefs_block[:6000],
        excerpts=excerpts[:8000],
        answer=(row.get("builder_answer") or "")[:2000],
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    t0 = time.perf_counter()
    try:
        raw, dt, usage = cerebras_chat(
            JUDGE, messages, MAX_TOKENS_JUDGE, temperature=JUDGE_TEMPERATURE
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate fail-visible
        return {
            "supported": None,
            "corrected_answer": None,
            "reasoning": f"judge_error: {exc}",
            "wall_s": time.perf_counter() - t0,
            "usage": {},
            "raw": "",
        }
    parsed = _parse_oracle_json(raw)
    if not isinstance(parsed, dict):
        parsed = {}
    # Coerce fields with defensive defaults.
    out = {
        "supported": bool(parsed.get("supported")) if "supported" in parsed else None,
        "corrected_answer": parsed.get("corrected_answer"),
        "reasoning": parsed.get("reasoning"),
        "wall_s": dt,
        "usage": usage,
        "raw": raw[:2000],
    }
    if out["supported"] is None:
        out["reasoning"] = f"judge_parse_failure: {out.get('reasoning') or raw[:200]}"
    return out


def _is_correct(verdict: str | None) -> bool:
    return verdict in {"supports", "partial"}


def _classify_flip(baseline_verdict: str | None, regrade_verdict: str | None) -> str:
    """Three-way flip classification. 'noop' covers both 'same bin'
    and 'judge didn't rewrite so we didn't regrade'."""
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
    # Only process rows that had a baseline oracle verdict. Rows that
    # never got graded stay pass-through (copied verbatim).
    graded = [r for r in rows if r.get("oracle") and r["oracle"].get("verdict")]
    if args.scope == "failing":
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
        else in_path.with_name(in_path.stem + ".judge.jsonl")
    )
    print(
        f"[judge-post-hoc] in={in_path.name} n_rows={len(rows)} graded={len(graded)} "
        f"scope={args.scope} target={len(target)} out={out_path.name}",
        flush=True,
    )

    oracle = _oracle_client()
    written: list[dict] = []
    n_gain = 0
    n_loss = 0
    n_noop = 0
    n_rewritten = 0
    n_judge_parse_fail = 0
    n_judge_approve = 0
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for i, row in enumerate(rows):
            base = dict(row)  # shallow copy; we mutate nothing on input
            if row.get("qid") not in target_qids:
                f.write(json.dumps(base) + "\n")
                continue
            print(
                f"  [{i+1}/{len(rows)}] qid={row.get('qid')} "
                f"baseline={row['oracle']['verdict']}",
                flush=True,
            )
            judged = run_judge(row)
            base["judge"] = judged
            if judged.get("supported") is None:
                n_judge_parse_fail += 1
            elif judged.get("supported") is True:
                n_judge_approve += 1
            # Regrade iff the Judge proposed a corrected answer AND
            # it differs from the original. If Judge says supported,
            # we take it at its word and do not burn a Gemini call.
            corrected = (judged.get("corrected_answer") or "").strip()
            orig = (row.get("builder_answer") or "").strip()
            regrade = None
            if (
                judged.get("supported") is False
                and corrected
                and corrected != orig
            ):
                regrade = oracle_score(
                    oracle,
                    row.get("question") or "",
                    str(row.get("ground_truth") or ""),
                    corrected,
                )
                n_rewritten += 1
            base["regrade"] = regrade
            # Effective final verdict: regrade if any, else baseline
            # (Judge-approve path keeps baseline).
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
    n_rewrite_same_bin = n_rewritten - n_gain - n_loss
    print(
        f"[judge-post-hoc] done in {wall:.1f}s  rewrites_regraded={n_rewritten}  "
        f"gain={n_gain}  loss={n_loss}  noop={n_noop}  net={n_gain - n_loss:+d}  "
        f"approve={n_judge_approve}  parse_fail={n_judge_parse_fail}  "
        f"rewrite_same_bin={n_rewrite_same_bin}",
        flush=True,
    )
    # Also emit a tiny summary json next to the jsonl for fast grep.
    summary = {
        "input": str(in_path),
        "scope": args.scope,
        "n_target": len(target),
        "n_rewritten_regraded": n_rewritten,
        "n_judge_approve": n_judge_approve,
        "n_judge_parse_fail": n_judge_parse_fail,
        "n_rewrite_same_bin": n_rewrite_same_bin,
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
    ap.add_argument("--pipeline-run", required=True,
                    help="Path to a pipeline_runner .jsonl output")
    ap.add_argument("--out", default=None,
                    help="Output jsonl path (default: <input>.judge.jsonl)")
    ap.add_argument("--scope", choices=["failing", "supports", "all"], default="all",
                    help="Which graded rows to re-judge")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of target rows (smoke mode)")
    args = ap.parse_args()
    return process(args)


if __name__ == "__main__":
    raise SystemExit(main())
