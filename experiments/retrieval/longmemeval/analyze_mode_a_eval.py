"""Post-hoc analyser for ``mode_a_eval.py`` runs.

The eval script prints a top-line summary but the audit JSONL has
far more signal (per-question-type breakdown, token economics,
disagreement cases where conditions diverge, claim-level specifics).
This script produces a RESULTS.md-ready report without re-running
anything -- it is pure JSONL read + pandas-style aggregation.

Run:
  python -m experiments.retrieval.longmemeval.analyze_mode_a_eval \\
      experiments/retrieval/longmemeval/mode_a_eval_runs/n50_seed42.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = ("supports", "partial", "contradicts", "neutral")
CONDITIONS = ("control", "rag", "mode_a")


def _correct(v: str) -> bool:
    return v in ("supports", "partial")


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            # Skip error rows from the summary; they're logged for
            # debugging but do not count toward correct-rate.
            continue
        rows.append(r)
    return rows


def headline_table(rows: list[dict]) -> str:
    n = len(rows)
    lines = [f"## Headline (N={n})\n"]
    header = (
        "| condition | correct | supports | partial | contradicts | "
        "neutral | avg_tok | avg_s |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines.extend([header, sep])
    for cond in CONDITIONS:
        verdicts = [r["conditions"][cond]["oracle"]["verdict"] for r in rows]
        toks = [int(r["conditions"][cond].get("total_tokens") or 0) for r in rows]
        walls = [float(r["conditions"][cond].get("_total_s") or 0) for r in rows]
        s = verdicts.count("supports")
        p = verdicts.count("partial")
        c = verdicts.count("contradicts")
        nt = verdicts.count("neutral")
        lines.append(
            f"| {cond} | {(s+p)/n*100:.1f}% ({s+p}/{n}) | {s} | {p} | "
            f"{c} | {nt} | {sum(toks)//n} | {sum(walls)/n:.1f}s |"
        )
    return "\n".join(lines)


def by_question_type(rows: list[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r.get("question_type") or "unknown"].append(r)
    lines = ["\n## Correct rate by question_type\n"]
    lines.append("| question_type | n | control | rag | mode_a |")
    lines.append("|---|---|---|---|---|")
    for qt, qrows in sorted(grouped.items()):
        n = len(qrows)
        pcts = {}
        for cond in CONDITIONS:
            ok = sum(1 for r in qrows if _correct(r["conditions"][cond]["oracle"]["verdict"]))
            pcts[cond] = f"{ok/n*100:.0f}% ({ok}/{n})"
        lines.append(f"| {qt} | {n} | {pcts['control']} | {pcts['rag']} | {pcts['mode_a']} |")
    return "\n".join(lines)


def disagreements(rows: list[dict]) -> str:
    """Find the cases where mode_a and rag disagreed on correctness.
    Those are the interesting cells of the confusion matrix.
    """
    win_for_mode_a: list[dict] = []
    win_for_rag: list[dict] = []
    for r in rows:
        rag_ok = _correct(r["conditions"]["rag"]["oracle"]["verdict"])
        ma_ok = _correct(r["conditions"]["mode_a"]["oracle"]["verdict"])
        if ma_ok and not rag_ok:
            win_for_mode_a.append(r)
        elif rag_ok and not ma_ok:
            win_for_rag.append(r)

    lines = [
        f"\n## RAG vs Mode A disagreements (out of {len(rows)})\n",
        f"- Mode A correct, RAG wrong: {len(win_for_mode_a)}",
        f"- RAG correct, Mode A wrong: {len(win_for_rag)}",
    ]
    if win_for_mode_a:
        lines.append("\n### Mode A wins (where the Judge layer mattered)\n")
        for r in win_for_mode_a[:10]:
            qid = r["question_id"]
            qt = r.get("question_type") or "?"
            ma = r["conditions"]["mode_a"]
            rag = r["conditions"]["rag"]
            rag_v = rag["oracle"]["verdict"]
            ma_v = ma["oracle"]["verdict"]
            lines.append(f"- **{qid}** ({qt}): RAG={rag_v}, Mode A={ma_v}")
            lines.append(f"  - Q: {r['question'][:140]!r}")
            lines.append(f"  - GT: {r['ground_truth'][:140]!r}")
            lines.append(f"  - RAG answer: {rag['answer'][:140]!r}")
            lines.append(f"  - Mode A final (trimmed): {ma['answer'][:140]!r}")
    if win_for_rag:
        lines.append("\n### RAG wins (where Mode A over-corrected or refused)\n")
        for r in win_for_rag[:10]:
            qid = r["question_id"]
            qt = r.get("question_type") or "?"
            ma = r["conditions"]["mode_a"]
            rag = r["conditions"]["rag"]
            rag_v = rag["oracle"]["verdict"]
            ma_v = ma["oracle"]["verdict"]
            lines.append(f"- **{qid}** ({qt}): RAG={rag_v}, Mode A={ma_v}")
            lines.append(f"  - Q: {r['question'][:140]!r}")
            lines.append(f"  - GT: {r['ground_truth'][:140]!r}")
            lines.append(f"  - RAG answer: {rag['answer'][:140]!r}")
            lines.append(f"  - Mode A final (trimmed): {ma['answer'][:140]!r}")
    return "\n".join(lines)


def mode_a_specifics(rows: list[dict]) -> str:
    n = len(rows)
    grounded = 0
    with_leak = 0
    total_claims = 0
    unsupported_claims = 0
    judge_verdicts: Counter = Counter()
    for r in rows:
        ma = r["conditions"]["mode_a"]
        jud = ma.get("judgment", {})
        if (jud.get("quoted_evidence") or "").strip():
            grounded += 1
        claims = jud.get("claims") or []
        bad = sum(1 for c in claims if c.get("verdict") in ("contradicts", "neutral"))
        if bad > 0:
            with_leak += 1
        total_claims += len(claims)
        unsupported_claims += bad
        judge_verdicts[jud.get("verdict", "unknown")] += 1
    lines = ["\n## Mode A telemetry\n"]
    lines.append(f"- grounded (quoted_evidence present): {grounded}/{n} ({grounded/n*100:.1f}%)")
    lines.append(
        f"- claim-level leak (>=1 unsupported sub-claim): "
        f"{with_leak}/{n} ({with_leak/n*100:.1f}%)"
    )
    lines.append(f"- total sub-claims: {total_claims}")
    lines.append(
        f"- unsupported sub-claims: {unsupported_claims} "
        f"({unsupported_claims/max(total_claims,1)*100:.1f}% of all sub-claims)"
    )
    lines.append("- Judge top-level verdict distribution:")
    for v, c in judge_verdicts.most_common():
        lines.append(f"  - {v}: {c}")
    return "\n".join(lines)


def cost_table(rows: list[dict]) -> str:
    n = len(rows)
    lines = ["\n## Cost per question\n"]
    lines.append("| condition | avg_tokens | avg_wall_s | notes |")
    lines.append("|---|---|---|---|")
    for cond in CONDITIONS:
        toks = [int(r["conditions"][cond].get("total_tokens") or 0) for r in rows]
        walls = [float(r["conditions"][cond].get("_total_s") or 0) for r in rows]
        avg_tok = sum(toks) // max(n, 1)
        avg_wall = sum(walls) / max(n, 1)
        lines.append(f"| {cond} | {avg_tok} | {avg_wall:.1f}s | |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="JSONL from mode_a_eval.py")
    args = p.parse_args()

    rows = _load(args.path)
    if not rows:
        print("no completed rows; nothing to analyse")
        return 1

    report = "\n".join([
        headline_table(rows),
        by_question_type(rows),
        mode_a_specifics(rows),
        cost_table(rows),
        disagreements(rows),
    ])
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
