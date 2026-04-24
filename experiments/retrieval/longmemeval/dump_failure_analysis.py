"""One-shot dump of the 13 baseline-failing qids (seed=44 N=30)
with all ingredients needed for a per-qid failure forensics pass:
question, ground_truth, baseline_answer, baseline_verdict,
full brief_hits and episodic_hits text, plus what each post-hoc
approach produced (judge_v2, claim_extract) and the k=10 result
for the 8 forever-wrong subset.

Output is a single human-readable markdown file for inspection.
Not a scaled experiment -- pure forensics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
PIPELINE_RUNS = ENGRAM / "experiments/retrieval/longmemeval/pipeline_runs"

BASELINE = PIPELINE_RUNS / "pipeline_rag_seed44_n30_n30_20260423T074602Z.jsonl"
JUDGE    = PIPELINE_RUNS / "pipeline_rag_seed44_n30_n30_20260423T074602Z.judge_v2.jsonl"
CLAIM    = PIPELINE_RUNS / "pipeline_rag_seed44_n30_n30_20260423T074602Z.claim_extract.jsonl"
K10_8Q   = PIPELINE_RUNS / "pipeline_rag_seed44_n8_k10_forever_wrong_20260424T052925Z.jsonl"
OUT      = ENGRAM / "notes/2026-04-24-failing-qid-forensics.md"


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> int:
    base = _rows(BASELINE)
    judge = {r["qid"]: r for r in _rows(JUDGE)}
    claim = {r["qid"]: r for r in _rows(CLAIM)}
    k10 = {r["qid"]: r for r in _rows(K10_8Q)}

    failing = [
        r for r in base
        if r.get("oracle") and r["oracle"]["verdict"] in {"contradicts", "neutral"}
    ]
    failing.sort(key=lambda r: (r["oracle"]["verdict"], r["qid"]))

    out = []
    out.append("# Per-qid forensics: 13 failing qids (seed=44 N=30)\n")
    out.append(
        f"Substrate: merken brief_v1 full pipeline seed=44. "
        f"Baseline supports 14/27, non-supports 13/27 (10 contradicts + 3 neutral).\n\n"
        f"Post-hoc coverage (rows in each file): judge={len(judge)}, "
        f"claim_extract={len(claim)}, k10_subset={len(k10)}.\n"
    )
    out.append("Classification legend for final column (per approach):\n")
    out.append("- **GAIN** = flipped to supports/partial\n")
    out.append("- **noop** = stayed in failure bin (may have rewritten but regrade was not better)\n")
    out.append("- **(not tested)** = approach didn't run on this qid\n\n")

    for idx, r in enumerate(failing, 1):
        qid = r["qid"]
        out.append("---\n")
        out.append(f"## {idx}. qid=`{qid}` | baseline={r['oracle']['verdict']}\n")
        out.append(f"**Q:** {r['question']}\n")
        out.append(f"**GT:** {r['ground_truth']}\n")
        out.append(f"**Baseline answer:** {r['builder_answer'][:500]}\n")
        out.append(f"**answer_session_ids:** `{r.get('answer_session_ids')}`\n\n")

        # Post-hoc outcomes
        j = judge.get(qid)
        c = claim.get(qid)
        k = k10.get(qid)
        def _verdict(row):
            if row is None: return "(not run)"
            v = (row.get("regrade") or {}).get("verdict") or row.get("oracle", {}).get("verdict")
            fv = row.get("final_verdict") or v
            f = row.get("flip") or ""
            tag = "GAIN" if f == "gain" else "LOSS" if f == "loss" else "noop"
            if row is k:
                k_v = (row.get("oracle") or {}).get("verdict")
                is_corr = k_v in ("supports", "partial")
                tag = "GAIN" if is_corr else "noop"
                return f"{tag} (k=10 verdict: {k_v})"
            return f"{tag} (final: {fv})"
        out.append(f"- **Judge post-hoc:** {_verdict(j)}\n")
        out.append(f"- **Claim extract:** {_verdict(c)}\n")
        out.append(f"- **k=10 retrieval:** {_verdict(k)}\n\n")

        if j and j.get("judge"):
            jr = j["judge"]
            out.append(f"**Judge's rewrite:** {(jr.get('corrected_answer') or '(approve)')[:350]}\n")
            out.append(f"**Judge reasoning:** {(jr.get('reasoning') or '')[:250]}\n\n")
        if c and c.get("claim_extract"):
            cr = c["claim_extract"]
            out.append(f"**Claim extract shape:** {cr.get('shape')}\n")
            out.append(f"**Claim answer:** {(cr.get('answer') or '')[:350]}\n")
            out.append(f"**Claim reasoning:** {(cr.get('reasoning') or '')[:250]}\n\n")

        # Retrieved context
        out.append("**brief_hits (top 3):**\n")
        for i, h in enumerate(r.get("brief_hits") or []):
            out.append(f"- [{i}] `{h['title']}` score={h['score']:.4f}\n")
            out.append(f"  ```\n  {h['text'][:600]}\n  ```\n")
        out.append("\n**episodic_hits (top 3):**\n")
        for i, h in enumerate(r.get("episodic_hits") or []):
            out.append(f"- [{i}] `{h['title']}` score={h['score']:.4f}\n")
            out.append(f"  ```\n  {h['text'][:500]}\n  ```\n")
        out.append("\n")

    OUT.write_text("".join(out))
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes, {len(failing)} qids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
