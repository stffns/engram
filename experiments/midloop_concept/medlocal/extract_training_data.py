"""Audit-log -> training-data extractor.

Every Mode A run (cerebras_midloop.py) writes one audit row per
question. Each row is already a labeled tuple: draft, retrieved
excerpts, judge verdict, quoted evidence, corrected text. This
script flattens those rows into three training-ready JSONL files,
one per candidate downstream model:

  1. detector_examples.jsonl -- binary claim-detection labels
     (response text + has_claim y/n). Feeds a NanoGPTClaimDetector
     style tagger.
  2. verifier_examples.jsonl -- verification labels (draft +
     excerpts + verdict + quoted_evidence). Feeds a verifier
     fine-tune, either NLI-style or generative.
  3. preference_pairs.jsonl -- DPO/RM pairs (question, chosen=
     corrected_text, rejected=draft) only for verdict=contradicts.
     Feeds a preference tuner that teaches the Builder to produce
     grounded drafts without needing the judge at inference time.

The loop is self-bootstrapping: as more Mode A queries run, the
training sets grow for free. No manual labeling, no LLM batch
job. The same audit rows that give the user provenance at runtime
give the ML pipeline labels at training time.

Usage:
  python extract_training_data.py [audit_log.jsonl] [--out-dir ...]

Defaults:
  input:  cerebras_smoke.jsonl (next to this script)
  output: ./training_data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_audit(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"skip line {i}: {e}")
    return rows


def emit_detector_examples(rows: list[dict]) -> list[dict]:
    """One example per audit row: (draft, has_claim).

    The judge's ``has_claim`` boolean (plus verdict as a richer
    signal) becomes the label. A detector trained on this tells
    you up-front whether a draft is worth retrieving for, skipping
    the Judge call when has_claim is false in production.
    """
    out: list[dict] = []
    for r in rows:
        j = r.get("judgment") or {}
        has_claim = bool(j.get("has_claim", False))
        # verdict="no_claim" is an explicit negative; all others
        # imply the detector should have fired even if the verifier
        # disagreed downstream.
        verdict = j.get("verdict", "neutral")
        if verdict == "no_claim":
            has_claim = False
        out.append({
            "audit_id": r.get("audit_id"),
            "text": r.get("draft", ""),
            "has_claim": has_claim,
            "verdict_hint": verdict,
            "source_run": {
                "builder": r.get("builder_model"),
                "judge": r.get("judge_model"),
                "timestamp": r.get("timestamp"),
            },
        })
    return out


def emit_verifier_examples(rows: list[dict]) -> list[dict]:
    """One example per audit row: (question, draft, excerpts,
    verdict, quoted_evidence, corrected_text).

    This is the fattest label of the three; it carries everything
    a verifier needs to learn both the VERDICT task and the
    QUOTED-EVIDENCE extraction task. Skip rows where retrieval
    returned nothing (nothing to verify against).
    """
    out: list[dict] = []
    for r in rows:
        excerpts = r.get("retrieved") or []
        if not excerpts:
            continue
        j = r.get("judgment") or {}
        verdict = j.get("verdict")
        if verdict not in ("supports", "contradicts", "neutral"):
            continue
        out.append({
            "audit_id": r.get("audit_id"),
            "question": r.get("question", ""),
            "draft": r.get("draft", ""),
            "excerpts": excerpts,
            "verdict": verdict,
            "cited_excerpt_ids": j.get("cited_excerpt_ids") or [],
            "quoted_evidence": j.get("quoted_evidence", ""),
            "corrected_text": j.get("corrected_text") or "",
            "source_run": {
                "builder": r.get("builder_model"),
                "judge": r.get("judge_model"),
                "timestamp": r.get("timestamp"),
            },
        })
    return out


def emit_preference_pairs(rows: list[dict]) -> list[dict]:
    """One pair per contradicts row: chosen=corrected, rejected=draft.

    DPO / preference tuning on these pairs teaches the Builder to
    emit outputs closer to the grounded correction without needing
    the judge-retrieve-rewrite loop at inference time. This is the
    most powerful use of the audit data: it closes the loop by
    updating the Builder from its own corrected runs.

    Filter out cases where corrected_text is empty or too short
    (the judge occasionally returns an empty string on
    neutral-adjacent contradicts); those are noise, not signal.
    """
    out: list[dict] = []
    for r in rows:
        j = r.get("judgment") or {}
        if j.get("verdict") != "contradicts":
            continue
        corrected = (j.get("corrected_text") or "").strip()
        if len(corrected) < 20:
            continue
        draft = (r.get("draft") or "").strip()
        if not draft or corrected == draft:
            continue
        out.append({
            "audit_id": r.get("audit_id"),
            "prompt": r.get("question", ""),
            "chosen": corrected,
            "rejected": draft,
            "source_run": {
                "builder": r.get("builder_model"),
                "judge": r.get("judge_model"),
                "timestamp": r.get("timestamp"),
            },
            "grounding": {
                "cited_excerpt_ids": j.get("cited_excerpt_ids") or [],
                "quoted_evidence": j.get("quoted_evidence", ""),
            },
        })
    return out


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main() -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_log", nargs="?", type=Path,
                    default=here / "cerebras_smoke.jsonl")
    ap.add_argument("--out-dir", type=Path, default=here / "training_data")
    args = ap.parse_args()

    if not args.audit_log.exists():
        sys.exit(f"audit log not found: {args.audit_log}")

    rows = load_audit(args.audit_log)
    if not rows:
        sys.exit(f"audit log is empty: {args.audit_log}")
    print(f"loaded {len(rows)} audit rows from {args.audit_log}")

    detector = emit_detector_examples(rows)
    verifier = emit_verifier_examples(rows)
    preference = emit_preference_pairs(rows)

    n_det = write_jsonl(args.out_dir / "detector_examples.jsonl", detector)
    n_ver = write_jsonl(args.out_dir / "verifier_examples.jsonl", verifier)
    n_pref = write_jsonl(args.out_dir / "preference_pairs.jsonl", preference)

    print(f"\nwrote {args.out_dir}/")
    print(f"  detector_examples.jsonl   {n_det} examples")
    print(f"  verifier_examples.jsonl   {n_ver} examples")
    print(f"  preference_pairs.jsonl    {n_pref} examples")

    # --- distribution sanity ---------------------------------------
    verdict_counts: dict[str, int] = {}
    for r in rows:
        v = (r.get("judgment") or {}).get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print("\nverdict distribution in source audit:")
    for v, c in sorted(verdict_counts.items()):
        print(f"  {v:15s} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
