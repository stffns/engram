"""Scale up Phase 1 on HF-guidelines chunks.

Picks N chunks from ``staging/hf_guidelines/chunks/{who,icrc}/`` via
the sample_case_gen selector, then runs the full Phase 1 pipeline on
them:

  1. case_generator (Gemini 2.5 Flash, structured-output) -> cases.jsonl
  2. response_generator (lmstudio gemma-4-e4b-it-mlx)     -> responses.jsonl
  3. aligner (fastembed BAAI/bge-small-en-v1.5)           -> aligned.jsonl
  4. format_for_training (boundary labels)                -> midloop_v1_hf.jsonl

Resumable via ``--skip``; each step checks for its output file and
skips if present (mirror of run_pilot.py).

Output dir defaults to ``scaleup_out_hf/`` under this package. Kept
separate from ``out/`` and ``scaleup_out/`` so the existing v0
artifacts stay untouched.

Usage:
  GOOGLE_API_KEY=... python -m experiments.midloop_pilot.scaleup_phase1_hf \
      --n-chunks 300 --n-cases 5

  python -m experiments.midloop_pilot.scaleup_phase1_hf --skip cases align
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Reuse selection + parsing from the 10-chunk sample script.
from experiments.midloop_pilot.sample_case_gen import select_chunks

HERE = Path(__file__).parent
OUT_DIR = HERE / "scaleup_out_hf"
LMSTUDIO_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "gemma-4-e4b-it-mlx"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_S = 90.0
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def step_select(
    out_dir: Path, n_chunks: int, seed: int, min_chars: int, max_chars: int,
    max_per_doc: int = 1,
) -> Path:
    """Persist the selected chunks as 'protocols.jsonl' so the rest of
    the pipeline reads the same format run_pilot expects."""
    path = out_dir / "protocols.jsonl"
    who_target = round(n_chunks * 0.7)
    icrc_target = n_chunks - who_target
    source_mix = {"who": who_target, "icrc": icrc_target}
    chunks = select_chunks(
        n=n_chunks, source_mix=source_mix,
        min_chars=min_chars, max_chars=max_chars, seed=seed,
        max_per_doc=max_per_doc,
    )
    print(f"  selected {len(chunks)} chunks "
          f"(who={sum(1 for c in chunks if c['source']=='who')}, "
          f"icrc={sum(1 for c in chunks if c['source']=='icrc')})")
    if not chunks:
        raise SystemExit("no chunks selected; did you run chunk_hf_guidelines?")
    with path.open("w") as f:
        for c in chunks:
            slug = re.sub(r"[^A-Za-z0-9]+", "-", c["heading"][:40].lower()).strip("-")
            protocol_id = f"{c['source']}-hf-{c['doc_id'][:8]}-{slug}"[:80]
            row = {
                "protocol_id": protocol_id,
                "text": c["body"],
                "metadata": {
                    "source": f"hf_{c['source']}",
                    "hf_doc_id": c["doc_id"],
                    "heading": c["heading"],
                    "density": c["density"],
                    "source_path": c["path"],
                },
            }
            f.write(json.dumps(row) + "\n")
    print(f"  wrote -> {path}")
    return path


def step_cases(protocols_path: Path, out_dir: Path, n_per: int) -> Path:
    from merken.training.case_generator import (
        CaseGenerator, ProtocolClause, default_gemini_client,
    )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY required for case generation")
    client = default_gemini_client(
        api_key=api_key, model=GEMINI_MODEL, structured=True,
        timeout_s=GEMINI_TIMEOUT_S,
    )
    gen = CaseGenerator(client, structured=True)

    out_path = out_dir / "cases.jsonl"
    tmp_path = out_dir / "cases.jsonl.tmp"
    n_protocols = 0
    n_cases = 0
    n_failed = 0
    t0 = time.time()
    with tmp_path.open("w") as fout:
        for line in protocols_path.open():
            row = json.loads(line)
            n_protocols += 1
            clause = ProtocolClause(
                protocol_id=row["protocol_id"],
                text=row["text"],
                metadata=row["metadata"],
            )
            try:
                cases = gen.generate(clause, n=n_per)
            except Exception as e:
                print(f"    ERR {clause.protocol_id[:50]}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                n_failed += 1
                continue
            for case in cases:
                fout.write(json.dumps({
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "truth": case.truth,
                    "metadata": case.metadata,
                }, ensure_ascii=False) + "\n")
                n_cases += 1
            if n_protocols % 10 == 0:
                print(f"    {n_protocols} protocols / {n_cases} cases / "
                      f"{n_failed} failed / {time.time()-t0:.0f}s")
    tmp_path.rename(out_path)
    print(f"  total: {n_cases} cases from {n_protocols} protocols "
          f"({n_failed} failed) in {time.time()-t0:.0f}s -> {out_path}")
    return out_path


def step_responses(cases_path: Path, out_dir: Path) -> Path:
    from merken.training.case_generator import GeneratedCase
    from merken.training.response_generator import ResponseGenerator

    import requests
    SYSTEM = (
        "You are a community health worker assistant in a low-resource "
        "clinical setting. Respond with ONLY the recommended clinical "
        "action in 1-3 short sentences. Do NOT use markdown headers, "
        "bullet lists, tables, code blocks, or extended explanations. "
        "Do NOT add disclaimers, caveats, or signatures. Be direct, "
        "specific (drug names + doses + durations), and brief."
    )
    session = requests.Session()

    def lmstudio_fn(prompt: str) -> str:
        resp = session.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={
                "model": LMSTUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    gen = ResponseGenerator(lmstudio_fn)

    out_path = out_dir / "responses.jsonl"
    tmp_path = out_dir / "responses.jsonl.tmp"
    n = 0
    n_failed = 0
    t0 = time.time()
    with tmp_path.open("w") as fout:
        for line in cases_path.open():
            row = json.loads(line)
            case = GeneratedCase(
                case_id=row["case_id"],
                prompt=row["prompt"],
                truth=row["truth"],
                metadata=row.get("metadata", {}),
            )
            try:
                cwr = gen.respond(case)
            except Exception as e:
                print(f"    ERR {case.case_id}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                n_failed += 1
                continue
            fout.write(json.dumps({
                "case_id": cwr.case_id,
                "prompt": cwr.prompt,
                "truth": cwr.truth,
                "model_response": cwr.model_response,
                "metadata": cwr.metadata,
            }, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"    {n} responses / {n_failed} failed / "
                      f"{time.time()-t0:.0f}s")
    tmp_path.rename(out_path)
    print(f"  total: {n} responses ({n_failed} failed) in "
          f"{time.time()-t0:.0f}s -> {out_path}")
    return out_path


def step_align(responses_path: Path, out_dir: Path, threshold: float) -> Path:
    from merken.training.midloop_dataset import (
        Aligner, Case, default_embed_fn,
    )

    out_path = out_dir / "aligned.jsonl"
    tmp_path = out_dir / "aligned.jsonl.tmp"
    aligner = Aligner(
        embed_fn=default_embed_fn(model_name=EMBED_MODEL),
        similarity_threshold=threshold,
    )
    n = 0
    n_int = 0
    n_div = 0
    t0 = time.time()
    with tmp_path.open("w") as fout:
        for line in responses_path.open():
            row = json.loads(line)
            case = Case(
                truth=row["truth"],
                model_response=row["model_response"],
                prompt=row["prompt"],
                case_id=row["case_id"],
                metadata=row.get("metadata", {}),
            )
            result = aligner.process(case)
            interventions_out = [
                {
                    "model_token_start": r.model_start,
                    "model_token_end": r.model_end,
                    "truth_token_start": r.truth_start,
                    "truth_token_end": r.truth_end,
                    "truth_text": r.truth_text,
                    "model_text": r.model_text,
                    "cosine_sim": r.cosine_sim,
                }
                for r in result.interventions
            ]
            drops_out = [
                {
                    "truth_text": r.truth_text,
                    "model_text": r.model_text,
                    "cosine_sim": r.cosine_sim,
                }
                for r in result.semantic_drops
            ]
            fout.write(json.dumps({
                "case_id": case.case_id,
                "prompt": case.prompt,
                "truth": case.truth,
                "model_response": case.model_response,
                "metadata": case.metadata,
                "n_divergent": len(result.divergent_regions),
                "n_dropped_semantic": len(result.semantic_drops),
                "n_interventions": len(result.interventions),
                "interventions": interventions_out,
                "semantic_drops": drops_out,
            }, ensure_ascii=False) + "\n")
            n += 1
            n_int += len(result.interventions)
            n_div += len(result.divergent_regions)
            if n % 100 == 0:
                print(f"    {n} aligned / {n_int} interventions / "
                      f"{time.time()-t0:.0f}s")
    tmp_path.rename(out_path)
    print(f"  aligned {n} cases in {time.time()-t0:.0f}s -> {out_path}")
    print(f"  divergent regions: {n_div}, interventions: {n_int}")
    return out_path


def step_format(aligned_path: Path, out_dir: Path) -> Path:
    """Convert aligned.jsonl to the training_data format (boundary labels).

    convert_case returns None for empty responses or zero tokenized
    output. Count these explicitly so silent data loss is visible.
    """
    from experiments.midloop_pilot.format_for_training import convert_case
    out_path = out_dir / "midloop_v1_hf.jsonl"
    n = 0
    n_dropped = 0
    with out_path.open("w") as fout:
        for line in aligned_path.open():
            row = json.loads(line)
            formatted = convert_case(row, labeling="boundary")
            if formatted is None:
                n_dropped += 1
                continue
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            n += 1
    print(f"  formatted {n} cases (dropped {n_dropped} empty/untokenizable) "
          f"-> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=300)
    ap.add_argument("--n-cases", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-per-doc", type=int, default=1,
                    help="max chunks allowed per source doc; higher = more "
                         "coverage, lower = more diversity. v1 used 1.")
    ap.add_argument("--skip", choices=["select", "cases", "responses", "align", "format"],
                    nargs="*", default=[])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== scaleup_phase1_hf ===")
    print(f"out_dir: {OUT_DIR}")
    print(f"n_chunks: {args.n_chunks}, n_cases per chunk: {args.n_cases}")
    print(f"threshold: {args.threshold}")
    print()

    print("step 1/5: select chunks")
    protocols_path = OUT_DIR / "protocols.jsonl"
    if "select" not in args.skip or not protocols_path.exists():
        protocols_path = step_select(
            OUT_DIR, args.n_chunks, args.seed, args.min_chars, args.max_chars,
            args.max_per_doc,
        )

    print("\nstep 2/5: generate cases (Gemini structured)")
    cases_path = OUT_DIR / "cases.jsonl"
    if "cases" not in args.skip or not cases_path.exists():
        cases_path = step_cases(protocols_path, OUT_DIR, args.n_cases)

    print("\nstep 3/5: generate responses (lmstudio gemma)")
    responses_path = OUT_DIR / "responses.jsonl"
    if "responses" not in args.skip or not responses_path.exists():
        responses_path = step_responses(cases_path, OUT_DIR)

    print("\nstep 4/5: align + filter")
    aligned_path = OUT_DIR / "aligned.jsonl"
    if "align" not in args.skip or not aligned_path.exists():
        aligned_path = step_align(responses_path, OUT_DIR, args.threshold)

    print("\nstep 5/5: format for training")
    training_path = OUT_DIR / "midloop_v1_hf.jsonl"
    if "format" not in args.skip or not training_path.exists():
        training_path = step_format(aligned_path, OUT_DIR)

    print(f"\n=== DONE: {training_path} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
