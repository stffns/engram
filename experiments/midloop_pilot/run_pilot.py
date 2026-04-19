"""Midloop dataset pilot -- 3 protocols end-to-end.

Goal: validate the Phase 1 pipeline on REAL clinical content before
committing to the 114-protocol scale-up. Produces a labeled JSONL
that a (future) midloop classifier could train on. Does NOT train
anything.

Stack chosen for the pilot:
  - cases: Gemini 2.5 Flash (Jay's MedLocal teacher_llm; GOOGLE_API_KEY)
  - responses: lmstudio gemma-4-e4b-it-mlx via OpenAI-compatible API
    on http://localhost:1234 (Jay's existing setup, MLX-accelerated)
  - aligner: vstash.embed via fastembed (BAAI/bge-small-en-v1.5)

Outputs to experiments/midloop_pilot/out/:
  protocols.jsonl   -- 3 source clauses ingested from data/core/protocols/
  cases.jsonl       -- Gemini-generated (prompt, truth) pairs
  responses.jsonl   -- lmstudio gemma responses
  aligned.jsonl     -- final labeled dataset
  pilot_report.md   -- cosine distribution, drop ratio, observations

Usage:
  python -m experiments.midloop_pilot.run_pilot
  python -m experiments.midloop_pilot.run_pilot --skip cases   # skip steps
  python -m experiments.midloop_pilot.run_pilot --threshold 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

# Default protocol root: Jay's local MedLocal checkout. Override via
# --protocol-root CLI flag or MIDLOOP_PROTOCOL_ROOT env var so the
# script is portable to other machines / CI without editing source.
DEFAULT_PROTOCOL_ROOT = Path.home() / "Desktop/Personal/Projects/medlocal/data/core/protocols"
PROTOCOL_ROOT = Path(
    os.environ.get("MIDLOOP_PROTOCOL_ROOT", str(DEFAULT_PROTOCOL_ROOT))
)
OUT_DIR = Path(__file__).resolve().parent / "out"

PILOT_PROTOCOLS = ["pneumonia-imci.md", "dengue-who.md", "anemia-who.md"]
N_CASES_PER_PROTOCOL = 5
LMSTUDIO_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "gemma-4-e4b-it-mlx"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_S = 60  # per-call hard timeout (the v1 hang debug)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_protocols(
    files: list[str] | None = None,
    dirs: list[Path] | None = None,
) -> list[dict]:
    """Read protocols from disk, return list of {id, text, metadata}.

    Two modes:
      - ``files`` -- explicit filename list under PROTOCOL_ROOT (legacy
        pilot path; default when neither arg is set).
      - ``dirs`` -- list of directories; every ``*.md`` under each is
        loaded (scale-up path).
    """
    paths: list[Path] = []
    if dirs:
        for d in dirs:
            paths.extend(sorted(d.glob("*.md")))
    elif files:
        for fname in files:
            paths.append(PROTOCOL_ROOT / fname)
    else:
        for fname in PILOT_PROTOCOLS:
            paths.append(PROTOCOL_ROOT / fname)

    out = []
    for path in paths:
        if not path.exists():
            print(f"ERROR: protocol not found: {path}", file=sys.stderr)
            sys.exit(2)
        text = path.read_text(encoding="utf-8")
        out.append({
            "protocol_id": path.stem,
            "text": text,
            "metadata": {
                "source_file": str(path),
                "source": "medlocal_authoritative",
                "source_dir": path.parent.name,
                "ingested_at_pilot": date.today().isoformat(),
            },
        })
    return out


def step_protocols(dirs: list[Path] | None = None) -> Path:
    """Step 0: serialize protocols to JSONL for traceability + reuse."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "protocols.jsonl"
    protocols = load_protocols(dirs=dirs)
    with path.open("w") as f:
        for p in protocols:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  wrote {len(protocols)} protocols -> {path}")
    return path


def gemini_client():
    """Build an LLMClient backed by google.genai (Gemini 2.5 Flash).

    Per-call hard timeout via concurrent.futures: the google-genai SDK
    relies on the underlying gRPC/HTTP layer for timeouts and we observed
    a 23-min hang on a single call during the 84-protocol scale-up
    (SSL_read blocking forever). Wrapping in a futures.Future with
    GEMINI_TIMEOUT_S gives us a hard bound so a single bad call cannot
    block the entire batch.
    """
    try:
        from google import genai
    except ImportError as e:
        raise SystemExit(
            "google-genai not installed. Run "
            "`pip install google-genai` (it is an optional dep for the "
            f"midloop pilot script; not required for core merken). [{e}]"
        ) from e

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) required for case generation"
        )
    client = genai.Client(api_key=key)

    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _do_call(system: str, user: str) -> str:
        full = f"{system}\n\n{user}"
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=full)
        return (resp.text or "").strip()

    def _fn(system: str, user: str) -> str:
        future = pool.submit(_do_call, system, user)
        try:
            return future.result(timeout=GEMINI_TIMEOUT_S)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise TimeoutError(
                f"Gemini call exceeded {GEMINI_TIMEOUT_S}s timeout"
            ) from e

    return _fn


def lmstudio_client():
    """Build a GenerateFn backed by lmstudio's OpenAI-compatible API.

    System prompt constrains the response to match MedLocal's
    production use case: a CHW assistant returning a concise
    clinical action (1-3 sentences, no markdown). Without this,
    the small model produces ChatGPT-style essays (2k+ chars) that
    have no possible style-overlap with the protocol-derived
    truth (~100-200 chars), and EVERY divergence becomes an
    intervention by accident -- the aligner cannot distinguish
    "wrong" from "verbose" without matched response shape.

    Uses a single ``requests.Session`` for connection pooling: at
    570+ calls the connection-reuse savings are measurable (per
    PR #21 review). Validates ``choices`` before indexing so a
    malformed lmstudio response surfaces a clear error rather than
    an opaque KeyError / IndexError.
    """
    try:
        import requests
    except ImportError as e:
        raise SystemExit(
            "requests not installed. Run `pip install requests` "
            "(optional dep for the midloop pilot script; not required "
            f"for core merken). [{e}]"
        ) from e

    SYSTEM = (
        "You are a community health worker assistant in a low-resource "
        "clinical setting. Respond with ONLY the recommended clinical "
        "action in 1-3 short sentences. Do NOT use markdown headers, "
        "bullet lists, tables, code blocks, or extended explanations. "
        "Do NOT add disclaimers, caveats, or signatures. Be direct, "
        "specific (drug names + doses + durations), and brief."
    )

    session = requests.Session()

    def _fn(prompt: str) -> str:
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
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"lmstudio returned no choices: {str(data)[:200]}"
            )
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        return content.strip()

    return _fn


def step_cases(protocols_path: Path) -> Path:
    """Step 1: Gemini -> cases.jsonl."""
    from merken.training.case_generator import CaseGenerator, ProtocolClause

    out_path = OUT_DIR / "cases.jsonl"
    gen = CaseGenerator(gemini_client())

    n_protocols = 0
    n_cases = 0
    t0 = time.time()
    with out_path.open("w") as fout:
        for line in protocols_path.open():
            row = json.loads(line)
            n_protocols += 1
            clause = ProtocolClause(
                protocol_id=row["protocol_id"],
                text=row["text"],
                metadata=row["metadata"],
            )
            try:
                cases = gen.generate(clause, n=N_CASES_PER_PROTOCOL)
            except Exception as e:
                print(f"  ERR {clause.protocol_id}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            for case in cases:
                fout.write(json.dumps({
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "truth": case.truth,
                    "metadata": case.metadata,
                }, ensure_ascii=False) + "\n")
                n_cases += 1
            print(f"  {clause.protocol_id}: {len(cases)} cases")

    print(f"  total: {n_cases} cases from {n_protocols} protocols "
          f"in {time.time() - t0:.1f}s -> {out_path}")
    return out_path


def step_responses(cases_path: Path) -> Path:
    """Step 2: lmstudio gemma-4-e4b -> responses.jsonl."""
    from merken.training.case_generator import GeneratedCase
    from merken.training.response_generator import ResponseGenerator

    out_path = OUT_DIR / "responses.jsonl"
    gen = ResponseGenerator(lmstudio_client())

    n = 0
    t0 = time.time()
    with out_path.open("w") as fout:
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
                print(f"  ERR {case.case_id}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            fout.write(json.dumps({
                "case_id": cwr.case_id,
                "prompt": cwr.prompt,
                "truth": cwr.truth,
                "model_response": cwr.model_response,
                "metadata": cwr.metadata,
            }, ensure_ascii=False) + "\n")
            n += 1
            if n % 5 == 0:
                print(f"  responded {n} cases ({(time.time() - t0):.1f}s)")
    print(f"  total: {n} responses in {time.time() - t0:.1f}s -> {out_path}")
    return out_path


def step_align(responses_path: Path, threshold: float) -> Path:
    """Step 3: token alignment + semantic filter -> aligned.jsonl."""
    from merken.training.midloop_dataset import (
        Aligner, Case, default_embed_fn,
    )

    out_path = OUT_DIR / "aligned.jsonl"
    aligner = Aligner(
        embed_fn=default_embed_fn(model_name=EMBED_MODEL),
        similarity_threshold=threshold,
    )
    n = 0
    n_div = 0
    n_drop = 0
    n_int = 0
    t0 = time.time()
    with out_path.open("w") as fout:
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
            interventions_out = []
            for r in result.interventions:
                interventions_out.append({
                    "model_token_start": r.model_start,
                    "model_token_end": r.model_end,
                    "truth_token_start": r.truth_start,
                    "truth_token_end": r.truth_end,
                    "truth_text": r.truth_text,
                    "model_text": r.model_text,
                    "cosine_sim": r.cosine_sim,
                })
            drops_out = []
            for r in result.semantic_drops:
                drops_out.append({
                    "truth_text": r.truth_text,
                    "model_text": r.model_text,
                    "cosine_sim": r.cosine_sim,
                })
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
            n_div += len(result.divergent_regions)
            n_drop += len(result.semantic_drops)
            n_int += len(result.interventions)
    print(f"  aligned {n} cases in {time.time() - t0:.1f}s -> {out_path}")
    print(f"  divergent regions: {n_div}, drops: {n_drop} ({100*n_drop/max(n_div,1):.1f}%), "
          f"interventions: {n_int} ({100*n_int/max(n_div,1):.1f}%)")
    return out_path


def write_report(aligned_path: Path) -> None:
    """Compute distribution stats + write a markdown report."""
    rows = [json.loads(line) for line in aligned_path.open()]
    n_cases = len(rows)
    n_div_total = sum(r["n_divergent"] for r in rows)
    n_drop_total = sum(r["n_dropped_semantic"] for r in rows)
    n_int_total = sum(r["n_interventions"] for r in rows)

    drop_sims = [
        d["cosine_sim"] for r in rows for d in r["semantic_drops"]
        if d.get("cosine_sim") is not None
    ]
    int_sims = [
        i["cosine_sim"] for r in rows for i in r["interventions"]
        if i.get("cosine_sim") is not None
    ]

    def stats(xs):
        if not xs:
            return {"n": 0, "min": None, "median": None, "max": None, "mean": None}
        return {
            "n": len(xs),
            "min": min(xs),
            "median": statistics.median(xs),
            "max": max(xs),
            "mean": statistics.mean(xs),
        }

    drop_stats = stats(drop_sims)
    int_stats = stats(int_sims)

    # Per-protocol breakdown
    by_protocol: dict[str, dict] = {}
    for r in rows:
        pid = r["metadata"].get("protocol_id", "?")
        b = by_protocol.setdefault(pid, {"cases": 0, "div": 0, "drop": 0, "int": 0})
        b["cases"] += 1
        b["div"] += r["n_divergent"]
        b["drop"] += r["n_dropped_semantic"]
        b["int"] += r["n_interventions"]

    # Sample interventions (the actual labels)
    sample_ints = []
    for r in rows[:5]:
        for iv in r["interventions"][:3]:
            sample_ints.append({
                "case_id": r["case_id"],
                "truth": iv["truth_text"][:80],
                "model": iv["model_text"][:80],
                "cos": iv["cosine_sim"],
            })

    sample_drops = []
    for r in rows[:5]:
        for d in r["semantic_drops"][:2]:
            sample_drops.append({
                "case_id": r["case_id"],
                "truth": d["truth_text"][:80],
                "model": d["model_text"][:80],
                "cos": d["cosine_sim"],
            })

    n_protocols_actual = len({r["metadata"].get("protocol_id", "?") for r in rows})
    out = OUT_DIR / "pilot_report.md"
    lines = []
    lines.append(
        f"# Midloop pilot report -- {n_protocols_actual} protocols "
        f"({date.today().isoformat()})"
    )
    lines.append("")
    lines.append("## Stack")
    lines.append(f"- cases: Gemini 2.5 Flash via google.genai")
    lines.append(f"- responses: lmstudio {LMSTUDIO_MODEL} via OpenAI-compatible API")
    lines.append(f"- aligner: fastembed {EMBED_MODEL}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- protocols: {len(PILOT_PROTOCOLS)} ({', '.join(PILOT_PROTOCOLS)})")
    lines.append(f"- cases: {n_cases}")
    lines.append(f"- divergent regions: {n_div_total}")
    lines.append(f"- semantic drops: {n_drop_total} "
                 f"({100*n_drop_total/max(n_div_total,1):.1f}%)")
    lines.append(f"- intervention labels: {n_int_total} "
                 f"({100*n_int_total/max(n_div_total,1):.1f}%)")
    lines.append("")
    lines.append("## Per-protocol breakdown")
    lines.append("| protocol | cases | div | drop | intv |")
    lines.append("|---|---:|---:|---:|---:|")
    for pid, b in sorted(by_protocol.items()):
        lines.append(f"| {pid} | {b['cases']} | {b['div']} | {b['drop']} | {b['int']} |")
    lines.append("")
    lines.append("## Cosine similarity distribution")
    lines.append(f"### Semantic drops (above-threshold paraphrases)")
    lines.append(f"- n={drop_stats['n']}, "
                 f"min={drop_stats['min']:.3f}, median={drop_stats['median']:.3f}, "
                 f"mean={drop_stats['mean']:.3f}, max={drop_stats['max']:.3f}"
                 if drop_stats['n'] else "- (none)")
    lines.append(f"### Interventions (below-threshold real divergences)")
    lines.append(f"- n={int_stats['n']}, "
                 f"min={int_stats['min']:.3f}, median={int_stats['median']:.3f}, "
                 f"mean={int_stats['mean']:.3f}, max={int_stats['max']:.3f}"
                 if int_stats['n'] else "- (none)")
    lines.append("")
    lines.append("## Sample interventions (first 5 cases)")
    for s in sample_ints[:15]:
        lines.append(f"- [{s['case_id']}] cos={(f"{s['cos']:.3f}" if s['cos'] is not None else "N/A")}  "
                     f"truth=`{s['truth']}` -> model=`{s['model']}`")
    if sample_drops:
        lines.append("")
        lines.append("## Sample semantic drops (first 5 cases)")
        for s in sample_drops[:10]:
            lines.append(f"- [{s['case_id']}] cos={(f"{s['cos']:.3f}" if s['cos'] is not None else "N/A")}  "
                         f"truth=`{s['truth']}` -> model=`{s['model']}`")

    out.write_text("\n".join(lines))
    print(f"  wrote report -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", choices=["protocols", "cases", "responses", "align"],
                    nargs="*", default=[],
                    help="skip steps whose output already exists")
    ap.add_argument(
        "--threshold", type=float, default=0.85,
        help="cosine threshold for semantic-drop filter. Default 0.85 "
             "(empirically calibrated 2026-04-19 on pneumonia/dengue/anemia; "
             "see experiments/midloop_pilot/RESULTS.md for the calibration).",
    )
    ap.add_argument(
        "--protocol-root", type=Path, default=None,
        help="override the protocol root directory (default: "
             "$MIDLOOP_PROTOCOL_ROOT or DEFAULT_PROTOCOL_ROOT). The 3-file "
             "pilot reads .md files relative to this dir.",
    )
    ap.add_argument("--scale-up", action="store_true",
                    help="load all protocols from data/core/protocols/ + who/ "
                         "(84 files) instead of the 3-file pilot list. "
                         "Writes to a separate scaleup_out/ directory so the "
                         "3-file pilot artifacts stay untouched.")
    ap.add_argument("--out-subdir", default=None,
                    help="override OUT_DIR subdir name (default: out for pilot, "
                         "scaleup_out for --scale-up).")
    args = ap.parse_args()

    # Resolve protocol source + output dir.
    global OUT_DIR, PROTOCOL_ROOT
    if args.out_subdir:
        OUT_DIR = Path(__file__).resolve().parent / args.out_subdir
    elif args.scale_up:
        OUT_DIR = Path(__file__).resolve().parent / "scaleup_out"
    if args.protocol_root:
        PROTOCOL_ROOT = args.protocol_root

    protocol_dirs = None
    if args.scale_up:
        # Scale-up reads from siblings of PROTOCOL_ROOT: ../protocols + ../who
        # Default PROTOCOL_ROOT IS .../core/protocols, so parent is .../core
        base = PROTOCOL_ROOT.parent
        protocol_dirs = [base / "protocols", base / "who"]

    print("=== midloop pilot ===")
    print(f"out_dir: {OUT_DIR}")
    print(f"protocols: {PILOT_PROTOCOLS}")
    print(f"n_cases_per_protocol: {N_CASES_PER_PROTOCOL}")
    print(f"threshold: {args.threshold}")
    print()

    print("step 1/4: load protocols")
    protocols_path = OUT_DIR / "protocols.jsonl"
    if "protocols" not in args.skip or not protocols_path.exists():
        protocols_path = step_protocols(dirs=protocol_dirs)

    print("\nstep 2/4: generate cases (Gemini)")
    cases_path = OUT_DIR / "cases.jsonl"
    if "cases" not in args.skip or not cases_path.exists():
        cases_path = step_cases(protocols_path)

    print("\nstep 3/4: generate responses (lmstudio gemma)")
    responses_path = OUT_DIR / "responses.jsonl"
    if "responses" not in args.skip or not responses_path.exists():
        responses_path = step_responses(cases_path)

    print("\nstep 4/4: align + filter")
    aligned_path = OUT_DIR / "aligned.jsonl"
    if "align" not in args.skip or not aligned_path.exists():
        aligned_path = step_align(responses_path, args.threshold)

    print("\nfinal: write report")
    write_report(aligned_path)

    print("\n=== pilot complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
