"""Sample case_generator on a handful of HF-guideline chunks with Gemini.

Selects a diverse set of chunks from
``staging/hf_guidelines/chunks/{who,icrc}/``, runs the structured
Gemini client (PR #25) through ``CaseGenerator``, and reports:

  - how many chunks produce the requested N cases (pass rate),
  - per-chunk (prompt, truth) coherence spot-check (first case),
  - total tokens / wall time (as Gemini SDK exposes them).

Writes full outputs to
``staging/hf_guidelines/sample_cases.jsonl`` so you can eyeball or
promote to the production protocols/ dir afterwards.

Usage:
  GOOGLE_API_KEY=... python -m experiments.midloop_pilot.sample_case_gen
  python -m experiments.midloop_pilot.sample_case_gen --n-chunks 10 --n-cases 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from merken.training.case_generator import (
    CaseGenerator,
    ProtocolClause,
    default_gemini_client,
)

HERE = Path(__file__).parent
CHUNK_ROOT = HERE / "staging" / "hf_guidelines" / "chunks"
OUT_PATH = HERE / "staging" / "hf_guidelines" / "sample_cases.jsonl"
REPORT_PATH = HERE / "staging" / "hf_guidelines" / "sample_report.md"

_FRONT_RE = re.compile(r"\A---\n(.*?)\n---\n\n?", re.DOTALL)


def parse_chunk(path: Path) -> dict:
    """Parse frontmatter .md chunk into {source, doc_id, heading, density, body}."""
    raw = path.read_text()
    m = _FRONT_RE.match(raw)
    if not m:
        raise ValueError(f"no frontmatter: {path}")
    fm = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        fm[k.strip()] = v.strip()
    body = raw[m.end():].lstrip()
    # Strip the duplicate "# heading" line the chunker prepended.
    body = re.sub(r"\A#\s+[^\n]+\n+", "", body)
    return {
        "path": str(path),
        "source": fm.get("source", "unknown"),
        "doc_id": fm.get("doc_id", ""),
        "heading": fm.get("heading", ""),
        "density": float(fm.get("density", 0.0)),
        "chars": int(fm.get("chars", 0)),
        "body": body.strip(),
    }


def select_chunks(
    n: int,
    source_mix: dict[str, int],
    min_chars: int,
    max_chars: int,
    seed: int,
    max_per_doc: int = 1,
) -> list[dict]:
    """Pick chunks density-first with per-doc diversity cap.

    ``max_per_doc=1`` forces one chunk per source document (the v1
    selection mode, trades coverage for diversity). Higher values
    (e.g. 4-8) allow more samples per doc so sections of long
    guidelines get covered -- useful when the corpus is small and
    source docs are already topically diverse internally.
    """
    if max_per_doc < 1:
        raise ValueError(f"max_per_doc must be >= 1, got {max_per_doc}")
    rng = random.Random(seed)
    all_chunks: list[dict] = []
    n_failed = 0
    # sorted() required for cross-filesystem reproducibility; pathlib.glob
    # order is filesystem-defined (APFS happens to be ordered, ext4 is not).
    for sub in ("who", "icrc"):
        sub_dir = CHUNK_ROOT / sub
        if not sub_dir.exists():
            continue
        for p in sorted(sub_dir.glob("*.md")):
            try:
                c = parse_chunk(p)
            except Exception as e:
                import sys as _sys
                print(f"warn: skipping {p.name}: {type(e).__name__}: {e}",
                      file=_sys.stderr)
                n_failed += 1
                continue
            if min_chars <= c["chars"] <= max_chars:
                all_chunks.append(c)
    if n_failed:
        print(f"warn: {n_failed} chunks failed to parse", file=__import__('sys').stderr)

    all_chunks.sort(key=lambda c: -c["density"])
    picked: list[dict] = []
    per_doc: dict[str, int] = {}
    per_src_done = {s: 0 for s in source_mix}
    for c in all_chunks:
        if per_doc.get(c["doc_id"], 0) >= max_per_doc:
            continue
        if per_src_done.get(c["source"], 0) >= source_mix.get(c["source"], 0):
            continue
        picked.append(c)
        per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
        per_src_done[c["source"]] = per_src_done.get(c["source"], 0) + 1
        if sum(per_src_done.values()) >= n:
            break

    # Under-fill warning: if the corpus could not supply the requested mix,
    # surface this instead of silently returning fewer chunks.
    import sys as _sys
    for s, target in source_mix.items():
        got = per_src_done.get(s, 0)
        if got < target:
            print(f"warn: requested {target} {s!r} chunks, got {got} "
                  f"(corpus + max_per_doc={max_per_doc} cap)",
                  file=_sys.stderr)

    rng.shuffle(picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    # Proportional to full corpus (WHO 4041 / ICRC 1474 = ~73% / 27%)
    who_target = round(args.n_chunks * 0.7)
    icrc_target = args.n_chunks - who_target
    source_mix = {"who": who_target, "icrc": icrc_target}

    chunks = select_chunks(
        n=args.n_chunks,
        source_mix=source_mix,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        seed=args.seed,
    )
    print(f"selected {len(chunks)} chunks "
          f"(who={sum(1 for c in chunks if c['source']=='who')}, "
          f"icrc={sum(1 for c in chunks if c['source']=='icrc')})")
    for c in chunks:
        print(f"  [{c['source']}] density={c['density']:5.2f} "
              f"chars={c['chars']:4d}  {c['heading'][:70]}")

    if args.dry_run:
        return

    client = default_gemini_client(
        api_key=api_key,
        model=args.model,
        structured=True,
        timeout_s=args.timeout_s,
    )
    gen = CaseGenerator(client, structured=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_pass = 0
    n_cases_total = 0
    t0 = time.time()
    for i, c in enumerate(chunks):
        # Build protocol_id that will survive into midloop_v0.jsonl metadata
        slug = re.sub(r"[^A-Za-z0-9]+", "-", c["heading"][:40].lower()).strip("-")
        protocol_id = f"{c['source']}-hf-{c['doc_id'][:8]}-{slug}"[:80]
        clause = ProtocolClause(
            protocol_id=protocol_id,
            text=c["body"],
            metadata={
                "source": c["source"],
                "hf_doc_id": c["doc_id"],
                "heading": c["heading"],
                "density": c["density"],
            },
        )
        try:
            t_call = time.time()
            cases = gen.generate(clause, n=args.n_cases)
            dt = time.time() - t_call
            n_pass += 1 if len(cases) >= args.n_cases else 0
            n_cases_total += len(cases)
            print(f"  [{i+1}/{len(chunks)}] {protocol_id[:50]}: "
                  f"{len(cases)}/{args.n_cases} cases in {dt:.1f}s")
            rows.append({
                "protocol_id": protocol_id,
                "source_path": c["path"],
                "heading": c["heading"],
                "n_cases": len(cases),
                "dt_s": round(dt, 2),
                "cases": [
                    {"case_id": cs.case_id, "prompt": cs.prompt,
                     "truth": cs.truth, "metadata": cs.metadata}
                    for cs in cases
                ],
            })
        except Exception as e:
            print(f"  [{i+1}/{len(chunks)}] {protocol_id[:50]}: FAIL {e.__class__.__name__}: {e}")
            rows.append({
                "protocol_id": protocol_id,
                "source_path": c["path"],
                "heading": c["heading"],
                "n_cases": 0,
                "error": f"{e.__class__.__name__}: {e}",
                "cases": [],
            })
    total_dt = time.time() - t0

    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT_PATH}")
    print(f"total time: {total_dt:.1f}s")
    print(f"pass rate (>= {args.n_cases} cases): {n_pass}/{len(chunks)} "
          f"({n_pass/len(chunks)*100:.0f}%)")
    print(f"total cases produced: {n_cases_total} "
          f"(expected {args.n_cases * len(chunks)})")

    # Brief markdown report
    lines = [
        "# HF-guidelines case_generator sample",
        "",
        f"- n_chunks: {len(chunks)}",
        f"- n_cases requested per chunk: {args.n_cases}",
        f"- pass rate: {n_pass}/{len(chunks)} ({n_pass/len(chunks)*100:.0f}%)",
        f"- total cases produced: {n_cases_total}",
        f"- total wall time: {total_dt:.1f}s",
        f"- model: {args.model}",
        "",
        "## Per-chunk results",
        "",
    ]
    for r in rows:
        lines.append(f"### {r['protocol_id']}")
        lines.append("")
        lines.append(f"- heading: `{r['heading'][:200]}`")
        lines.append(f"- cases: {r['n_cases']}")
        if "error" in r:
            lines.append(f"- **ERROR**: {r['error']}")
        elif r["cases"]:
            c0 = r["cases"][0]
            lines.append("- first case:")
            lines.append(f"  - **prompt:** {c0['prompt']}")
            lines.append(f"  - **truth:** {c0['truth']}")
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
