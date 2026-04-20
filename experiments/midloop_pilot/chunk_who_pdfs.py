"""Convert WHO-PDF sections into the same chunk format as HF guidelines.

The extract_pdfs.py `--mode sections` output lives at
``staging/who_extracted_sections/<pdf-stem>/<section-id>.md`` as raw
text with an HTML-comment header. This script normalises those into
the same schema the HF-guidelines chunker produces (YAML-style
frontmatter + `# heading`), writing to
``staging/hf_guidelines/chunks/who_pdf/*.md`` so the existing
``sample_case_gen.select_chunks`` picks them up if its source list is
extended to include ``who_pdf``.

Frontmatter schema (matches chunk_hf_guidelines.write_chunk):
  source: who_pdf
  doc_id: sha1(pdf filename)  # 40 hex chars
  chunk_idx: int  # order within the PDF (used for protocol_id)
  heading: first H1-like line from the section
  url: pdf basename (no URL available for local PDFs)
  chars: body length
  density: clinical-action density score

Default filter: density >= 5.0 (same threshold that produced the
5515 HF chunks in v1). Drops process/policy boilerplate; keeps
dosing / management / assessment sections.

Usage:
  python -m experiments.midloop_pilot.chunk_who_pdfs
  python -m experiments.midloop_pilot.chunk_who_pdfs --min-density 3.0
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chunk_hf_guidelines import clinical_density, _slugify, _SAFE_SOURCES

HERE = Path(__file__).parent
SECTION_ROOT = HERE / "staging" / "who_extracted_sections"
OUT_DIR = HERE / "staging" / "hf_guidelines" / "chunks" / "who_pdf"

_FRONTMATTER_RE = re.compile(r"\A<!--\s*(.*?)\s*-->\s*", re.DOTALL)


def parse_section(path: Path) -> dict:
    raw = path.read_text()
    m = _FRONTMATTER_RE.match(raw)
    body = raw[m.end():].strip() if m else raw.strip()
    # The first non-empty line after stripping is the de-facto heading
    # (extract_pdfs.sections mode prepends the section title).
    lines = [ln for ln in body.splitlines() if ln.strip()]
    heading = lines[0].strip()[:120] if lines else path.stem
    return {"heading": heading, "body": body, "chars": len(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-density", type=float, default=5.0)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "who_pdf" not in _SAFE_SOURCES:
        raise SystemExit(
            "who_pdf must be added to _SAFE_SOURCES in "
            "chunk_hf_guidelines.py before writing these chunks"
        )

    kept = 0
    per_pdf: dict[str, int] = {}
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_dir in sorted(SECTION_ROOT.iterdir()):
        if not pdf_dir.is_dir():
            continue
        pdf_name = pdf_dir.name
        doc_id = hashlib.sha1(pdf_name.encode()).hexdigest()
        # Assign a stable chunk_idx per PDF by sorting section files.
        sections = sorted(pdf_dir.glob("*.md"))
        for idx, sec_path in enumerate(sections):
            info = parse_section(sec_path)
            body = info["body"]
            chars = info["chars"]
            if chars < args.min_chars or chars > args.max_chars:
                continue
            density = clinical_density(body)
            if density < args.min_density:
                continue

            heading = info["heading"]
            out_name = (
                f"{doc_id[:10]}-{idx:04d}-{_slugify(heading)}.md"
            )
            if args.dry_run:
                kept += 1
                per_pdf[pdf_name] = per_pdf.get(pdf_name, 0) + 1
                continue

            header = (
                f"---\n"
                f"source: who_pdf\n"
                f"doc_id: {doc_id}\n"
                f"chunk_idx: {idx}\n"
                f"heading: {heading[:200]}\n"
                f"url: {pdf_name}.pdf\n"
                f"chars: {chars}\n"
                f"density: {density:.2f}\n"
                f"---\n\n"
                f"# {heading}\n\n"
            )
            (OUT_DIR / out_name).write_text(header + body + "\n")
            kept += 1
            per_pdf[pdf_name] = per_pdf.get(pdf_name, 0) + 1

    print(f"total chunks written: {kept}")
    for pdf, n in sorted(per_pdf.items(), key=lambda kv: -kv[1]):
        print(f"  {pdf}: {n}")
    if not args.dry_run:
        print(f"wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
