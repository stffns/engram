"""Extract WHO PDF reference books into staged text for clause chunking.

Each PDF in the WHO corpus is a 50-300 page reference book containing
many distinct clinical protocols (IMAI acute care, IMCI booklets,
snakebite guidelines, essential medicines list, etc.). They CANNOT
drop into the existing ``protocols/`` dir as one .md per PDF -- a
single 100-page file is too broad a source for ``case_generator.py``
to derive a specific (prompt, truth) pair from.

This script handles the EXTRACTION step of a two-step pipeline:

  1. (this script) PDF -> raw text or coarsely-split sections, written
     to a STAGING dir (NOT the production protocols/ dir).
  2. (separate, manual) human or LLM-assisted clause chunking from the
     staged text into 50-100 line ``protocols/<topic>-who.md`` files.
     Pair with ``merken.training.case_generator.default_gemini_client(
     structured=True)`` so the chunking pass returns typed clause
     records on the wire.

Modes:

- ``--mode dump`` (default, safest): one .md per PDF containing the
  full extracted text. Useful for human review + downstream LLM
  chunking. Adds a YAML-ish header recording source, page count,
  extraction timestamp.

- ``--mode sections``: splits by detected numbered headings
  (``^\\d+(\\.\\d+)*\\s+[A-Z]``). Quality varies wildly between
  PDFs; ``imci-adaptation-guide.pdf`` produces clean splits,
  ``postnatal-care-recommendations.pdf`` produces zero. Always
  eyeball the output before treating sections as authoritative.

The output dir defaults to ``staging/who_extracted/`` under this
package and is gitignored alongside the rest of the pilot artifacts.

Usage:
  python -m experiments.midloop_pilot.extract_pdfs
  python -m experiments.midloop_pilot.extract_pdfs --mode sections
  python -m experiments.midloop_pilot.extract_pdfs \\
      --pdf-dir /path/to/who --out staging/who_extracted
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Staging dir kept inside the pilot package so it's covered by the
# experiments/* gitignore. Promotion to protocols/ is a separate
# manual step (see module docstring).
DEFAULT_PDF_DIR = Path(
    "/Users/jaysonsteffens/Desktop/Personal/Projects/medlocal/data/core/who"
)
DEFAULT_OUT = Path(__file__).parent / "staging" / "who_extracted"

# Heuristic: a numbered section start. Catches "1 Executive summary",
# "3.2 Classification", "10.5.1 Whatever". Capped at section number
# <=99 (and 99.99.99) so addresses like "1211 Geneva 27 Switzerland"
# do NOT register as headings. Requires a capitalized first word and
# at least one lowercase letter in the title (filters out "1 ALL CAPS
# BANNER" pseudo-headings that PDFs frequently have in margins).
_SECTION_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\s+"
    r"(?P<title>[A-Z][A-Za-z][^\n]*?[a-z][^\n]{0,80})$",
    re.MULTILINE,
)


def _slugify(text: str) -> str:
    """Filesystem-safe slug from a section title."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower())
    return s.strip("-")[:60] or "section"


def extract_full_text(pdf_path: Path) -> tuple[str, int]:
    """Return (full_text, page_count). Pages joined with form feeds
    so a downstream chunker can recover page boundaries if it cares."""
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError(
            "pdfplumber not installed. Run `pip install pdfplumber`. "
            f"[{e}]"
        ) from e
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            pages.append(pg.extract_text() or "")
    return "\f".join(pages), len(pages)


def _header(pdf_path: Path, n_pages: int, mode: str) -> str:
    """Front-matter block recording provenance. Plain text so it
    survives any markdown linter; treat as comments downstream."""
    return (
        "<!--\n"
        f"source_pdf: {pdf_path.name}\n"
        f"source_path: {pdf_path}\n"
        f"page_count: {n_pages}\n"
        f"extraction_mode: {mode}\n"
        f"extracted_at: {datetime.now(timezone.utc).isoformat()}\n"
        "extraction_tool: pdfplumber via experiments.midloop_pilot.extract_pdfs\n"
        "promotion_status: STAGED -- needs clause chunking before use\n"
        "-->\n\n"
    )


def dump_pdf(pdf_path: Path, out_dir: Path) -> Path:
    """Write one .md per PDF containing all extracted text."""
    text, n_pages = extract_full_text(pdf_path)
    out_path = out_dir / f"{pdf_path.stem}.md"
    out_path.write_text(_header(pdf_path, n_pages, "dump") + text)
    return out_path


def split_sections(pdf_path: Path, out_dir: Path) -> list[Path]:
    """Split extracted text by numbered headings into one .md per
    section. Returns the list of written paths.

    A leading ``00-front-matter.md`` captures any text BEFORE the
    first detected heading so nothing is silently dropped (TOC,
    foreword, etc.). If the regex finds zero headings the whole
    text lands in 00-front-matter.md (effectively a dump)."""
    text, n_pages = extract_full_text(pdf_path)
    matches = list(_SECTION_RE.finditer(text))

    section_dir = out_dir / pdf_path.stem
    section_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if not matches:
        # No structure detected; preserve the full text in a single file.
        front = section_dir / "00-front-matter.md"
        front.write_text(_header(pdf_path, n_pages, "sections") + text)
        return [front]

    # Front matter = anything before the first section start.
    if matches[0].start() > 0:
        front = section_dir / "00-front-matter.md"
        front.write_text(
            _header(pdf_path, n_pages, "sections")
            + text[: matches[0].start()]
        )
        written.append(front)

    # Drop trivially-short matches (TOC line items, page-margin
    # addresses, single-line annex titles). A real protocol section
    # has multiple paragraphs; threshold picked empirically against
    # the 7 WHO PDFs.
    MIN_BODY_CHARS = 250

    seen: dict[str, int] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end]
        if len(body.strip()) < MIN_BODY_CHARS:
            continue
        slug = _slugify(m.group("title"))
        # Prefix with section number so files sort meaningfully.
        num_safe = m.group("num").replace(".", "_")
        base = f"{num_safe}__{slug}"
        # Disambiguate: many PDFs reprint the same section title (TOC
        # entry + actual section). Suffix duplicates rather than
        # overwriting silently.
        seen[base] = seen.get(base, 0) + 1
        suffix = "" if seen[base] == 1 else f"__{seen[base]:02d}"
        path = section_dir / f"{base}{suffix}.md"
        path.write_text(_header(pdf_path, n_pages, "sections") + body)
        written.append(path)

    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
                    help="dir of WHO PDFs (default: medlocal/data/core/who)")
    ap.add_argument("--out", "--out-dir", dest="out", type=Path,
                    default=DEFAULT_OUT,
                    help="staging output dir")
    ap.add_argument("--mode", choices=["dump", "sections"], default="dump",
                    help="dump=one md per PDF; sections=split by heading")
    ap.add_argument("--only", default=None,
                    help="optional substring filter on PDF basenames")
    args = ap.parse_args()

    if not args.pdf_dir.exists():
        print(f"ERROR: pdf-dir not found: {args.pdf_dir}", file=sys.stderr)
        return 2

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if args.only in p.name]
    if not pdfs:
        print(f"ERROR: no PDFs matched in {args.pdf_dir}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"=== extract_pdfs (mode={args.mode}) ===", file=sys.stderr)
    print(f"input dir:  {args.pdf_dir}", file=sys.stderr)
    print(f"output dir: {args.out}", file=sys.stderr)
    print(f"PDFs:       {len(pdfs)}", file=sys.stderr)

    total_files = 0
    for pdf in pdfs:
        try:
            if args.mode == "dump":
                out = dump_pdf(pdf, args.out)
                print(f"  {pdf.name} -> {out.name}", file=sys.stderr)
                total_files += 1
            else:
                outs = split_sections(pdf, args.out)
                print(
                    f"  {pdf.name} -> {len(outs)} sections "
                    f"in {pdf.stem}/",
                    file=sys.stderr,
                )
                total_files += len(outs)
        except Exception as e:
            print(f"  ERROR processing {pdf.name}: {e}", file=sys.stderr)
            return 3

    print(f"\nwrote {total_files} files. Next: clause chunking before "
          f"promotion to protocols/. See module docstring.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
