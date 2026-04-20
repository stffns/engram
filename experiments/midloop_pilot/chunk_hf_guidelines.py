"""Chunk epfl-llm/guidelines filtered JSONL into per-protocol .md files.

Each input row is a full guideline document (sometimes 100KB+). We
split it on markdown-style `#` headings and write one .md per chunk
that passes size / content sanity. Output goes to
``staging/hf_guidelines/chunks/<source>/<doc_id>-<chunk_idx>-<slug>.md``
so the protocol-promotion step is a simple `cp` into
``experiments/midloop_pilot/protocols/``.

Filters (conservative; too-small or too-large chunks are dropped):
  - min_chars: drop boilerplate and navigation stubs (default 400).
  - max_chars: drop chunks that look like whole sections rather than
    protocols (default 8000). Larger chunks are re-split on ``##``
    sub-headings if that produces at least 2 sub-chunks passing the
    size filter; otherwise they are skipped.
  - content sanity: chunk must contain at least 3 lowercase words
    after stripping headings, rules out ALL-CAPS banners.

Frontmatter per output .md:

    ---
    source: who|icrc
    doc_id: <sha1 40 chars>
    chunk_idx: <int>
    heading: <first heading of the chunk>
    url: <parent URL or 'None'>
    chars: <len>
    ---

Usage:
  python -m experiments.midloop_pilot.chunk_hf_guidelines
  python -m experiments.midloop_pilot.chunk_hf_guidelines \\
      --in staging/hf_guidelines/who_icrc.jsonl \\
      --out staging/hf_guidelines/chunks
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_IN = HERE / "staging" / "hf_guidelines" / "who_icrc.jsonl"
DEFAULT_OUT = HERE / "staging" / "hf_guidelines" / "chunks"

_HEADING_RE = re.compile(r"^(#+)\s+(.+?)\s*$", re.MULTILINE)

# Clinical-action lexicon. A chunk with high density of these terms is
# likely to contain a protocol that a CHW-assistant pipeline can turn
# into a (prompt, truth) training pair. Calibrated on a 10-doc sample
# (see RESULTS_v0 iteration notes).
_CLIN_VERBS = frozenset([
    "administer", "administered", "give", "prescribe", "treat", "treatment",
    "dose", "dosage", "mg", "ml", "injection", "oral", "intravenous",
    "diagnose", "diagnosis", "assess", "assessment", "refer", "referral",
    "symptoms", "signs", "fever", "pain", "bleeding", "monitor",
    "pregnant", "infant", "child", "adult", "dehydration", "shock",
    "antibiotic", "antimalarial", "vaccine", "immunize",
])
_Q_MARKERS = ("if ", "when ", "in case of", "should ", "must ", "do not")


def clinical_density(body: str) -> float:
    """Clinical-action terms per 1000 chars. 0 for non-clinical prose."""
    lb = body.lower()
    n_verbs = sum(lb.count(v) for v in _CLIN_VERBS)
    n_qm = sum(lb.count(q) for q in _Q_MARKERS)
    return (n_verbs + n_qm) / max(len(body) / 1000.0, 1.0)


def _slugify(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower())
    return s.strip("-")[:limit] or "section"


def _split_on_headings(text: str, level_re: re.Pattern = None) -> list[tuple[str, str]]:
    """Split text on top-level `#` headings. Returns [(heading, body), ...]."""
    if level_re is None:
        level_re = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
    chunks: list[tuple[str, str]] = []
    matches = list(level_re.finditer(text))
    if not matches:
        return []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append((heading, body))
    return chunks


def _has_content(body: str) -> bool:
    # strip other headings, count non-heading lowercase words
    stripped = _HEADING_RE.sub("", body)
    words = re.findall(r"\b[a-z]{3,}\b", stripped)
    return len(words) >= 20


def _resplit_on_subheadings(body: str, min_chars: int, max_chars: int) -> list[tuple[str, str]]:
    subs = _split_on_headings(body, re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE))
    good = [
        (h, b) for (h, b) in subs
        if min_chars <= len(b) <= max_chars and _has_content(b)
    ]
    return good if len(good) >= 2 else []


def process_doc(
    doc: dict, min_chars: int, max_chars: int
) -> list[dict]:
    out: list[dict] = []
    chunks = _split_on_headings(doc["clean_text"])
    for idx, (heading, body) in enumerate(chunks):
        n = len(body)
        if n < min_chars:
            continue
        if not _has_content(body):
            continue
        if n > max_chars:
            sub = _resplit_on_subheadings(body, min_chars, max_chars)
            if not sub:
                continue
            for j, (sh, sb) in enumerate(sub):
                out.append({
                    "source": doc["source"],
                    "doc_id": doc["id"],
                    "url": doc["url"] or "None",
                    "chunk_idx": idx * 1000 + j,
                    "heading": f"{heading} :: {sh}",
                    "body": sb,
                })
        else:
            out.append({
                "source": doc["source"],
                "doc_id": doc["id"],
                "url": doc["url"] or "None",
                "chunk_idx": idx,
                "heading": heading,
                "body": body,
            })
    return out


_SAFE_SOURCES = frozenset({"who", "icrc", "cdc", "nice"})


def write_chunk(out_dir: Path, chunk: dict) -> Path:
    src = chunk["source"]
    # Whitelist: `source` becomes a subdir name; reject anything that could
    # escape out_dir via path traversal (..) or absolute paths.
    if src not in _SAFE_SOURCES:
        raise ValueError(f"unsafe chunk source: {src!r} not in {sorted(_SAFE_SOURCES)}")
    slug = _slugify(chunk["heading"])
    doc_short = chunk["doc_id"][:10]
    filename = f"{doc_short}-{chunk['chunk_idx']:04d}-{slug}.md"
    src_dir = out_dir / src
    src_dir.mkdir(parents=True, exist_ok=True)
    path = src_dir / filename
    body = chunk["body"]
    header = (
        f"---\n"
        f"source: {chunk['source']}\n"
        f"doc_id: {chunk['doc_id']}\n"
        f"chunk_idx: {chunk['chunk_idx']}\n"
        f"heading: {chunk['heading'][:200]}\n"
        f"url: {chunk['url']}\n"
        f"chars: {len(body)}\n"
        f"density: {chunk.get('density', 0.0):.2f}\n"
        f"---\n\n"
        f"# {chunk['heading']}\n\n"
    )
    path.write_text(header + body + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT))
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--limit-docs", type=int, default=0, help="0=all")
    parser.add_argument(
        "--min-density", type=float, default=0.0,
        help="clinical action density cutoff; >=3 picks clinical, "
             ">=5 picks protocol-dense (default 0=keep all)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    if args.limit_docs:
        rows = rows[: args.limit_docs]
    print(f"input docs: {len(rows)}")

    all_chunks: list[dict] = []
    per_source: dict[str, int] = {}
    dropped_density = 0
    for doc in rows:
        cs = process_doc(doc, args.min_chars, args.max_chars)
        for c in cs:
            c["density"] = clinical_density(c["body"])
        if args.min_density > 0:
            before = len(cs)
            cs = [c for c in cs if c["density"] >= args.min_density]
            dropped_density += before - len(cs)
        all_chunks.extend(cs)
        per_source[doc["source"]] = per_source.get(doc["source"], 0) + len(cs)
    if dropped_density:
        print(f"dropped by density < {args.min_density}: {dropped_density}")

    chars = [len(c["body"]) for c in all_chunks]
    print(f"chunks produced: {len(all_chunks)}")
    for s, c in sorted(per_source.items()):
        print(f"  {s}: {c}")
    if chars:
        print(f"chunk chars: min={min(chars)} p50={sorted(chars)[len(chars)//2]} "
              f"p90={sorted(chars)[int(len(chars)*0.9)]} max={max(chars)}")

    if args.dry_run:
        # Print first 10 chunk headings to eyeball
        print("\nFirst 10 chunks (dry-run):")
        for c in all_chunks[:10]:
            print(f"  [{c['source']}] {c['heading'][:80]}  ({len(c['body'])} chars)")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    for c in all_chunks:
        write_chunk(out_path, c)
    print(f"wrote {len(all_chunks)} .md files to {out_path}")


if __name__ == "__main__":
    main()
