# WHO PDF extraction -- staging pipeline

## Why this isn't just `pdftotext > protocols/`

The 7 WHO PDFs in `medlocal/data/core/who/` are reference books, not
single-protocol files. Each is 50-300 pages and contains many distinct
clinical protocols. Dropping a 100-page PDF into `protocols/` as one
clause would break the pipeline -- `case_generator.py` cannot derive
a specific (prompt, truth) pair from a source that broad.

So this directory is the EXTRACTION step of a two-step pipeline:

1. **(this script)** PDF -> raw text or coarsely-split sections,
   written to `staging/who_extracted/` (gitignored, regeneratable).
2. **(separate, manual)** clause chunking from the staged text into
   50-100 line `protocols/<topic>-who.md` files. Pair with
   `merken.training.case_generator.default_gemini_client(structured=True)`
   so the chunking pass returns typed clause records on the wire.

## Running the extractor

```
python -m experiments.midloop_pilot.extract_pdfs              # dump mode
python -m experiments.midloop_pilot.extract_pdfs --mode sections
python -m experiments.midloop_pilot.extract_pdfs --only snakebite
```

## What the 7 PDFs actually look like

| PDF | pages | extracted size | structure | suitable mode |
|---|---:|---:|---|---|
| `essential-medicines-24th-2025.pdf` | many | 174 KB | drug formulary tables | dump (chunk per drug entry) |
| `imai-acute-care.pdf` | 121 | 141 KB | flowchart book | dump (chart-aware chunking) |
| `imci-adaptation-guide.pdf` | 78 | 153 KB | numbered sections | sections (cleanest split) |
| `imci-assessment-booklet.pdf` | 6 | 20 KB | fold-out chart | dump (visual layout, mostly TOC after extraction) |
| `maternal-newborn-quality-standards.pdf` | many | 207 KB | numbered standards + measures | sections |
| `postnatal-care-recommendations.pdf` | 12 | 39 KB | recommendation list | dump |
| `snakebite-management.pdf` | 208 | 399 KB | full clinical book | sections (TOC-heavy) |

`imci-assessment-booklet.pdf` is the worst offender -- it's a fold-out
chart, so the extracted text is essentially a flat dump of TOC labels
and column headings without the visual layout that conveys meaning.
That PDF likely needs vision-LLM extraction over rendered pages, not
text extraction. Out of scope here.

## Section-split heuristic

`--mode sections` matches `^\d{1,2}(\.\d{1,2}){0,3}\s+[A-Z][A-Za-z]...[a-z]...$`
on full extracted text. Constraints:

- Section number capped at 99 to filter address lines like
  "1211 Geneva 27 Switzerland".
- Title must contain a lowercase letter to filter ALL-CAPS banners
  PDFs put in margins.
- Sections under 250 chars are dropped (TOC entries, page-margin
  references, single-line annex titles).
- Slug collisions get `__02` / `__03` suffixes so TOC reprints don't
  silently overwrite the real section.

Quality varies wildly between PDFs. `imci-adaptation-guide.pdf`
produces 30+ clean sections; `postnatal-care-recommendations.pdf`
produces zero (its headings aren't numbered). The header in every
output file records `extraction_mode` so downstream consumers can
tell what they're looking at.

## What's NOT in scope

- **Clause chunking.** Step 2 above. Needs an LLM pass
  (Gemini structured output + clinical-knowledge prompt) to split
  staged text into proper `ProtocolClause`-shaped records.
- **Layout-aware extraction** for the chart-style PDFs. Would
  need vision LLM over rendered pages.
- **Promotion to `protocols/`.** Manual review gate: the
  `extracted_at` + `promotion_status: STAGED` markers in every
  staged file are the breadcrumb. A future PR moves promoted
  clauses with proper provenance metadata.

## Stats from a clean run (2026-04-19)

```
=== extract_pdfs (mode=dump) ===
PDFs: 7 -> 7 staged .md files (~1.1 MB total extracted text)
```

The staging dir totals ~1.1 MB of clinical text. Naive token estimate
at ~4 chars/token = ~275K tokens of source material to chunk. Even
at a generous 500 chars per resulting clause, that's ~2200 candidate
clauses -- bounded by the chunking-pass quality, not the raw size.
