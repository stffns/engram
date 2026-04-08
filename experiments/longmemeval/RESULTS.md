# LongMemEval results

> **Status: no results yet.** This file exists as a pre-commitment to the
> empirical bar. Numbers land here in Phase 2 of the ultra-plan.

## How to read this file

Every row records:

- **Date** — when the run was made.
- **Commit** — the engram commit SHA the run was made against.
- **Baseline** — `vstash` (substrate only) / `engram-raw` (no policies) /
  `engram-classified` (Phase 3 on) / `engram-consolidated` (Phase 5 on) /
  `mempalace` / `mem0` / `zep`.
- **Mode** — the configuration knob that matters for that baseline (e.g.
  `raw` vs `aaak` for mempalace; `episodic-only` vs `episodic+semantic` for
  engram).
- **R@5** — recall @ 5 on LongMemEval, with 95% CI from a bootstrap.
- **API calls per query** — zero unless explicitly noted.
- **Notes** — what was different about this run, what we learned, what we'd
  change next time.

## Results table

| Date | Commit | Baseline | Mode | R@5 (95% CI) | API/q | Notes |
|------|--------|----------|------|--------------|-------|-------|
| —    | —      | —        | —    | —            | —     | _no runs yet — Phase 2 will populate this_ |

## Honesty discipline

If a number we publish here turns out to be wrong, the fix is to add a row
with the corrected number *and* leave the old row in place with a strikethrough
and a link to the correction. We do not silently edit history. See mempalace's
April 2026 correction note for the model.
