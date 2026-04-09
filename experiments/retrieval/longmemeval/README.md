# longmemeval/

Public retrieval benchmark, ~500 multi-session questions. Measures
R@k: *"out of the top-k retrieved chunks, does at least one come from
a session that actually contains the answer?"*

## What this benchmark can and can't tell us

**Can tell us:** how engram-on-vstash sits in absolute terms against a
standard academic benchmark. Whether the substrate (vstash) is finding
the right chunks on chat-replay data.

**Cannot tell us:** whether engram's decision loop adds value on a
live agent stream. LongMemEval is offline, single-shot, and has no
duplicates in the haystack — which means `should_remember`'s
`dup_exact` rule never fires, `should_consolidate` has nothing to
consolidate, and the whole loop collapses to "ingest everything, then
search." For loop-quality questions, see `../../loop_quality/`.

## Reproduction

```bash
# from the engram repo root, using a sample so it finishes in minutes
python -m experiments.retrieval.longmemeval.runner \
    --subset longmemeval_s \
    --questions 10 \
    --seed 42 \
    --top-k 5 \
    --baseline vstash \
    --baseline engram-heuristic
```

First run downloads `longmemeval_s_cleaned.json` (~277 MB) from
HuggingFace into a gitignored `.cache/` directory.

## Known wall-clock profile (this hardware, CPU only)

- ~35–40 s per question per baseline once the embedder is warm
- 500-question full run: ~6 hours per baseline
- The CONSTITUTION §9 "5 minutes on a laptop" target **does not** apply
  to the full run. Either we accept this as an overnight bench, or we
  find a faster vstash embedder profile. Tracked in the chat history,
  not yet in an issue.

## Dataset

- Source: `xiaowu0162/longmemeval-cleaned` on HuggingFace.
- We use the *cleaned* version on purpose; the original is marked
  deprecated upstream.
- Downloaded into `.cache/` on first run, never committed.

## See also

- `../README.md` — why retrieval/ is only half of the empirical bar
- `../../loop_quality/` — the other half, where engram's loop is on trial
- `../../../CONSTITUTION.md` §9 — the no-benchmark-no-ship rule
