# Mode C next-session scope (2026-04-22 EOD, second session)

## State at close

### Preface provenance fixed

The H18/H6b preface strings that produced the historic 21/30 on
seed=42 were never committed to source in their winning session;
they had been passed inline via `--prompt-preface` and lost.
Recovered from a prior-session Claude context on 2026-04-22 and
now committed as constants in `mode_c_demo.py`:

```python
PROMPT_PREFACE_H6B  # commit-to-context, extract mandate
PROMPT_PREFACE_H18  # H6B + aggregation + temporal + recency
```

Aliased in `mode_c_benchmark.py` as `PREFACE_H6B` / `PREFACE_H18`
and routable via `--preface-name {h6b, h18}`. Failed reconstructions
live under `_ALT1`, `_ALT2` for evidence. Anchor check: N=30
seed=42 with recovered H18 reproduced historic 21/30 = 70.0%.

Going forward: **every preface used in a run that feeds RESULTS.md
must be a named constant in source**. Inline strings via
`--prompt-preface` are banned for anchor runs.

### Seed robustness finding (the big one)

Three-seed check of H18 at N=30:

| seed | correct | supports | partial | neutral | contradicts |
|---|---|---|---|---|---|
| 42 (recovered) | 21/30 (70.0%) | 19 | 2 | 3 | 6 |
| 43 | 16/30 (53.3%) | 15 | 1 | 11 | 3 |
| 44 | 15/30 (50.0%) | 15 | 0 | 11 | 4 |

Mean 17.3/30 = **57.8%**, range [50%, 70%], stdev 3.21.

The "Mode C matches RAG-k3 at 70%" headline in prior RESULTS was
a seed=42 sample artifact, not a stable ceiling. Seed=42 had 8
single-session-user questions (where H18 scores ~100%); seed=43
had 3, seed=44 had 5. The question-type mix drives most of the
spread.

### RAG-k3 multi-seed baseline

N=30 across 3 seeds (2026-04-22 EOD):

| seed | RAG-k3 | Mode C H18 | gap (RAG-MC) |
|---|---|---|---|
| 42 | 22/30 (73.3%) | 21/30 (70.0%) | +3.3pp |
| 43 | 15/30 (50.0%) | 16/30 (53.3%) | **-3.3pp (Mode C wins)** |
| 44 | 19/30 (63.3%) | 15/30 (50.0%) | +13.3pp |
| **mean** | **18.7/30 (62.2%)** | **17.3/30 (57.8%)** | **+4.4pp** |

RAG-k3 stdev 3.51 correct, Mode C stdev 3.21 correct. Mean gap
is within 1.5 stdevs -- **not statistically significant at N=30
3-seed**. Mode C is in the same ballpark as RAG-k3, not 15pp
behind as the old single-seed seed=42 comparison implied.

Per-seed asymmetry is informative: on seed=43 Mode C ties RAG;
on seed=44 RAG holds up (63%) while Mode C collapses (50%).
The seed=44 gap is where retrieval-side work should concentrate.
seed=42 historical was a friendly sample for both systems.

### Canonical headline numbers

- **Mode C H18 on LongMemEval_s N=30: 57.8% ± 10pp** (3-seed mean,
  range [50%, 70%], stdev 3.21 correct)
- **RAG-k3 on LongMemEval_s N=30: 62.2% ± 12pp** (3-seed mean,
  range [50%, 73%], stdev 3.51 correct)
- Gap +4.4pp favoring RAG, within noise.

## Top priority for 2026-04-23 (depends on RAG seed=43/44)

The right experiment tomorrow depends on which scenario the
RAG reruns land in:

Result landed: **Scenario C** -- RAG dropped partially (mean
62.2% vs headline 73.3% on seed=42). 4.4pp mean gap within
oracle + sample noise.

### Selected priority: retrieval upgrades first (task #14)

Attack seed=44 specifically since that is where the 13pp gap
lives. The retrieval-miss fails we characterized earlier
(a08a253f missing chunk, gpt4_e061b84g vocabulary mismatch)
are the most likely culprits of Mode C's seed=44 collapse.
Cost: one retrieval code edit (`cerebras_midloop.retrieve()`),
no extra inference pass, no API spend.

**Exact changes in one edit:**
1. Add 4th interleaved pure-vec branch to `retrieval_mode="dual"`:
   `mem.search(query, top_k=top_k, vec_weight=1.0, fts_weight=0.0)`
2. Raise `RETRIEVAL_POOL` 10 -> 50 (or new `--retrieval-pool` flag
   with default 50)
3. Skip chunks whose `source_id.split('::')[1]` starts with
   `sharegpt_`

**Measurement:** N=30 seed=44 first (the anomalous seed). If the
13pp gap closes to <=5pp on seed=44, promote to full 3-seed rerun
to confirm mean lift. If no improvement on seed=44, move on to
Judge post-hoc (task #21).

### Secondary priority: Judge post-hoc (task #21)

Only invoke if retrieval upgrades do not close the seed=44 gap,
OR if we want to push the overall mean above RAG-k3's 62%.
Local gemma Judge (same model as Builder), ported from
`cerebras_midloop.py:judge_once` template but swapped to
`mlx_lm` call. Target: raise mean from 57.8% to 70%+.

## Experiment queue (ordered by expected signal/cost)

1. **#14 Retrieval upgrades combo** -- three-in-one edit in
   `cerebras_midloop.retrieve()`:
   - (a) add 4th interleaved pure-vec branch
     `mem.search(query, top_k=top_k, vec_weight=1.0,
     fts_weight=0.0)` to complement the existing hybrid + 2 FTS
   - (b) raise default `RETRIEVAL_POOL` 10 -> 50 (or flag it)
   - (c) drop chunks whose `source_id.split('::')[1]` starts
     with `sharegpt_` (haystack pollution, not the user's
     conversations)
   Cost: +100ms per retrieval call, zero LLM spend. Target
   qids: a08a253f, gpt4_e061b84g, 6d550036.

2. **#21 Unconditional Judge post-hoc** -- `judge_once(question,
   mode_c_answer, retrieved_chunks)` using the same MLX gemma
   instance already loaded for the Builder. Use the prompt shape
   from `experiments/midloop_concept/medlocal/cerebras_midloop.py::
   judge_once` (lines 393-442) as the template but swap
   `cerebras_chat(JUDGE, ...)` for a local `mlx_lm` call against
   the shared model. Target: raise mean correctness from 58% to
   70-80% by catching the 5 structural fails.

3. **#9 H31 splice-awareness smoke** -- PREFACE_H31 already
   wired in `mode_c_benchmark.py` with SPLICE_AWARENESS_V1 block.
   Smoke on 4 pinned qids against H18 baseline. Target: reduce
   envelope-regurgitation and turn-hijack by making the Builder
   aware that mid-stream tokens are authoritative retrieved
   facts. If smoke shows +1 flip with 0 regressions, promote
   to N=30.

4. **#17 HyDE local-gen** -- before retrieval, ask gemma to
   produce a hypothetical 1-2 sentence answer; embed THAT as
   the retrieval query instead of the raw question. Targets
   the vocabulary-mismatch fails (6e984302, gpt4_e061b84g).

5. **#20 Gated Judge ablation** -- conditional variant of #21:
   only invoke Judge when Mode C output matches a fail pattern
   (refusal text, empty answer block). Run as an ablation vs
   unconditional Judge to quantify the cost/recall tradeoff.

## Invariants not to break

- Never pass prefaces inline via `--prompt-preface` in anchor
  runs. Always use `--preface-name`.
- No corpus manipulation for benchmark gain (session summaries,
  injected timestamps, synthetic facts). Query-time and
  retrieval-config only. See
  `memory/feedback_no_corpus_manipulation_for_benchmark.md`.
- Judge post-hoc uses **local gemma-4-E4B**, not cloud API, to
  preserve the "Mode C = local + free" value prop. Cloud Judge
  would collapse Mode C's differentiator vs RAG.
- 3-seed minimum before claiming any new ceiling/parity number.
  Single-seed comparisons are the artifact this session
  uncovered.
- Code-review any retrieval edit via `code-reviewer` subagent
  before running N=30 against it. See
  `memory/feedback_code_review_before_experiments.md`.

## Reproducer commands

```bash
# H18 seed sweep (N=30)
for seed in 42 43 44; do
  python -m experiments.retrieval.longmemeval.mode_c_benchmark \
    --n 30 --seed $seed \
    --out experiments/retrieval/longmemeval/mode_c_runs_v10 \
    --model ~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit \
    --force-first-fire 30 --preface-name h18 \
    --tag H18_seed${seed}
done

# RAG-k3 seed sweep (cheap, ~10 min each)
for seed in 42 43 44; do
  python -m experiments.retrieval.longmemeval.mode_a_eval \
    --grid experiments/retrieval/longmemeval/grids/rag_k3_seed_robustness.yml \
    --subset longmemeval_s --n 30 --seed $seed \
    --out experiments/retrieval/longmemeval/mode_a_eval_grids
done

# H31 (splice-awareness) smoke -- 4 pinned qids
python -m experiments.retrieval.longmemeval.mode_c_benchmark \
  --qids caf03d32,a1eacc2a,2b8f3739,gpt4_4edbafa2 \
  --out experiments/retrieval/longmemeval/mode_c_smoke \
  --model ~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit \
  --force-first-fire 30 --preface-name h31 --tag H31_smoke
```

## Files to read on resume

- `experiments/retrieval/longmemeval/RESULTS.md` (long, look at
  the 2026-04-22 sections and the "Final session state" at the
  end)
- `experiments/midloop_concept/medlocal/mode_c_demo.py` lines
  149-230 for preface constants, `_apply_chat`
- `experiments/midloop_concept/medlocal/cerebras_midloop.py`
  lines 393-480 for `judge_once` template (must be ported to
  local gemma before use)
- `experiments/retrieval/longmemeval/mode_c_benchmark.py` lines
  65-160 for PREFACES_BY_NAME and flag wiring

## Oracle variance disclaimer

All N=30 numbers carry ~+-2pp variance from Gemini 2.5 Flash
single-draw scoring. Differences within 1 correct (~3pp) are
noise at N=30. Use the 3-seed mean as the stable number; use
per-type breakdowns to diagnose what moved.

Multi-draw consensus would tighten to <1pp but costs 3-5x the
oracle spend. Not worth it at current precision.
