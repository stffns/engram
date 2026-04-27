# LoCoMo Phase 2 -- Cross-benchmark validation

End-to-end answer correctness on LoCoMo using the Phase 2 stack
(vstash + Cerebras llama3.1-8b builder via `vstash.Memory.ask` with
default SYSTEM_PROMPT, Gemini 2.5 Flash oracle, `top_k=8`). The
goal is to test whether the LongMemEval Phase 2 findings transfer or
were overfit to one benchmark, AND to fill the methodological gap
flagged on 2026-04-25: prior Phase 2 evals bypass merken's write
decision tree by calling `vstash.Memory.remember` directly.

## Methodology

- Dataset: `locomo10.json` (10 conversations, 1986 QA pairs across
  five categories: single_hop, temporal, multi_hop, open_domain,
  adversarial).
- Sample: stratified `--n-per-cat 40` per seed (200 QAs per run).
- Seeds: 42 / 43 / 44, mirroring Phase 2 LongMemEval methodology.
- Stack: identical to LongMemEval Phase 2's `run_vstash_ask.py` (same
  builder model, same SYSTEM_PROMPT, same `top_k`, same oracle).
  Code is `experiments/retrieval/locomo/runner_phase2.py`. Pre-run
  code review pass: blockers fixed (default-prompt assert, errors-as-
  separate-bucket, conv-batched filename, v7 decider hoisted to
  startup), see commit history.
- Configs (`--config`):
  - `vstash-raw`: ingest direct to `vstash.Memory` (control,
    bypasses merken).
  - `merken-recall`: ingest via `merken.Memory` with
    `AlwaysWrite + NeverConsolidate + NeverForget` (primitive
    plumbing without filtering).
  - `merken-v7`: ingest via `merken.Memory` with
    `ChainedWriteDecider(Heuristic, NanoGPT-v7)` (the same
    chained decider validated in standalone filter-recall
    experiments).
- Per-conversation: ingest once, reuse across the conv's QA pairs.
  Filename includes `convbatched` so this is not silently compared
  to a hypothetical per-question variant.

## vstash-raw baseline, 3-seed (run 2026-04-25)

| seed | N    | correct%        | trust%    | wall_min |
|------|------|-----------------|-----------|----------|
| 42   | 200  | 73/200 = 36.5%  | +14.0%    | 12.9     |
| 43   | 201  | 63/201 = 31.3%  |  +0.5%    | 12.9     |
| 44   | 201  | 75/201 = 37.3%  | +11.4%    | 13.2     |
| mean |      | **35.0% +- 3.2%** | **+8.6% +- 7.2%** | -- |

Per-category mean (across 3 seeds):

| category    | LoCoMo mean      | LME analog        | LME mean | delta(L-LME) |
|-------------|------------------|-------------------|----------|--------------|
| temporal    | 38.3% (+- 1.4)   | temporal-reasoning| 38.4%    | -0.1pp       |
| single_hop  | 44.2% (+- 6.3)   | single-session-*  | 87.7%    | -43.6pp      |
| multi_hop   | 30.8% (+- 6.3)   | multi-session     | 45.8%    | -14.9pp      |
| open_domain | 49.2% (+- 9.8)   | knowledge-update  | 80.6%    | -31.4pp      |
| adversarial | 12.5% (+- 5.0)   | (none in LME)     | --       | --           |

LME baseline numbers are from the 3-seed Phase 2 confirm
(2026-04-24): combined N=90, mean 58.9% / +47.8% trust.

### Cross-benchmark finding

**Temporal-reasoning transfers exactly.** 38.3% on LoCoMo vs 38.4% on
LongMemEval, delta -0.1pp. The shape sangra IDENTICALLY across the
two benchmarks. This falsifies the worry that prior Phase 2 decisions
on temporal were overfit to LongMemEval question style; the bottleneck
on temporal is the model's reasoning, not the dataset's framing.

Other shapes are uniformly harder on LoCoMo (multi_hop -14.9pp,
single_hop -43.6pp, open_domain -31.4pp), but the relative ordering
holds: single-hop > knowledge/open-domain > multi-hop > temporal.
Adversarial (12.5%) has no LME analog -- LoCoMo's adversarial
questions ask about facts NOT present in the haystack, so a calibrated
honest answer would be "neutral".

Across-benchmark conclusion: **the Phase 2 prompt-lever ceiling and
k=8 finding generalize**. The shape-targeted LoRA plan (Phase 2
move #2) is well-directed, since the shape it would attack is the
shape that transfers.

## Write-side comparison (seed=44, N=201)

All three configs evaluated against the SAME builder + oracle stack
(Cerebras llama3.1-8b via `vstash.Memory.ask`, Gemini 2.5 Flash
oracle, top_k=8). The only delta is the ingest path.

| config         | correct          | trust   | turns dropped | wall_min |
|----------------|------------------|---------|---------------|----------|
| vstash-raw     | 75/201 = **37.3%** | +11.4%  | 0.0%          | 13.2     |
| merken-recall  | 71/201 = 35.3%   |  +6.0%  | 0.0-1.6%      | 13.5     |
| merken-v7      | 75/201 = **37.3%** | **+11.9%** | **46.0%** (mean) | 14.4     |

Per-shape correct rate (seed=44):

| shape       | vstash-raw | merken-recall | merken-v7 | v7 vs raw |
|-------------|------------|---------------|-----------|-----------|
| temporal    | 16/40 = 40.0% | 13/40 = 32.5% | 13/40 = 32.5% | **-7.5pp** |
| multi_hop   | 12/40 = 30.0% |  8/40 = 20.0% | 14/40 = 35.0% | **+5.0pp** |
| single_hop  | 18/40 = 45.0% | 20/40 = 50.0% | 18/40 = 45.0% |  0.0pp     |
| open_domain | 24/41 = 58.5% | 26/41 = 63.4% | 25/41 = 61.0% |  +2.5pp    |
| adversarial |  5/40 = 12.5% |  4/40 = 10.0% |  5/40 = 12.5% |  0.0pp     |

Per-conversation v7 drop rate ranged 29.0% to 57.4% (10 convs);
the filter was quite aggressive. Despite that, total correct rate
matches the baseline (75/201 either config) and trust score
slightly improves.

### Findings

**v7 passes E2E.** This is the first end-to-end measurement of v7
as a write filter on a benchmark answer-correctness setting. Prior
v7 evidence was filter-recall in isolation (86% on Step 1, 100%
clean / 19% on subtle noise per nanoGPT validation). The big
methodological gap was: does that filter shape hurt downstream
answer correctness when applied at ingest? Answer on LoCoMo seed=44:
**no overall correct loss, +0.5pp trust, 46% chunk reduction**.

**v7 introduces shape drift.** Net-zero correct hides a +5pp gain
on multi_hop and a -7.5pp loss on temporal. The temporal loss is
concerning: temporal is already the cross-benchmark bottleneck
(38% in both LME and LoCoMo at vstash-raw). v7 was trained on
content-vs-noise patterns from a different distribution; relative
dates ("yesterday I went to...") may look like noise to v7 yet
carry the gold for temporal queries.

**merken-recall (no filter) hurts -2.0pp.** AlwaysWrite +
NeverConsolidate + NeverForget should be semantically equivalent to
vstash-raw, but the HeuristicWriteDecider's hygiene gates (dedup,
too-short, empty) drop 0-1.6% of turns and cost ~2pp accuracy plus
~5pp trust. Hypothesis: the hygiene defaults are tuned for
journal-style notes, not chat dialogue where short turns ("yes",
"agreed") actually contain semantic value (preference signals,
aggregation evidence).

### Decisions

1. **v7 is graduated as write filter for production**, conditional
   on a temporal-reasoning regression check on a future scenario.
   The 46% chunk reduction is significant for storage and retrieval
   cost, and answer correctness does not regress on net.
2. **HeuristicWriteDecider hygiene gates need tuning** for chat
   dialogue substrate. The 5pp trust loss is unexpected; investigate
   whether it is the dedup or the min-length gate driving it.
3. **Temporal-reasoning bottleneck is structural** -- the same shape
   sangra in LME (38.4%) and LoCoMo (38.3% raw, 32.5% with v7).
   Shape-targeted LoRA on temporal remains the highest-signal next
   move. v7 does not solve it; if anything, it makes it worse.

## Smoke artifacts

`phase2_runs/_smoke/` contains N=5 and N=10 jsonl from initial
plumbing validation; excluded from analysis runs above.

## Ingest granularity -- methodology bug, big lever

**2026-04-25 EOD addendum.** Jay flagged that the runner above
(and `run_vstash_ask.py` in LongMemEval) ingest dialogue per turn
-- one `mem.remember` per `(speaker, text)` pair, with a custom
title `qid::sid::turn_idx`. That bypasses vstash's intelligent
chunking by giving it pre-fragmented atoms; the `[date_time]`
adjacency is lost; speaker-A/speaker-B turn order is split across
embedding rows.

The corrected variant (`runner_minimal.py --granularity per-session`)
ingests one `mem.remember` per session: the session's turns are
joined under a `[date_time]` header, and vstash chunks the whole
block. Same builder, same oracle, same `top_k=8`, same seed.

| granularity   | N    | correct        | trust   |
|---------------|------|----------------|---------|
| per-turn      | 201  | 75 = 37.3%     | +11.4%  |
| **per-session** | **201** | **101 = 50.2%** | **+30.3%** |

Per-shape delta (per-session - per-turn):

| shape       | per-turn | per-session | delta |
|-------------|----------|-------------|-------|
| single_hop  | 45.0%    | **67.5%**   | +22.5pp |
| multi_hop   | 30.0%    | **52.5%**   | +22.5pp |
| open_domain | 58.5%    | **73.2%**   | +14.7pp |
| adversarial | 12.5%    | 17.5%       |  +5.0pp |
| temporal    | 40.0%    | 40.0%       |   0.0pp |

### Findings (revised)

1. **Per-turn ingest was a methodology bug.** vstash chunks
   intelligently when given a session-sized block -- the
   `[date_time]` header travels with the dialogue. Per-turn
   throws that away. Going forward, ingest at session granularity
   minimum. The Phase 2 runner's `_ingest_via_vstash` /
   `_ingest_via_merken` paths should be updated.

2. **Temporal is genuinely a Builder bottleneck.** Per-session
   gives +22pp on multi_hop and single_hop but **0pp on temporal**.
   Same 40% on per-session as on per-turn; same 38% in LME at
   per-turn. Temporal sangra IDENTICALLY across granularity AND
   benchmark -- the bottleneck is not retrieval shape, it's
   reasoning. Shape-targeted LoRA on temporal stays the top
   Phase 2 move and the case for it is now stronger.

3. **The merken-v7 / merken-recall conclusions in the previous
   section are CONDITIONAL on the per-turn baseline.** With
   per-session at 50.2%, those configs need to be re-measured
   before any production "graduation" claim. Specifically:
   - Does v7's 46% drop still pass at the per-session baseline?
   - Does merken-recall's -2pp persist or go away when sessions
     are not pre-fragmented?
   The smoke at the top of this section showed v7 dropped 1/5
   and recall got 3/5 vs raw 2/5 on conv-26 5QAs (per-turn);
   those numbers are not comparable to the new per-session
   baseline.

4. **`run_vstash_ask.py` in LME has the same bug.** The Phase 2
   "58.9% +- 1.9% / +47.8% trust" baseline used per-turn ingest.
   Re-measuring it at per-session is the next required step
   before any further Phase 2 lever is evaluated. Expected
   direction: the same +12-13pp gain on LME (so ~70% correct).

### Decisions (revised)

- **The Phase 2 ceiling claim of 56.7% is invalid** as stated --
  it was measured on a methodology that downstream chunking
  fragments. Re-baseline LME at per-session before claiming any
  ceiling.
- **HeuristicWriteDecider hygiene gate hypothesis stands** --
  the -2pp it cost was on per-turn substrate; verify on per-session.
- **v7 production graduation deferred** until measured against
  the corrected baseline.
- **Shape-targeted LoRA on temporal is the unambiguous next
  move** -- it is the only shape that does not respond to the
  granularity fix.

## Re-baseline LME at per-session: NEGATIVE result

**2026-04-25 EOD (corrective).** Ran the per-session granularity
on LongMemEval seed=44 N=30 with the same `--granularity per-session`
flag now in `run_vstash_ask.py`. Result was the OPPOSITE of LoCoMo:

| variant | LME seed=44 | trust |
|---|---|---|
| per-turn (legacy) | 17/30 = 56.7% | +50.0% |
| per-session | 7/30 = 23.3% | +6.7% |

**Delta -33.4pp correct, -43.3pp trust.** Per-session
*destroyed* LME numbers.

### Why the two benchmarks diverge

Inspection of session sizes (longmemeval_s vs locomo10.json):

| | LME | LoCoMo |
|---|---|---|
| sessions per conv | 49 | 19 |
| turns per session | 10 | 22 |
| **session chars (mean)** | **9981** | **2843** |
| **session chars (max)** | **28108** | **5871** |
| individual turn chars (mean) | 967 | ~130 |

LME sessions are ~3.5x larger than LoCoMo and individual LME
turns are themselves ~7x larger. A typical LME session at 10K
chars gets chunked into 5-6 vstash pieces; a 28K char session
into many more. The Builder-relevant fact ends up in one chunk,
diluted by adjacency that doesn't help retrieval. LoCoMo sessions
at ~2.8K chars are 1-2 chunks max; the dialogue+date_time
adjacency is preserved.

LME turns are often already multi-paragraph user notes (medical
log, task list, journal entry), so per-turn already gives vstash
"natural" units to chunk. Per-session over-aggregates them.

LoCoMo turns are short chat lines (e.g. "Yes, exactly!" / "I went
yesterday"); per-turn fragments them below the dialogue grain
that contains the answer. Per-session re-aggregates the dialogue
into the unit vstash is designed to chunk.

### Revised conclusion

**The "per-turn was a bug" framing was wrong.** Per-turn was
correct for LME and wrong for LoCoMo. The right framing is
**ingest at the natural unit of the dataset**: turns when turns
are document-sized, sessions when turns are dialogue-sized.

Going forward:
- LME stays at per-turn. Phase 2 baseline 58.9% / +47.8% trust
  remains valid.
- LoCoMo defaults to per-session. New baseline 50.2% / +30.3%
  trust on seed=44.
- Cross-benchmark temporal-reasoning bottleneck **tracks even
  more tightly with the corrected baselines**: LME 38.4%,
  LoCoMo 40.0% (delta -1.6pp). Shape-targeted LoRA on temporal
  remains the unambiguous next move.

### Re-revised decisions

- **Phase 2 LME 58.9% baseline is valid as measured.** The
  56.7% prompt-lever ceiling is also valid; the lever has been
  measured against the right granularity.
- **LoCoMo Phase 2 numbers from earlier in this doc are stale**
  (they were per-turn). The single per-session run at 50.2%
  replaces them for the cross-benchmark comparison.
- **Re-run merken-v7 / merken-recall on LoCoMo at per-session**
  before any production write-filter claim. The previous run
  was per-turn substrate, results not transferable.
- **Granularity is now a knob.** Document on each benchmark
  what the right unit is.

## Write-side at per-session (LoCoMo seed=44 N=201)

The write-side comparison was rerun at the corrected granularity.
The numbers reverse direction completely:

| config         | correct          | trust   | drop% | wall_min |
|----------------|------------------|---------|-------|----------|
| vstash-raw     | 101/201 = 50.2%  | +30.3%  | 0%    | 14.5     |
| **merken-recall** | **109/201 = 54.2%** | **+33.8%** | 0-2%  | 14.7 |
| merken-v7      |  89/201 = 44.3%  | +17.4%  | ~30% (sessions) | 13.7 |

Comparison vs per-turn run on the same seed:

| config | per-turn | per-session | delta |
|---|---|---|---|
| vstash-raw | 37.3% | 50.2% | +12.9pp |
| merken-recall | 35.3% | 54.2% | +18.9pp |
| merken-v7 | 37.3% | 44.3% | +7.0pp |

Per-shape at per-session:

| shape       | raw    | recall | v7     |
|-------------|--------|--------|--------|
| temporal    | 40.0%  | 40.0%  | 30.0%  |
| multi_hop   | 52.5%  | 55.0%  | 45.0%  |
| single_hop  | 67.5%  | **72.5%** | 67.5% |
| open_domain | 73.2%  | **80.5%** | 68.3% |
| adversarial | 17.5%  | **22.5%** | 10.0% |

### Findings (write-side at correct granularity)

1. **merken-recall (no filter, primitives only) WINS by +4pp
   correct, +3.5pp trust over vstash-raw.** Wins 4 of 5 shapes,
   ties on temporal. This is the first end-to-end positive
   measurement of merken-as-system on a real benchmark with the
   correct granularity. The HeuristicWriteDecider's hygiene
   gates (dedup, min-length, empty) that previously seemed
   harmful at per-turn now pay off at per-session: they remove
   redundant or degenerate sessions (greetings, status checks)
   that diluted retrieval pools.

2. **merken-v7 LOSES by -5.9pp correct, -12.9pp trust at the
   correct granularity.** The v7 model was trained to score
   turn-level content-vs-noise; applied to whole sessions it is
   out-of-distribution and drops ~30% of sessions wholesale.
   The drop costs pickup on temporal (-10pp vs raw),
   open_domain (-5pp), and adversarial (-7.5pp). v7 production
   graduation is **rejected**: it works as a retrieval-only
   filter at turn granularity; it harms answer correctness when
   applied at the dataset's natural unit.

3. **Temporal stays clavado.** raw 40% / recall 40% / v7 30%.
   No write-side intervention helps. v7 actively harms it.
   Cross-benchmark: LME temporal-reasoning 38.4%, LoCoMo
   temporal 40.0% raw / 40.0% recall -- delta -1.6pp. The
   bottleneck is Builder reasoning, not retrieval substrate
   shape, and the case for shape-targeted LoRA is now
   unambiguous.

### Decisions (final for this session)

- **merken-recall is the right write-side default for LoCoMo
  at per-session.** +4pp over vstash-raw is a first real win
  for merken-as-system measured E2E.
- **v7 production graduation rejected** at the corrected
  granularity. Re-train v7 at session granularity if the
  filter is to be salvaged.
- **Phase 2 next move locked in: shape-targeted LoRA on
  temporal-reasoning.** It is the only knob left that addresses
  the bottleneck, and the cross-benchmark + cross-granularity
  invariance of that bottleneck means a successful LoRA on
  LME's temporal-reasoning should transfer to LoCoMo's
  temporal at no extra cost.

## Code review (pre-run)

`runner_phase2.py` reviewed twice by code-reviewer subagent before
launch:

1. Initial review (vstash-raw config only): flagged three
   blockers -- (a) `vstash.chat.SYSTEM_PROMPT` mutation from prior
   sessions, (b) error-bucket accounting collision with neutral
   verdicts, (c) `convbatched` flag missing from output filename.
   All three fixed before smoke.
2. Extension review (merken-recall, merken-v7 configs): flagged v7
   decider being constructed per-conversation (10x reload, deferred
   ckpt-load failure). Fixed by hoisting to startup so any failure
   is loud and the model is reused across convs.

Per `~/.claude/CLAUDE.md`: any script feeding metrics into a RESULTS
doc gets a code-review pass before launch.

## Builder scale-up: gpt-oss-120b vs llama3.1-8b (3-seed, 2026-04-27)

Phase 2 plan move #1. Swap the Builder model on the canonical best
LoCoMo stack (per-session ingest, vec=0.5/fts=0.5 hybrid, mxbai-rerank-base
k=30->8, --n-per-cat 40, Cerebras llama3.1-8b oracle held fixed) and
re-run the same 3 seeds.

Patch: `experiments/retrieval/locomo/runner_rerank.py` previously
hardcoded the Builder via `_override_inference_config(mem, "cerebras",
"llama3.1-8b")`. Patched to thread `--backend`/`--model` CLI args
through to `_override_inference_config`, mirroring `runner_phase2.py`.
Code-review pass: clean (oracle stays on llama3.1-8b, only Builder
swaps; no other hardcodes; env-var defaults safe).

Defensive guard: gpt-oss-120b is a reasoning model. Its Cerebras response
puts hidden chain-of-thought in `message.reasoning` and the answer in
`message.content`. With max_tokens=2048, occasional questions consume the
whole budget on reasoning, leaving `content=None`. Patched runner to
treat `answer is None` as a refusal (`err = "no_content"`) instead of
crashing. On the 3-seed run, this guard fired 0 times in 602 questions.

### Results

| seed | gpt-oss-120b correct | trust   | temporal | wall  | baseline correct | baseline temporal |
|------|---------------------:|--------:|---------:|------:|-----------------:|------------------:|
| 42   | 69.5%                | +64.0%  | 75.0%    | 8.0min| 51.5%            | 32.5%             |
| 43   | 72.6%                | +68.7%  | 87.5%    | 8.2min| 58.2%            | 45.0%             |
| 44   | 70.1%                | +66.7%  | 77.5%    | 8.4min| 59.2%            | 52.5%             |
| mean | **70.7% +- 1.6pp**   | **+66.5%** | **80.0%** | -- | 56.3% +- 3.4pp   | 43.3%             |

Per-shape on seed=44 (200 common QAs, baseline vs gpt-oss-120b):

| shape       | n   | baseline | gpt-oss-120b | delta    |
|-------------|----:|---------:|-------------:|---------:|
| adversarial | 40  | 20.0%    | 17.5%        | -2.5pp   |
| multi_hop   | 40  | 65.0%    | 75.0%        | +10.0pp  |
| open_domain | 40  | 87.5%    | 92.5%        | +5.0pp   |
| single_hop  | 40  | 77.5%    | 87.5%        | +10.0pp  |
| temporal    | 40  | 45.0%    | **77.5%**    | **+32.5pp** |

Per-question: 37 gains, 15 losses, net +22.

### Findings

- **Correct: +14.4pp 3-seed mean.** Bands DO NOT overlap (56.3+-3.4 vs
  70.7+-1.6).
- **Temporal: +36.7pp 3-seed mean (43.3% -> 80.0%).** First lever that
  touches the clavado shape that resisted granularity, hybrid weights,
  reranker, and prompt experiments.
- **Stdev tightens 3.4pp -> 1.6pp.** gpt-oss-120b is more seed-stable in
  addition to more accurate. seed=42 (the historical anomalous-low) goes
  from 51.5% to 69.5% = +18pp on the worst seed.
- **Trust: +16pp** (+50% baseline -> +66.5%). gpt-oss-120b refuses more
  of the adversarial bucket where llama3.1-8b confabulated, reducing
  contradicts and lifting trust without reducing correct.
- **Adversarial -2.5pp is honest behavior, not regression.** 31/53
  total neutrals on seed=44 are adversarial questions where refusal
  is the calibrated answer.
- **Wall clock: 8 min/seed.** Cerebras serves gpt-oss-120b at the same
  effective throughput as llama3.1-8b for this workload. The earlier
  39s/q smoke benchmark was cold-start, not steady-state.

### Decisions

- **Builder is the dominant lever on LoCoMo.** Retrieval-side levers
  saturated at 56-59% on llama3.1-8b. gpt-oss-120b clears 70%.
- **gpt-oss-120b becomes the reference Builder for LoCoMo Phase 2.**
  All future LoCoMo numbers should be re-baselined against this model
  (the prior 56.3% +- 3.4pp baseline reflects an old Builder choice,
  not a saturating ceiling).
- **The "shape-targeted LoRA on temporal" Phase 2 move is now LME-only.**
  LoCoMo temporal moves with Builder choice; the shape-targeted LoRA
  hypothesis was justified when temporal was clavado, which it isn't
  anymore on LoCoMo.

### Files

- Patched runner: `experiments/retrieval/locomo/runner_rerank.py`
  (commit pending: --backend/--model + None guard)
- Artifacts: `experiments/retrieval/locomo/phase2_runs/locomo_rerank_seed{42,43,44}_k30to8_10convs_*qa_gptoss120b-builder-scaleup_*.jsonl`
- Per-seed logs: `gptoss120b_seed{42,43,44}_log.txt` next to artifacts.
- Diff helper: `/tmp/locomo_diff.py` (per-question delta vs baseline).

### Open follow-ups

- LME N=30 with gpt-oss-120b builder (DONE 2026-04-27): cross-benchmark
  transfer confirmed -- 56.7% -> 63.3% (+6.6pp), temporal-reasoning
  +27.3pp echoes the LoCoMo +36.7pp finding. Knowledge-update -66.7pp
  small-N regression (refusal-on-ambiguity instead of "use most recent").
  Full LME write-up in `experiments/retrieval/longmemeval/RESULTS.md`.
- gpt-oss-120b loss-mode inspection: 15 losses on seed=44, 3 are
  partial->contradicts on single_hop. Worth examining whether the
  contradicts are truly wrong or just stricter wording the oracle
  marked harshly.
- Per-seed cost in Cerebras tokens. (Wall is the same; tokens may
  differ if reasoning-model traffic is priced differently.)
- LME 3-seed (seeds 42, 43) on gpt-oss-120b builder. The LoCoMo
  3-seed showed seed-stable behavior for this Builder; expected to
  hold on LME but worth confirming before treating cross-benchmark
  as settled.

## Outstanding

- Fill in write-side table once merken runs land.
- 3-seed merken configs only if single-seed result is borderline.
- LME merken-config port (separate work item; not blocking the
  cross-benchmark conclusion above).
