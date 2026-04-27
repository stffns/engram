# vstash.ask prompt-lever experiment -- 2026-04-24 EOD

Jay's late-day prompt: "vstash is a glass box with `miss_why` --
maybe there's a config that helps." Sub-experiment that surfaced
more than expected.

## Setup

- `vstash.Memory.ask(query, top_k=10)` endpoint -- retrieval +
  generation in one call via the built-in `chat.py` prompt.
- Cerebras `llama3.1-8b` backend (same as pipeline_runner's winner).
- Same embedder (`bge-small-en-v1.5`), same retrieval (vstash hybrid
  + RRF), same oracle (Gemini 2.5 Flash).
- Same LongMemEval-s seed=44 N=30 substrate used all day.
- Three prompt variants probed.

## Results

### Seed=44 N=30 aggregate

| config                           | sup | par | con | neu | correct   | trust_score |
|----------------------------------|----:|----:|----:|----:|----------:|------------:|
| pipeline_runner + briefs         | 19  | 1   | 6   | 4   | 66.7%     | +46.7%      |
| pipeline_runner no briefs        | 17  | 2   | 7   | 4   | 63.3%     | +40.0%      |
| **vstash.ask DEFAULT**           | 17  | 0   | 2   | 11  | **56.7%** | **+50.0%**  |
| vstash.ask HYBRID (my additions) | 10  | 0   | 3   | 17  | 33.3%     | +23.3%      |

`trust_score = (supports + partial - contradicts) / N * 100`.

vstash.ask's default prompt produced the **highest trust_score of
any configuration tested in the project**. It hedges on 11 qids
(11/30 = 37%) but only mis-answers 2 (2/30 = 7%). The Pareto
trade-off vs `pipeline_runner + briefs` is real: -10pp correct,
+3-10pp trust.

### Per-qid diff: hybrid vs default

| direction        | count | interpretation                       |
|------------------|------:|--------------------------------------|
| supports -> neutral    | 6  | over-hedging introduced               |
| supports -> contradicts| 1  | over-hedging + wrong shift            |
| neutral -> contradicts | 1  | hedge replaced by new lie             |
| contradicts -> neutral | 1  | trivial gain                          |
| gains (any -> supports/partial) | 0 | no unlock from stricter rules |

**Net:** -7 correct, 0 gains. Hybrid unambiguously regressed.

### Stripped prompt on 4-qid subset

Test case: 2 hybrid-regressed qids (`8c18457d`, `ef66a6e5`) + 2
stuck-neutral qids (`a96c20ee`, `gpt4_213fd887`).

Stripped prompt (3 rules only):
> You are a precise document assistant. Answer questions based
> strictly on the provided context.
>
> Rules:
> - Answer only from the context. Do not invent information.
> - If the context doesn't contain the answer, say so clearly.

| qid              | default     | hybrid        | **stripped**  |
|------------------|:------------|:--------------|:--------------|
| 8c18457d         | supports    | neutral       | **supports** (recovered) |
| ef66a6e5         | supports    | contradicts   | contradicts (still lost, model genuinely miscounts 3 vs GT 2) |
| a96c20ee         | neutral     | neutral       | neutral (stuck) |
| gpt4_213fd887    | neutral     | neutral       | neutral (stuck) |

Stripped recovers the one regression that hybrid's prohibitive
clauses caused; matches default on the others. Confirms that the
**default vstash prompt is already at the prompt-lever ceiling**
for this benchmark class -- decorative rules (citation, correction,
code) don't hurt but removing prohibitive rules matters.

## Retrieval is saturated (diagnostic via `miss_analysis`)

`experiments/retrieval/longmemeval/vstash_miss_diagnostic.py` runs
`mem.search(question, top_k=30)` for each of the 13 default-failing
qids and checks the rank at which an answer-session chunk first
appears.

| category             | count | implication                                            |
|----------------------|------:|--------------------------------------------------------|
| answer in top-10     | **13/13** | retrieval already delivers; Builder/prompt is bottleneck |
| answer in rank 11-30 | 0     | raising top_k would not help                            |
| answer missing top-30| 0     | retrieval config already sufficient                     |

The prompt-lever and retrieval-depth experiments independently
confirm: on LongMemEval-s at k=10, retrieval is not the limiting
factor. The remaining failures are cognitive.

## The four stuck reasoning shapes

Qids that neither default nor any prompt variant unlocked. Each
represents a distinct Builder failure mode:

1. **Implicit inference** (`a96c20ee`: "At which university did I
   present a poster?"). Context says "attended conference at
   Harvard, presented poster". Builder refuses to stitch the two
   claims into "poster at Harvard". Same pattern appeared under
   Judge post-hoc and claim_extract earlier.

2. **Relative-date comparison** (`gpt4_213fd887`: "Which event did
   I participate in first, volleyball or charity 5K?"). Context
   has "volleyball started ~2 months ago" and "5K was ~1 month
   ago". Builder hedges on fuzzy temporal expressions rather than
   subtracting.

3. **Identity under pronoun shift** (`gpt4_0a05b494`: "Who did I
   meet first, the woman selling jam or the tourist?"). Context
   refers to the jam seller as "he"; Builder rejects the identity
   match against the question's "woman" premise.

4. **Fuzzy category strictness** (`ef66a6e5`: "How many sports
   have I played competitively?"). Context mentions 3 sports, but
   only 2 are explicitly "competitive"; Builder answers 3, GT is
   2. Lie-class, not hedge-class.

All four require reasoning (stitching, subtraction, identity
matching, category discrimination) that a 8B model cannot reliably
produce from prompt alone on a 0-shot basis.

## Lessons

1. **Default prompts shipped with RAG substrates are a chosen
   point on the trust/correctness curve.** vstash's point is
   trust-first and it is empirically near-optimal for that
   objective.

2. **Prohibitive prompt rules dominate additive ones at 8B scale.**
   Adding "do not substitute a related fact" / "respond EXACTLY:
   'not enough information in memory'" pushed the Builder into
   over-hedging regardless of the extraction rules layered above.

3. **The prompt lever has a ceiling.** Past the default's Pareto
   point, correctness gains require either a larger Builder or
   targeted fine-tuning on the reasoning shapes above; prompt
   tweaking cycles through different trade-off profiles without
   a meaningful net improvement.

4. **vstash's `miss_analysis` is a cheap and honest glass-box
   tool.** Confirming that 13/13 answer-session chunks were in
   top-10 took ~30s and ruled out retrieval as the lever for the
   remaining failures.

## Next steps (ranked by signal-per-effort)

1. **3-seed replication of vstash.ask default** on seeds 42/43 to
   confirm the 56.7%/+50% is not a seed=44 artifact. Cost: ~12 min
   + $0.02 Gemini. Decisive for declaring vstash.ask as a robust
   production baseline.

2. **Training data from the 4 stuck shapes** for Phase 2.3 LoRA.
   Instead of 414 rows of raw teacher imitation, curate ~30 rows
   per stuck-class (implicit inference, relative dates, pronoun
   identity, strict categories). ~$5 teacher cost, ~15 min.

3. **Larger Builder zero-shot on the 11 neutrals.** Ministral 3
   14B (same multimodal gotcha but text sub-config; can be loaded
   via Mistral3ForConditionalGeneration) or Qwen3-7B-Instruct-2507
   local MLX. If 14B zero-shot gets 2-3 of the 11 neutrals to
   supports, more-capable-Builder beats LoRA training.

4. **Document commit + branch merge** for today's work (Phase 2
   scripts + notes + paper addendum + memory entries).

## Artifacts

- `experiments/retrieval/longmemeval/run_vstash_ask.py` -- wrapper
  with `--prompt-variant default|hybrid` flag.
- `experiments/retrieval/longmemeval/vstash_miss_diagnostic.py` --
  per-qid retrieval depth diagnostic using vstash's `search`.
- Output jsonls: `pipeline_runs/vstash_ask_seed44_n30_*.jsonl`
  (default + hybrid variants).
- Diagnostic output: `pipeline_runs/vstash_miss_diagnostic.json`.
- Memory entry: `project_prompt_lever_ceiling.md`.
- Paper addendum: `papers/dual-channel-memory.md` (subsection
  "The prompt-layer ceiling").
