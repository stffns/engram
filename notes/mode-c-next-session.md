# Mode C next-session scope (2026-04-22 EOD)

## State at close

Two production candidates tied at 70% on LongMemEval_s N=30
seed=42:

- **H18**: `H1+H3+H12+H18` preface. Baseline, simpler,
  ~42.8s/q.
- **H27v2**: H18 + stop-at-first-turn + narrow-aggregation
  rerank + conditional pool=50. Same top-line, more
  confident verdicts (supports 16->19), ~30s/q.
- **H30** (this session's last, rejected): H27v2 minus stop-
  at-first-turn. 3/7 smoke -- charity flipped supports->neutral
  because without stop-at-turn the Builder kept thinking and
  never locked the correct aggregated answer. Stop-at-turn is
  necessary for aggregation cases.

Gap to RAG-k3 (70-74%) closed on correctness axis.

## The 6 shared fails (both H18 and H27v2 miss)

These require PR-level work, not preface/retrieval tweaks.

| qid | type | issue | next lever |
|---|---|---|---|
| 6d550036 | multi-ses | overcount ("mentioned" vs "led") | Judge post-hoc with explicit role disambiguation |
| 6e984302 | temporal | vocabulary mismatch ("investment"/"bought") | query expansion / synonym retrieval |
| a08a253f | multi-ses | incomplete retrieval recall | corpus-level rechunking |
| 4dfccbf7 | temporal | chunks lack absolute dates | corpus-level timestamp injection |
| gpt4_e061b84g | temporal | thinking-leak (no channel block emits) | stream-level stop at timeout+thinking pattern |
| 6a1eabeb | KU | recency invertida (ambig text) | Judge post-hoc multi-value verification |

## Top priority for next session: Judge post-hoc

Apply Mode A pattern to Mode C output. After generation:

1. Mode C emits answer + records retrieved chunks.
2. Pass (question, answer, retrieved chunks) to Gemini/Cerebras
   Judge.
3. Judge checks:
   - For aggregation: sum all numbers in chunks explicitly
   - For recency: identify most-recent timestamp marker
   - For count: discriminate "mentioned" vs "done by user"
4. Judge either confirms answer or rewrites + cites.

Expected impact: rescues 4-5 of the 6 shared fails. Gets
correctness past 75%, above RAG-k3 ceiling.

Cost: one extra API call per question (~$0.003 with Gemini
Flash). Budget-OK for N=30 eval.

## Second priority: corpus rechunking

For aggregation questions, ingest session-level summaries in
addition to turn-level chunks. Summaries are pre-computed by
LLM: "Session X mentioned: $500 at event A, $2,000 at event B,
..." -- numeric facts aggregated per session.

Retrieval brings summaries + turn chunks. Model has both
granularities.

Expected impact: rescues a08a253f fitness, d851d5ba charity
(already flipping with rerank but more consistently),
gpt4_e05b82a6 rollercoasters (already flipping). Gains might
overlap with Judge post-hoc but is a cheaper intervention.

Cost: pre-generation of session summaries (~$0.001 per
session × ~8 sessions per question × 30 questions = ~$0.25
per N=30). One-time cost per corpus.

## Third priority: thinking-leak fix for gpt4_e061b84g

Builder starts thinking preamble but never emits a
``<channel|>...<turn|>`` block. Budget expires, oracle receives
only "Thinking Process: 1. Analyze the Request..." prefix.

Fix: detect extended thinking without channel open for N
tokens (say 500), force-inject a "The answer is: " prompt
before budget expires. Not elegant but reliable.

## Things NOT to do next session

- Do not re-test Qwen without a better envelope + stop fix.
  Current Qwen experiments plateaued at 30-43% -- the
  Builder family is wrong for this task regardless of
  wrapping.
- Do not tweak preface further. H18's version is calibrated
  and any addition risks distributing regressions across
  non-target categories (observed in H23, H25b, H27 broad
  trigger).
- Do not re-investigate score thresholds. H25b proved
  bypassing lets too much noise in; H14 proved relative is a
  no-op; H25c narrow bypass via rerank is the right balance.
- Do not re-run chunking A turn-pairs without session-level
  summaries. 1/5 smoke confirmed pair-chunking alone
  doesn't close the recall gap.

## Pending adjustments surface (choose 1-2 for next session)

From H28/H30 analysis:

- **stop-at-first-turn calibration**: currently on or off, no
  mid-ground. If H30 proves stop-off is better, ship as
  default. If stop-on helps wall time on single-ses-user,
  consider auto-enable for single-session-user question types
  only.
- **Rerank regex expansion**: narrow aggregation regex misses
  "how many X in the past Y" (like 81507db6 graduations).
  Safer to keep narrow; broader regressed in H27 broad.
  Could add specific phrases like "in the past" as supplement.

## Final winner reproducer (pick H18 or H27v2 or H30)

```bash
# H18 baseline (simpler)
python -m experiments.retrieval.longmemeval.mode_c_benchmark \
  --n 30 --seed 42 \
  --model ~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit \
  --force-first-fire 30 \
  --prompt-preface "<H18 from RESULTS.md>"

# H27v2 enhanced
python -m experiments.retrieval.longmemeval.mode_c_benchmark \
  --n 30 --seed 42 \
  --model ~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit \
  --force-first-fire 30 \
  --stop-at-first-answer-block \
  --rerank-by-number-density \
  --prompt-preface "<H18 from RESULTS.md>"

# H30 (if validated): same as H27v2 minus --stop-at-first-answer-block
```

## Oracle variance disclaimer

All N=30 numbers have ~+-2pp run-to-run variance from Gemini
2.5 Flash single-draw scoring. "Tied at 70%" across runs is
within noise. Multi-draw consensus would tighten this to
<1pp but costs 3-5x the oracle spend.
