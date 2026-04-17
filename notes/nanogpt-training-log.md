# nanoGPT Training Log -- merken event classifier

*All experiments 2026-04-16. Model: 4 layers, 4 heads, 128 dim, ~800K params.*

## v1: Baseline (character-level, easy noise only)

### Data
- 1,314 examples (162 signal, 1,152 noise)
- Character-level tokenization, 81 char vocab
- Noise: obvious patterns ("Sprint planning:", "Ticket:", etc.)

### Training curve
```
Step    Train    Val      Gap     Note
0       4.45     4.45     0.00    random (ln(81) = 4.39)
100     1.96     1.95     0.01    structure learned
200     1.33     1.31     0.02    
300     0.71     0.72     0.01    classification begins
400     0.47     0.52     0.05    
500     0.30     0.42     0.12    
600     0.22     0.41     0.20    <-- BEST val loss
700     0.16     0.47     0.31    overfitting confirmed
800     0.13     0.46     0.33    
1000    0.10     0.52     0.42    
1500    0.05     0.64     0.59    
2600    0.04     0.79     0.75    total memorization
```

### Results
| Test | Accuracy |
|------|----------|
| Val noise | 100% (117/117) |
| Val signal (topic prediction) | 0% (0/15) |
| Overall val | 88.6% |
| Novel topics (unseen) | 100% decisions, 90% noise |
| **Borderline noise** | **19% (3/16)** |

### Lessons
- Model learned one heuristic: "tech name + detail = DECISION"
- Works perfectly for obvious noise, fails on subtle noise
- Topic prediction is pure memorization, doesn't generalize
- Overfitting starts at step 600, best checkpoint there

---

## v2: Borderline noise augmentation

### Data
- 1,922 examples (162 signal, 1,760 noise including 636 borderline)
- Same character-level tokenization, 81 char vocab
- Borderline noise: discussions, investigations, reviews, incidents
  that share vocabulary with decisions

### Training curve
```
Step    Train    Val      Gap     Note
0       4.45     4.45     0.00    
100     2.01     2.00     0.01    similar to v1
200     1.49     1.46     0.03    slower than v1 (harder data)
300     0.89     0.88     0.01    still generalizing well
400     0.51     0.54     0.03    gap half of v1 at same step
500     0.35     0.41     0.06    gap half of v1 at same step
600     0.24     0.37     0.12    
700     0.19     0.36     0.17    v1 was DEAD here, v2 still improving
800     0.16     0.34     0.18    <-- BEST val loss
900     0.14     0.35     0.21    overfitting begins
1000    0.13     0.37     0.25    
```

### Results
| Test | v1 | v2 | Change |
|------|----|----|--------|
| Borderline noise | 19% (3/16) | **69% (11/16)** | +50pp |
| True decisions | 100% (3/3) | 67% (2/3) | -33pp (more conservative) |
| Overall Q3 | 32% (6/19) | **68% (13/19)** | +36pp |
| Best val loss | 0.411 | **0.335** | -18% better |
| Optimal step | 600 | 800 | delayed overfitting |

### Lessons
- Borderline examples taught the model verb patterns
- "Discussed" vs "Replaced" now separable (v1 couldn't)
- Trade-off: more conservative, occasional false negatives
- Overfitting delayed by 200 steps (more data to learn from)
- Val loss 18% lower = real generalization improvement

### Remaining failures (v2)
5 borderline noise still misclassified:
- "Team debated replacing Kafka..." -- "replacing" triggers decision
- "WebAuthn registration failing..." -- tech name + detail
- "Enabled gzip compression..." -- genuinely ambiguous
- "Added circuit breaker..." -- genuinely ambiguous  
- "Configured auto-scaling..." -- genuinely ambiguous
Last 3 are legitimately ambiguous -- reasonable people would disagree.

---

## v3: Verb markers (in progress)

### Data
- 608 examples (fewer unique after dedup across scenarios)
- Verb markers prepended: `[VERB:replaced]`, `[VERB:discussed]`, `[CTX:routine]`
- Same 80 char vocab

### Verb marker extraction
```
[VERB:replaced]     -> Replaced Redis with Caffeine...
[VERB:discussed]    -> Discussed switching from Redis...
[VERB:reviewed]     -> Reviewed the Redis migration PR...
[VERB:investigated] -> Ticket SRE-2847: investigated Redis...
[CTX:routine]       -> Sprint planning: infra team...
[CTX:unknown]       -> Redis memory usage spiked to 4GB...
```

### Hypothesis
The attention mechanism can latch onto verb markers in layer 0 instead
of reconstructing verb identity from character sequences. This should:
1. Speed up learning (confirmed: val 0.34 at step 400 vs v2's 0.34 at step 800)
2. Improve borderline classification (to be tested)
3. But may overfit faster with smaller dataset (gap 0.14 at step 400)

### Training curve
```
Step    Train    Val      Gap     Note
0       4.42     4.42     0.00    
100     1.89     1.89     0.00    
200     1.16     1.18     0.02    faster than v1/v2
300     0.44     0.52     0.08    
400     0.20     0.34     0.14    already at v2's BEST val
500     0.11     0.29     0.18    <-- appeared to be BEST val loss
600     0.08     0.29     0.21    val plateaued
```

**BUG FOUND:** v3 trained on 0% signal data due to relative path bug
in prepare.py. All scenarios were "not found" because the script ran
from the nanoGPT directory. The model correctly learned "everything is
NOISE" because in its training data, everything WAS noise.

This produced a misleading result: val loss 0.287 (best ever!) but the
model classified ALL inputs as NOISE with P(D)=0.000. The low val loss
reflected perfect prediction of a trivial task (100% of examples are
the same class).

**Silt lesson #5:** Before interpreting a model's behavior, verify the
data it trained on. A five-second class balance check would have caught
this before three training runs were wasted.

### v3b: verb markers + correct data (fix applied)

Training data rebuilt with absolute paths. 1922 examples (147 signal
in train, 15 in val) -- same volume as v2 but with verb markers added.

```
Step    Train    Val      Gap     Note
0       4.44     4.44     0.00
100     1.96     1.95     0.01
200     1.33     1.35     0.02
300     0.82     0.84     0.02    between v1 and v2 pace
...     (in progress)
```

### Comparison at matched steps
```
Step 300   v1 val    v2 val    v3 val
           0.72      0.88      0.52

Step 400   v1 val    v2 val    v3 val
           0.52      0.54      0.34
```

v3 reaches v2's best val loss in HALF the steps. Whether it also
improves borderline accuracy will be tested after training completes.

---

## Neuron analysis (v2 checkpoint)

### Layer-by-layer differentiation
```
Layer 0: max diff = 0.06  (almost no signal/noise separation)
Layer 1: max diff = 0.15  (beginning to separate)
Layer 2: max diff = 0.38  (clear patterns)
Layer 3: max diff = 1.02  (strong separation)
```

Interpretation: early layers process character patterns, final layers
encode high-level concepts ("this is a decision" vs "this is noise").

### Key neurons (layer 3, 512 neurons)

**Decision-preferring neurons** (fire more for DECISION):
- Neuron 248: D=+1.08, N=+0.34, diff=+0.74
- Neuron 182: D=+0.71, N=+0.07, diff=+0.65
- Neuron 270: D=+1.02, N=+0.50, diff=+0.53

**Noise-preferring neurons** (fire more for NOISE):
- Neuron 387: D=+0.53, N=+1.55, diff=-1.02 (strongest differentiator)
- Neuron 329: D=+0.28, N=+1.28, diff=-1.00
- Neuron 277: D=+0.30, N=+1.21, diff=-0.91

### Neuron 387 activation by example type
```
Decisions (real):     0.32 - 0.37  (low = decision)
Noise (clear):        1.45 - 1.75  (high = noise)
"Switched Kafka..."   0.92         (ambiguous zone -> misclassified)
"Added circuit..."    0.59         (ambiguous zone -> classified as decision)
Borderline noise:     1.77 - 1.89  (correctly high = noise)
```

### Insight
The model has a single strongest discriminator neuron (387) that acts
as a "noise detector" -- high activation = noise, low = decision.
The failure zone is 0.5-1.0 where borderline events land. Verb markers
should sharpen this boundary by giving the attention mechanism a direct
signal to focus on.

---

## Summary table

| Version | Data | Best val | Borderline acc | Key improvement |
|---------|------|----------|----------------|-----------------|
| v1 | 1,314 (easy noise) | 0.411 | 19% | baseline |
| v2 | 1,922 (+borderline) | 0.335 | 69% | +50pp borderline |
| v3 | 608 (BUG: 0 signal) | 0.287 | 0% (all NOISE) | ghost dataset, Silt #5 |
| v3b | 1922 (+verb markers) | **0.330** | **74%** | best overall, recovers true decisions |

---

## Architecture decisions for future models

### What works
- Character-level for signal/noise binary classification (100%)
- Borderline noise augmentation (+50pp on subtle cases)
- Verb markers accelerate learning 2x

### What doesn't work
- Character-level for topic prediction (0% on held-out)
- Small model on small data for nuanced semantic role distinction

### Recommended next steps
1. Verb markers + borderline data combined (v3 has markers but fewer examples)
2. BPE tokenization for topic prediction (word-level tokens)
3. 4-way classifier (world/experience/opinion/entity a la Hindsight)
4. Vitality scoring as continuous output instead of binary classification

---

## v4: BPE tokenization (custom 512-token vocab)

### Data
- Same 1,922 examples as v2/v3b
- Custom BPE tokenizer trained on our data (512 tokens)
- Verb markers included
- 3.2x compression: 61K tokens vs 194K char-level
- Key tokens: "DECISION" (1 token), "NOISE" (1 token), "eplaced" (1 token),
  "Sprint" (1 token), "Redis" (varies)

### Training curve
```
Step    Train    Val      Gap     Note
0       6.25     6.25     0.00    ln(512) = 6.24
100     2.54     2.56     0.02    
200     1.01     1.24     0.23    gap already large
300     0.49     0.94     0.44    
400     0.30     0.92     0.62    <-- BEST val loss
500     0.20     1.06     0.86    val RISING, overfitting
```

Overfits much faster than char-level due to 3.2x fewer training tokens.
Best checkpoint at step 400 (char-level models: step 600-800).

### Results
| Test | v1 | v2 | v3b | BPE |
|------|-----|-----|------|-----|
| Borderline noise | 3/16 | 11/16 | 11/16 | **12/16** |
| True decisions | 3/3 | 2/3 | 3/3 | **3/3** |
| Novel decisions | 10/10 | 10/10 | 10/10 | 3/3* |
| Q3 Overall | 32% | 68% | 74% | **79%** |

*Novel decisions tested on 3 examples only for BPE (all correct).

### New win: "Added circuit breaker"
BPE correctly classifies "Added circuit breaker to the payment service"
as NOISE (P(D)=0.259, P(N)=0.736). All char-level models classified
this as DECISION with P(D) > 0.97. The word-level token "circuit" in
context of "payment service" gives the model enough signal to see this
as an operational change, not an architectural decision.

### Remaining 4 failures (same across all models)
1. "Team debated replacing..." -- "replacing" token triggers DECISION
2. "WebAuthn registration failing..." -- tech name + issue detail
3. "Enabled gzip compression..." -- genuinely ambiguous
4. "Configured auto-scaling..." -- genuinely ambiguous

### Key insight
BPE helps not by being "smarter" but by giving the model access to
word-level patterns directly. "circuit breaker" as tokens carry more
information than c-i-r-c-u-i-t-_-b-r-e-a-k-e-r as characters. The
model doesn't need to reconstruct word identity from characters -- it
starts with words and learns word-level patterns.

Trade-off: BPE overfits 2x faster (best at step 400 vs step 800).
With the same data volume, BPE sees each example more times per step
(shorter sequences = more examples per batch). More data would help.

---

## Complete progression

```
v1:   32%  char-level, easy noise only
v2:   68%  +borderline data augmentation (+36pp)
v3:   BUG  ghost dataset (0 signal), Silt #5
v3b:  74%  +verb markers (+6pp over v2)
BPE:  79%  BPE tokenization (+5pp over v3b)
```

Each iteration improves through a different mechanism:
- v2: more data (quantity)
- v3b: better features (verb markers)
- BPE: better tokenization (word-level access)

The remaining 21% (4 failures) are cases where the text is genuinely
ambiguous -- "Enabled gzip compression" and "Configured auto-scaling"
could reasonably be classified as either noise or decision depending
on organizational context that the text alone doesn't convey.

---

### Production requirements for write filter
- Latency: <5ms per event (current: ~1ms on CPU)
- Recall: >99% (no false negatives on real decisions)
- Precision: >90% (some false positives acceptable)
- Model size: <10MB checkpoint (current: ~10MB, could quantize to ~3MB)

---

## Write filter end-to-end on knowledge_update_50topics (2026-04-17)

First E2E measurement with `NanoGPTWriteDecider` integrated into
`Memory.remember()`. Scenario: 1100 events (150 signal across 50
topics, 950 noise), 50 retrieval queries, Gemini 2.0 Flash as answer
generator + judge. top_k=5.

Inference was pre-computed offline (1.0-1.2 s for 1100 events) and
loaded through `_PrecomputedDecider` at E2E time -- see Mistake #10
for why this split is necessary.

### Filter-quality (offline, before E2E)

| Model   | n_written | recall | FPR  | infer_time |
|---------|-----------|--------|------|------------|
| char v2 | 150 / 1100 | 100%  | 0.0% | 1.22 s     |
| BPE v4  | 150 / 1100 | 100%  | 0.0% | 1.09 s     |

Both models perfectly separate the 150 true decisions from the 950
noise events on this scenario (noise is the "easy" kind -- Sprint
planning, Ticket investigations, etc. -- that the training data
over-represents). Nothing surprising at the unit level; the harder
question is whether that perfect filter translates to a better E2E
answer-accuracy number.

### E2E accuracy (with Gemini Flash judge)

| Config          | Written | Skipped | Accuracy    | Δ vs AlwaysWrite |
|-----------------|---------|---------|-------------|------------------|
| always-write    | 1100    | 0       | 21/50 = 42% | --               |
| nanogpt-char-v2 | 150     | 950     | 27/50 = 54% | **+12 pp**       |
| nanogpt-bpe-v4  | 150     | 950     | 28/50 = 56% | **+14 pp**       |

Store reduction: 86% (wrote 150 of 1100 events). Directionally
consistent with Jay's parallel-session result from 2026-04-16 (+18 pp
on an earlier smaller scenario). CIs at n=50 are wide (~±14 pp) and
overlap, so statistical significance needs a larger query pool --
but the direction is unambiguous and the filter is perfect, so the
gain is not an artifact of filter noise.

### What this proves

1. The BPE v4 → production integration we shipped today (`NanoGPTWriteDecider`
   auto-detecting BPE vs char) works end-to-end.
2. The write filter is not a wash -- dropping 86% of the store cleanly
   improves retrieval quality for the surviving queries.
3. BPE v4 beats char v2 by +2 pp on E2E, consistent with the +5 pp Q3
   delta on borderline noise. Small but in the predicted direction.

### What this does NOT prove

- That nanoGPT should be the *default* `WriteDecider`. That requires
  (a) a real-content scenario (CONSTITUTION hard rule: "Test fixtures
  are not ground truth") and (b) running it on `jay_vstash_*_snapshot`
  and confirming no regression in pass_rate.
- That +14 pp generalizes. This scenario was built for nanoGPT-style
  decision vs noise; scenarios where the noise is less stereotyped
  (meeting summaries, documentation, chat transcripts) will likely
  close the gap.

### Files

- `experiments/consolidation/precompute_nanogpt.py` -- torch-only
  inference, saves `{event_id: Decision}` JSON.
- `experiments/consolidation/test_write_filter.py` -- E2E, reads
  precomputed JSON via `_PrecomputedDecider`.
- `/tmp/decisions_char.json`, `/tmp/decisions_bpe.json` -- artifacts
  of the runs above (not committed).

---

## Real-content validation on jay_vstash_2026_04_09_snapshot (2026-04-17)

Same pipeline on organic content (20 events across 6 topics: MedLocal
clinical/meta, vstash notes, merken design, daily review, kafka
meeting; 4 queries). The scenario has NO "noise" topic -- every event
is legitimate content. The question: does the filter kill real events?

### Filter quality (offline, no noise to false-positive on)

| Model   | n_written | filter recall | dropped |
|---------|-----------|---------------|---------|
| char v2 | 12 / 20   | 60%           | 8 real events |
| BPE v4  | 19 / 20   | 95%           | 1 real event  |

**Char v2 dropped 7 substantial MedLocal/vstash notes** (e.g., "MedLocal
Demo — Petequias/Meningococcemia", "vstash — Upstream Improvement
Ideas", "MedLocal — Ventaja Diferenciadora: Decision Tables"). These
are real decisions, design notes, and incident reports. Character-level
training on "Replaced X with Y" patterns does not generalize to
markdown-structured prose.

**BPE v4 dropped only one event** -- the "merken — Open Questions
Resolved" table. Shared failure mode between both models: bullet-dense
markdown tables trip the classifier. Hypothesis: training data had no
such structures; the model's "DECISION verb + detail" heuristic
doesn't match "| Question | Decision |" table syntax.

### E2E (Gemini Flash, 4 queries)

| Config       | Written | E2E Accuracy |
|--------------|---------|--------------|
| always-write | 20      | 1/4 = 25%    |
| char v2      | 12      | 1/4 = 25%    |
| BPE v4       | 19      | 1/4 = 25%    |

At n=4 the E2E metric cannot discriminate. All three configs miss
the same three queries -- they're retrieval-limited (short queries
against long markdown docs at top_k=5), not filter-limited. The
filter-quality numbers above are the load-bearing signal, not E2E
accuracy at this scale.

### Conclusions

1. **Char v2 is NOT a safe default.** Dropping 40% of real organic
   content is a disqualifier regardless of E2E numbers. The
   CONSTITUTION hard rule "Test fixtures are not ground truth" exists
   for exactly this: the 100/100 on knowledge_update_50topics was
   ground truth for stereotyped-noise detection, not for "does this
   event contain a decision?" in markdown-structured prose.

2. **BPE v4 is closer but not proven safe.** 95% filter recall means
   we'd still lose ~1 in 20 organic events. That's better than char
   v2 by a factor of 8, but a user noticing "my medical case notes
   keep disappearing" would be an immediate trust break.

3. **The +14 pp on knowledge_update_50topics did NOT replicate.** This
   is the key generalization failure: noise that *looks* like the
   training noise is trivially filtered, but the filter does not
   understand "decision-ness", it understands a surface pattern.

4. **Markdown tables are a systematic blind spot.** Both models fail
   on the "Open Questions Resolved" event -- this is independent of
   tokenization. Training-data coverage of structured content is the
   next intervention.

### What's next

- Do NOT flip the default to nanoGPT. Keep `HeuristicWriteDecider`.
- If nanoGPT is to graduate, the training data needs organic content
  samples (real notes, markdown docs, decision logs), not just
  synthetic "Replaced X with Y" + "Sprint planning" pairs.
- A useful intermediate: use nanoGPT BPE v4 as a *flag* (route flagged
  events to a secondary path / ask user), not a silent filter. That
  captures the signal without risking trust-breaking drops.

---

## v5: organic augmentation (2026-04-17)

Follow-up to the real-content failure above. Training data augmented
with 68 organic DECISION samples: 55 from `~/.merken/*.db` (full text
of documents Jay's merken loop kept) + 13 from
`jay_vstash_2026_04_09_snapshot` (4 topics held in for train; 2 topics
held out for val: `medlocal_clinical` + `vstash_notes` = 7 events).

### Training

- Same architecture as v4 (4 layer, 4 head, 128 dim, BPE 512 vocab,
  block 128). Iters bumped 2000 -> 2500 because organic prose adds
  2.4x more training tokens.
- Train loss 0.26, val loss 2.78 at plateau (iter 1400+). Training
  killed at 1800; best ckpt saved at iter 1700.
- Gap train/val is large because the LM loss on markdown prose is
  intrinsically higher than on "Replaced X with Y"; the classification
  head (next-token after `<|label|>`) can still converge cleanly.

### Evaluation

| Scenario                         | v4 recall | v5 recall | FPR (v5) |
|----------------------------------|-----------|-----------|----------|
| organic_val_held_out (7)         | 100%      | 100%      | --       |
| jay_vstash snapshot (20, mixed)  | 95%       | 100%      | --       |
| knowledge_update_50topics (1100) | 100%      | 100%      | 0.0%     |

No regression on synthetic. v5 rescues the one event v4 dropped on
the snapshot ("merken -- Open Questions Resolved", markdown table),
but that event was in v5's training data (`merken_design` topic
is held-in for train). This fix is **memorization**, not generalization.

The val split I chose (`medlocal_clinical` + `vstash_notes`) tests
medical case prose and debug playbooks, which are paragraph-heavy,
not table-heavy. Both v4 and v5 already handled those at 100%. To
prove v5 generalizes to markdown tables it hasn't seen, need a
markdown-tables-specific held-out scenario (see "What's next below").

### Honest summary

- **v5 does not regress on anything measured.**
- **v5 does not prove generalization to markdown tables** -- the only
  case it fixed was also in its training set.
- **v5 is safe to land** (no regression) but **not proven to improve
  the real weakness** (markdown blind spot).

### What's next (really)

- Build or find a markdown-tables scenario that no model in the v1
  -> v5 chain has seen. Run v4 and v5 on it. If v5 > v4, the
  augmentation generalized. If v4 == v5, the organic augmentation was
  pure memorization and a different intervention (features,
  architecture, richer synthetic tables) is needed.

### markdown_tables_held_out (2026-04-17) -- augmentation made it worse

Built a 12-event scenario of markdown tables (6 DECISION -- ADR with
scoring matrix, engine selection, biolab gate cutoffs, rollout plan,
satellite slot allocation, firmware feature freeze; 6 NOISE --
daily ops standup, meeting roster, sprint burndown, ticket triage,
capacity snapshot, oncall handoff). Content uses tech stacks absent
from training (aviation, game dev, bio lab, satellite, firmware).

Filter recall / FPR:

| Model | recall (6 sig) | FPR (6 noise) |
|-------|----------------|---------------|
| v4    | 100%           | 100%          |
| v5    | 100%           | 100%          |

Both models write all 12 events. The filter is completely blind to
markdown-table NOISE. But the per-event confidences tell a sharper
story:

- **v4** had one borderline call: `noise_meeting_roster` at
  P(D)=0.516, P(N)=0.484. With `confidence_threshold=0.6` this event
  would have been correctly skipped -- the model sensed *something*
  off.
- **v5** plows through with >=0.94 P(D) on every event, including all
  6 NOISE tables. The organic augmentation reinforced
  "markdown = DECISION" into a near-deterministic rule and erased
  the only residual noise signal v4 had.

**Conclusion: v5 is worse than v4 on markdown NOISE detection.** The
augmentation didn't generalize the filter; it specialized it harder
toward "long structured markdown = keep". That's net negative if your
real noise includes routine status tables.

**Root cause:** The NOISE training set is 100% single-paragraph flat
text ("Sprint planning: infra team..."). The model has no reference
for what a table-formatted NOISE looks like. Adding 68 more
markdown-DECISION examples without any markdown-NOISE examples tips
the decision boundary further in the wrong direction.

**What this points at:**

- **v6 must add markdown-formatted NOISE samples.** The scenario above
  can seed the NOISE side (6 examples) but more are needed -- routine
  status snapshots, handoff summaries, attendance rosters, etc. Not
  drawn from `~/.merken/*.db` because by definition those were all
  kept.
- Alternative: accept that the single-filter architecture has
  irreducible blind spots, and route high-confidence markdown through
  a secondary check (LLM judge, heuristic pattern).
- Empirically the safest default remains `HeuristicWriteDecider`.

**Do NOT graduate v5.** The synthetic-scenario win does not make up
for the markdown-NOISE regression vs v4.

---

## Shadow mode: bootstrap the training set instead of synthesizing it (2026-04-17)

Three sessions of synthetic-data iteration have landed the filter at a
provable architectural limit: each augmentation fixes one blind spot
and opens another, because the model is surface-pattern-matching and
the data distribution we build for it is always a proxy for Jay's
actual decision-making. The real signal lives in the audit log, but
the current default (`HeuristicWriteDecider`) rarely skips -- so the
audit is 58 "write: True" rows and zero "write: False". There is no
ground-truth label set to train on.

Shadow mode attacks the bootstrapping problem directly. A new
`ShadowWriteDecider` runs two deciders side by side: the primary is
authoritative and controls writes; the shadow only annotates the
audit reason with its prediction and whether the two agreed.
`Memory(write_decider=ShadowWriteDecider(HeuristicWriteDecider(),
NanoGPTWriteDecider(...)))` gives us:

- Zero behavior change for writes (primary wins, always).
- Every event tagged in the audit with shadow-agree or shadow-disagree
  plus the shadow's confidence.
- `merken audit | grep shadow_disagree` surfaces the flagged events.
- User-reviewed disagreements become the labeled training set the
  filter has always needed.

Reason format (appended to the primary's reason):

    |shadow_agree:<shadow.policy>=<write|skip>:<conf>
    |shadow_disagree:<shadow.policy>=<write|skip>:<conf>
    |shadow_error:<ExceptionClass>    (shadow failure; never blocks)

The intended graduation path: run shadow mode in real usage, wait for
~200 disagreements, have Jay review and label them, retrain v6 on
those labels, measure against v4 on the four scenarios we already
have. If v6 > v4 on `markdown_tables_held_out`, the augmentation
generalized; if not, the architecture itself is the ceiling and the
next move is a regression head / larger model / entirely different
approach.

Shadow mode is strictly additive. It does not commit us to anything:
if the disagreements show the shadow is usefully corrective, we
graduate. If they show the shadow is random, we delete the class and
move on with `HeuristicWriteDecider` permanently. Either outcome is a
cheap experiment.

---

## v6: markdown-NOISE synthesis breaks the blind spot (2026-04-17)

v5 regressed on `markdown_tables_held_out` because NOISE training was
100% single-paragraph flat text; the model had no reference for
table-formatted NOISE. v6 addresses this by adding 30 Gemini-
generated markdown-table NOISE samples (schedule, status_snapshot,
toc, pricing, log, roster categories in domains absent from the
held-out scenario).

Training: same architecture as v4/v5, 2500 iters, plateau at ~iter
2300, best ckpt saved automatically.

### Eval across 4 scenarios

| Scenario                          | v4      | v5      | v6           |
|-----------------------------------|---------|---------|--------------|
| markdown_tables_held_out recall   | 100%    | 100%    | 100%         |
| markdown_tables_held_out FPR      | 100%    | 100%    | **66.7%**    |
| organic_val_held_out recall       | 100%    | 100%    | 100%         |
| jay_vstash_snapshot recall        | 95%     | 100%    | 100%         |
| knowledge_update_50topics recall  | 100%    | 100%    | **99.3%**    |
| knowledge_update_50topics FPR     | 0%      | 0%      | 0%           |

### What v6 caught (vs v5)

Two of 6 NOISE tables correctly skipped:
- `noise_daily_ops_standup` (P(D)=0.154, P(N)=0.846) -- clear noise
- `noise_meeting_roster` (P(D)=0.363, P(N)=0.637) -- v4 was borderline
  here (0.516/0.484), v6 firms it up to the correct side.

### What v6 missed

Four NOISE tables still written:
- `noise_sprint_burndown_snapshot`, `noise_ticket_triage_summary`,
  `noise_capacity_snapshot`, `noise_oncall_handoff_log`.

All share operational vocabulary with real decisions ("incidents",
"capacity", "triage", "handoff"). Token-level cues are insufficient
to distinguish them from substantive incident post-mortems. A
single-round training-data top-up is not enough for this tier of
NOISE; more samples with operational language but routine intent
would help, OR the architecture itself needs help (richer features,
bigger model, hybrid with a gate).

### What v6 broke

One synthetic knowledge_update event misclassified: "Manual Nessus
scans quarterly. Results emailed to security team." P(D)=0.465,
P(N)=0.535. Honestly borderline content (procedural status dressed
as a decision). Not a systematic regression -- 149/150 still written
correctly.

### Summary

**v6 is net better than v4/v5 on the measured scenarios.** +33 pp
on markdown blind spot, -0.67 pp on the synthetic baseline, zero
regression on organic recall. First retraining round where the
augmentation clearly helped without a matching regression.

**v6 is still not usable as a silent filter.** 66.7% FPR on
table-formatted NOISE means 4 of every 6 routine status tables
would be kept. That's better than 100% but not production-safe.

**Honest next step:** ship v6 as an upgraded shadow backend (via the
existing `MERKEN_SHADOW_NANOGPT_CKPT`), keep collecting oracular
labels for real Jay content, retrain on that real distribution
before the next comparison. Synthetic augmentation has diminishing
returns at this point.

---

## Graduation criteria: when to flip MERKEN_PRIMARY (2026-04-17)

`MERKEN_PRIMARY=nanogpt` chains ``ChainedWriteDecider(Heuristic,
nanoGPT)`` so the classifier decides writes for anything that
passes the hygiene gates. Flipping this is the moment nanoGPT stops
being a shadow observer and starts affecting what Jay can find in
recall. The decision deserves numeric criteria, not vibes.

A candidate `vN` graduates only if **all five** hold:

1. **Filter recall on `organic_val_held_out` >= 100%.** The 7
   held-out medical / vstash notes are non-negotiable; dropping any
   of them is a trust-break on real content. (Measured offline via
   `precompute_nanogpt.py`.)

2. **Filter recall on `jay_vstash_2026_04_09_snapshot` >= 95%.**
   Allows v4-level "drop 1 of 20" but forbids worse. Includes
   training-set overlap so this is a memorize-or-match bar, not a
   generalization bar.

3. **No regression vs the previous graduated version on the four
   scenario filter numbers.** Each of recall / FPR on
   `knowledge_update_50topics`, `jay_vstash_snapshot`,
   `organic_val_held_out`, `markdown_tables_held_out` must be >=
   the previous baseline.

4. **Agreement with the oracle on >= 95% of N_labels>=200 labeled
   disagreements.** Oracle = Gemini 2.0 Flash via
   `merken audit --label-with gemini`. This is the only real-
   distribution bar -- synthetic recall numbers by themselves are
   not enough (v5's "100% rescues markdown table" was memorization).

5. **`markdown_tables_held_out` FPR <= 50%.** Explicit ceiling on
   the blind spot. v6 is at 66.7%; graduating at that rate writes 4
   of every 6 routine status tables, unacceptable.

A graduation run MUST re-measure all four scenarios AND run the
oracular comparison before flipping the env var. Suggested flow:

    MERKEN_SHADOW=nanogpt ...    # accumulate labels
    merken audit --label-with gemini --limit 50    # ... over days/weeks
    # When ~200 labels exist:
    PYTHONPATH=. python -m experiments.consolidation.precompute_nanogpt \
      --model bpe_v<N> --scenario <each of 4> ...
    # Cross-check oracle agreement via a small script over merken_labels.
    # Only if ALL criteria pass: update hook env from MERKEN_SHADOW to MERKEN_PRIMARY.

**Reversing** is cheap: unset `MERKEN_PRIMARY`, set
`MERKEN_SHADOW` again. All historical audit rows remain valid. Do
not conflate "the model was once graduated" with "the model will
always be graduated" -- the bar above is a live check, not a
lifetime certificate.

---

## Gemma 3 1B-IT: LLM backend becomes real (2026-04-17)

Second pass at the LLM shadow path, after 270M proved too weak.

### Model: google/gemma-3-1b-it (2 GB)

Three prompt iterations on the same 12 events of
`markdown_tables_held_out` + organic_val + jay_vstash scenarios:

| Prompt        | markdown recall | markdown FPR |
|---------------|-----------------|--------------|
| Zero-shot plain "Event: X Label:" | (see 270M) | (12-27% raw_mass, useless) |
| Chat template zero-shot            | 50%            | 0% (very conservative) |
| Chat template + few-shot multi-turn| 100%           | 66.7% (tied with v6) |
| **Chat template + few-shot single-turn** | **100%** | **16.7%** (5/6 skipped) |

Single-turn wins because small LMs treat multi-turn few-shot as "past
conversations" less relevant to the current query. Packing
instructions + examples + new event into one user message primes
more effectively. Fix landed in ``merken/classifiers/llm.py``:
DEFAULT_INSTRUCTIONS + DEFAULT_FEWSHOT, single-turn chat template
prompt.

### Final eval (single-turn chat+fewshot)

| Scenario                   | recall | FPR   | latency/event |
|----------------------------|--------|-------|---------------|
| markdown_tables_held_out   | 100%   | 16.7% | 3.75 s        |
| organic_val_held_out       | 100%   | --    | 5.82 s        |
| jay_vstash_snapshot        | 100%   | --    | 6.22 s        |

### vs nanoGPT v6

On the markdown blind spot specifically:
- v6: 2/6 NOISE caught (66.7% FPR)
- Gemma 1B: 5/6 NOISE caught (16.7% FPR)
- Recall parity (100% on real DECISIONs) in both.

The one NOISE table Gemma 1B still writes is the sprint_burndown
example, which is materially similar to an example in the few-shot
prompt -- an in-distribution miss.

### Cost

- Latency: 3-6 s / event on CPU. 100-1000x slower than nanoGPT.
- Memory: ~2 GB resident while the model is held.
- Disk: 2 GB one-time.

### Role

Usable as a **shadow backend**
(``MERKEN_SHADOW=llm``/``MERKEN_SHADOW_LLM_MODEL=google/gemma-3-1b-it``).
Not usable as primary (``MERKEN_PRIMARY``) because the 3-6 s
latency kills the write path. Complementary profile to nanoGPT:
nanoGPT fast-and-shallow, Gemma slow-and-sharp. An ensemble with
nanoGPT on every event and Gemma only on disagreements is the next
natural experiment.

### Things NOT tried yet

- Gemma 3 4B-IT (8.6 GB disk, likely 15-30s/event on CPU; might be
  too slow for any live role).
- SmolLM / Qwen / Phi mini-class models.
- Ensembling nanoGPT v6 + Gemma 1B at decision time.

---

## Gemma 3 270M-IT sanity (2026-04-17) -- backend runs, signal doesn't

Ran `LLMWriteDecider(model_name='google/gemma-3-270m-it',
device='cpu')` over four canonical events.

Load: 4.67 s one-time. Inference: 200-310 ms / event on CPU
(acceptable for the write path if the model actually worked).

Per-event:

| Event (summary)                  | P(D) | P(N) | raw_mass | latency |
|----------------------------------|------|------|----------|---------|
| Real decision ("Replaced ...")   | 0.989| 0.011| 0.124    | 207 ms  |
| Sprint-planning noise            | 0.965| 0.035| 0.217    | 188 ms  |
| Markdown ADR (decision table)    | 0.919| 0.081| 0.274    | 310 ms  |
| Markdown ops standup (noise tbl) | 0.972| 0.028| 0.269    | 251 ms  |

**Observation: `raw_mass` (= P(DECISION_tok) + P(NOISE_tok)) is only
12-27%.** Gemma's continuation after "Label:" puts most of its
probability on *other* tokens (whitespace, formatting, even random
words). The two class tokens are NOT the natural continuation the
model wants to produce, so the post-normalized P(D) ends up near
1.0 for almost everything.

Consequence: Gemma 3 270M zero-shot via two-token logit scoring
classifies every event as DECISION. It cannot be a drop-in shadow
or primary backend today.

**Why we're not giving up on the LLM branch:**
- Larger Gemma (2B, 7B) is likely to discriminate -- small-model
  instruction following is the bottleneck, not the idea.
- Few-shot prompting or chat-template wrapping would likely lift
  the class tokens to the top of the distribution.
- Full-string likelihood comparison (score entire " DECISION" vs
  " NOISE" strings) sidesteps the "what's the natural first
  token" problem.

**What we keep:**
- `LLMWriteDecider` + env activation stays, unchanged. The abstraction
  is fine; only this specific small-model + simple-scoring combo
  failed.
- `MERKEN_SHADOW=llm` is still the right contract: point it at a
  stronger backend when one is available.
- nanoGPT BPE v6 stays as the default shadow classifier until a
  better pipeline lands.

---

## Mistakes, dead ends, and lessons (the important part)

### Mistake 1: Celebrating 100/100 before testing properly

The v1 model showed 100% precision and 100% recall on the val set.
We almost shipped it as a write filter. Then Jay asked three questions:
1. Are train/test separated at the topic level? NO -- all val topics
   appear in train.
2. Does it generalize to unseen topics? YES for signal/noise, NO for
   topic prediction.
3. Does it survive subtle noise? NO -- 19% accuracy.

**Lesson:** A clean val split by example is not the same as a clean split
by concept. The model memorized "Sprint planning = NOISE" and "cache_v3 =
DECISION" as lookup entries, not as generalizable patterns. Always test
on held-out concepts, not just held-out examples.

**How to avoid:** Before celebrating a metric, ask "what distribution shift
would break this?" and test it. In our case: same-vocabulary-different-role
was the distribution shift that broke the 100/100.

### Mistake 2: Assuming LLM synthesis fixes bad clustering

In the consolidation experiments, the hypothesis was: "embedding clustering
produces bad facts, but if we use an LLM to synthesize instead of
concatenate, the LLM will produce better facts."

Result: LLM synthesis and concatenation produce identical accuracy (75%)
because both consume the same contaminated cluster. The LLM cannot
distinguish signal from noise if the cluster already mixed them.

**Lesson:** Garbage in, garbage out applies to LLMs too. The synthesis
quality is bounded by the input quality. If the clustering step fails,
no downstream processing can fix it.

**How to avoid:** Before optimizing a downstream step, verify the upstream
step is producing clean input. We should have measured cluster purity
BEFORE trying different materialization functions.

### Mistake 3: Embedding consolidation tested without checking the data first

We ran consolidation experiments (LoCoMo, knowledge_update) and got
negative results. Only AFTER the experiments did we measure the embedding
distribution and discover that intra-topic and cross-topic similarities
overlap at [0.700, 0.770].

Silt's rule says: "before proposing an algorithm, look at the distribution
of the data." We violated this by running experiments first and analyzing
the data second.

**Lesson:** If we had measured the embedding distribution first, we would
have predicted the failure without running the experiments. The overlap
zone makes clean clustering impossible at any single threshold.

**How to avoid:** Start every experiment with a distribution probe.
5 minutes of numpy > 2 hours of running a doomed experiment.

### Mistake 4: Overfitting to the easy test

v1's 88.6% overall accuracy looks good until you realize it's 100% on
noise (88% of the val set) and 0% on signal (12% of the val set).
The overall number is dominated by the easy class.

**Lesson:** Class-imbalanced evaluation hides failures on the minority
class. Always report per-class accuracy, not just overall.

**How to avoid:** Report accuracy per class. If the classes are imbalanced,
use the minority class accuracy as the headline number. "0% signal
accuracy" is the real number, not "88.6% overall."

### Mistake 5: ForgetConsolidated without coverage verification

We tested ForgetConsolidated and it tombstoned ALL original events after
consolidation, leaving only 1 consolidated fact for retrieval. Accuracy
collapsed from 23% to 0.7% (LoCoMo) and from 100% to 25%
(knowledge_update).

**Lesson:** Forgetting is irreversible damage if the consolidated artifact
doesn't fully capture the original content. The current implementation
tombstones based on "was consolidated" (binary), not "is fully covered
by the consolidated artifact" (verified).

**How to avoid:** Never tombstone without verifying coverage. The brief-aware
forgetting model (Layer 2 in three-layers-of-forgetting.md) addresses this
by checking content coverage before tombstoning.

### Mistake 6: MPS segfaults during mixed PyTorch + vstash workloads

Running nanoGPT inference (PyTorch MPS) in the same process as vstash
embedding (fastembed/ONNX) caused segfaults. We lost 20 minutes debugging
before discovering the issue and separating the workloads.

**Lesson:** MPS and other GPU backends don't play well with multiprocessing
libraries that also use GPU/accelerator resources.

**How to avoid:** When mixing PyTorch MPS with other ML libraries, either:
(a) force CPU for one of them, (b) run in separate processes, or
(c) free the model before loading the other library (del model).

### Mistake 7: nanoGPT training output lost due to shell buffering

Three training runs produced no visible output because Python's stdout
buffering + shell pipe filtering ate the log lines. We restarted training
multiple times before figuring out the buffering issue.

**Lesson:** Python buffers stdout when piped. nanoGPT's print() calls
don't reach the log file until the buffer flushes.

**How to avoid:** Always use `python3 -u` (unbuffered) when capturing
training output, and redirect to a file (`> log.txt 2>&1`) instead of
piping through grep.

### Mistake 8: brief_v1 fingerprint bug -- all briefs got same hash

All briefs in a consolidation cycle share the same `derived_from` (all
event paths). `fact_fingerprint()` hashes `derived_from`, so all briefs
got the same hash. In vstash, title is the key, so each brief overwrote
the previous one. Only 1 of 4 briefs survived.

Result: merken-brief showed 45% accuracy (same as baseline) instead of
the expected 86-96%. We spent 30 minutes debugging retrieval quality
before discovering the storage bug.

**Lesson:** When multiple artifacts share the same provenance, the
fingerprint must include content-specific information, not just provenance.

**How to avoid:** Use content hash (sha1 of brief text) for per-brief
identity. Use provenance hash (sha1 of event paths) for idempotency
checks across consolidation cycles. Two different hashes for two
different purposes.

### Mistake 9: Comparing configs that use different amounts of context

In the first LoCoMo run, vstash-raw used session-level ingestion (large
chunks, ~5800 tokens fed per query) while the initial attempt used
turn-level ingestion (tiny chunks, ~354 tokens per query). Turn-level
retrieval gave 0% accuracy because individual turns were too short for
meaningful embedding.

**Lesson:** Chunk size is a fundamental parameter that must be held
constant across configurations for fair comparison. Changing chunk size
changes what retrieval CAN find, not just how well it finds it.

**How to avoid:** Fix chunk strategy as the first design decision,
before any experiments. Document it as an experimental parameter.

### Mistake 10: torch + fastembed segfault across subprocess boundaries

Running `experiments/consolidation/test_write_filter.py` with all three
configs (AlwaysWrite, char v2, BPE v4) in one process segfaulted on the
second config -- Mistake #6 reappearing. The obvious fix was to split
into three separate `python -m ...` invocations. It did not work: both
char and BPE processes segfaulted with exit 139 *immediately on first
`NanoGPTWriteDecider(ckpt, meta)` construction*, even in a freshly
forked Python interpreter.

Root cause is subtler than Mistake #6 suggested. `merken/__init__.py`
imports `merken.memory`, which transitively imports vstash, which loads
fastembed/ONNX Runtime. The *order of first load* matters -- once
fastembed has touched the process, loading a torch model corrupts the
runtime. A fresh subprocess does not help if the script's first
statement is `from merken import AlwaysWrite, Memory`.

Two workarounds confirmed to work:

1. `import torch` at the very top of the script, before any merken
   import. Torch initializes before fastembed/ONNX and they coexist.
   Used for `precompute_nanogpt.py`.

2. Pre-compute nanoGPT decisions offline (torch-only process), save to
   JSON, then read through a torch-free `_PrecomputedDecider` at E2E
   time. Used for `test_write_filter.py` so the recall pipeline never
   touches torch.

**Lesson:** On macOS, torch and fastembed/ONNX are first-load-order
sensitive. Subprocess isolation is necessary but not sufficient -- the
first native library to load wins the process and corrupts later ones.

**How to avoid:** For any experiment that mixes torch with vstash, pick
one: (a) make torch load first in every entry point, or (b) separate
inference (torch) from retrieval (vstash) into distinct processes
communicating via files/pipes.

### Non-mistake: Things that worked on first try

Worth documenting to avoid second-guessing correct decisions:

- **AlwaysWrite for experimental configs:** Isolates the effect being
  measured (consolidation, forgetting) from the write filter's behavior.
- **Gemini 2.0 Flash as judge:** Stable, fast, cheap. Never disagreed
  with manual spot-checks on answer correctness.
- **Typed schemas for briefs (DECISION/ENTITY/EVENT/FREE):** The LLM
  consistently picks the right schema. No iteration needed.
- **Character-level tokenization for v1:** The simplest possible choice.
  Good for learning, good for signal/noise. Only fails on tasks that
  require word-level understanding (topic prediction).
- **Separating brief retrieval from episodic retrieval:** The
  architectural insight was correct on first implementation. 86% vs 38%
  when briefs compete in episodic pool validates the dual-channel design.

---

## Open questions for future sessions

1. Does v3 (verb markers) improve borderline accuracy over v2 (69%)?
   Val loss suggests yes (0.287 vs 0.335) but val loss and task accuracy
   are not perfectly correlated.

2. Would combining v2's data volume (1922 examples) with v3's verb markers
   give the best of both? v3 has better features but fewer examples.

3. Can BPE tokenization enable topic prediction? Character-level gives 0%
   on topics. Word-level tokens ("Redis", "Caffeine") would be directly
   matchable.

4. What's the minimum model size for signal/noise? 800K params is probably
   overkill for binary classification. A 2-layer, 64-dim model (~50K params)
   might suffice and would be 16x cheaper to run.

5. Can the same model architecture handle the 4-way Hindsight classification
   (world/experience/opinion/entity)? Or does that require more capacity?

6. Does the vitality scoring approach (continuous score instead of binary
   classification) require a different training objective (regression instead
   of next-token prediction)?
