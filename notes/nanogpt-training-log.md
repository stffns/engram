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
500     0.11     0.29     0.18    <-- BEST val loss (new record)
600     0.08     0.29     0.21    val plateaued, overfitting
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
| v3 | 608 (+verb markers) | **0.287** | TBD | 2x faster, 14% better val |

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

### Production requirements for write filter
- Latency: <5ms per event (current: ~1ms on CPU)
- Recall: >99% (no false negatives on real decisions)
- Precision: >90% (some false positives acceptable)
- Model size: <10MB checkpoint (current: ~10MB, could quantize to ~3MB)

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
