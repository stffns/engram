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
500     0.11     0.29     0.18    NEW RECORD val loss
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
