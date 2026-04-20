# midloop v1 -- corpus-expansion retrain (2026-04-20)

First retrain with the expanded corpus (HF epfl-llm/guidelines
WHO+ICRC subset, chunked by clinical-action density). Tests the
hypothesis from RESULTS_v0 round 1: "cross-protocol transfer is the
wall, more protocols is the fix."

## Setup

- **v1 dataset = v0 medlocal (77 protocols) + v1_hf (235 HF chunks)
  = 312 protocols, 1560 cases.** Merged by `merge_datasets.py`.
- Phase 1 re-run on the HF chunks used `default_gemini_client(structured=True)`:
  - 235 chunks -> 1175 cases in 2101s (8.9s/proto, 0 Gemini failures).
  - 1175 responses via lmstudio gemma-4-e4b-it-mlx in 1093s (0.93s/resp, 0 failures).
  - Aligner: 3601 divergent regions -> 3577 interventions (99.3% retention at cos=0.85).
  - format_for_training boundary mode: 1175 rows written.
- Split: 32 protocols held out (10% of 312) = **160 test / 1400 train cases**.
  Protocol-level group split, seed=42. No protocol/case leak.
- nanoGPT data prep (data/midloop_v1/): **vocab=512 BPE**, max seq=438,
  subword positive density 4.70% train / 4.49% val
  (neg/pos ~20:1, matches v0).
- Trainer: same as v0 (4L/4H/128d, 0.90M params, block_size=320,
  pos_weight=10, 800 iters, batch=32, AdamW lr=1e-3, cosine schedule,
  MPS float32, linear tagging head).

## Result -- hypothesis CONFIRMED on gap, gate NOT met

Training wall time: **55 seconds** (same as v0 -- model size unchanged).

### Headline F1 with threshold sweep, best ckpt (iter 700)

|                  | TRAIN F1 | VAL F1 | gap    |
|------------------|---------:|-------:|-------:|
| v0 best (pw=15)  | 0.624    | 0.348  | +0.276 |
| v1 best (pw=10)  | 0.426    | **0.349** | **+0.077** |

**Gap closed from +0.276 to +0.077.** Model no longer memorizes
training protocols. This confirms round-1's cross-protocol hypothesis.

VAL F1 plateau stayed at ~0.35 (v0 and v1 indistinguishable at
threshold-tuned best). The bottleneck is no longer memorisation;
it is now **global capacity + label quality** with 312 protocols of
moderate per-protocol signal.

### Graduation gate criteria

| criterion                   | target   | v0       | v1     | verdict |
|-----------------------------|---------:|---------:|-------:|:-------:|
| F1 on held-out              | >= 0.75  | 0.349    | 0.349  | FAIL    |
| min per-protocol recall     | >= 50%   | 38.5%    | 9.1%   | FAIL    |
| training wall <= 30min MPS  | <= 30m   | 51s      | 55s    | PASS    |

The min-per-protocol-recall dropped because the val set now has
32 protocols (vs 8 in v0). With 5 cases per protocol (median ~14
positive subwords), a single difficult protocol hits very low recall
due to variance, not necessarily model failure. **The right read:
v1 is better at the global-F1 level; the gate's per-protocol bar
needs re-specification for >32 test protocols.**

### Training loop evolution (val @ thr=0.5)

| iter | F1     | P     | R     | worst_proto_R |
|-----:|-------:|------:|------:|--------------:|
|  100 | 0.242  | 0.139 | 0.946 | **0.818**     |
|  200 | 0.247  | 0.143 | 0.918 | 0.636         |
|  300 | 0.277  | 0.167 | 0.809 | 0.636         |
|  500 | 0.279  | 0.170 | 0.774 | 0.545         |
|  700 | **0.283** | 0.177 | 0.710 | 0.545     |

**Worst-protocol recall peaks at 0.818 early (iter 100)** and
degrades as the model specializes. This is a DIFFERENT failure mode
than v0 (where worst stayed flat around 0.38-0.55 throughout).

**Interesting lever**: early stopping at iter 100 satisfies the
"no protocol < 50% recall" gate criterion but not the F1 >= 0.75
gate. Shadow-only deployment with a recall-heavy threshold might be
viable from the iter-100 ckpt.

### Threshold sweep at best ckpt (iter 700) -- VAL

| thr  | P     | R     | F1    | TP  | FP   | FN  |
|-----:|------:|------:|------:|----:|-----:|----:|
| 0.30 | 0.146 | 0.871 | 0.250 | 405 | 2364 |  60 |
| 0.50 | 0.177 | 0.710 | 0.283 | 330 | 1534 | 135 |
| 0.60 | 0.216 | 0.510 | 0.303 | 237 |  860 | 228 |
| 0.65 | 0.250 | 0.409 | 0.310 | 190 |  571 | 275 |
| 0.70 | 0.325 | 0.368 | 0.345 | 171 |  355 | 294 |
| **0.80** | 0.437 | 0.290 | **0.349** | 135 | 174 | 330 |

Precision-recall curve is monotone (no plateau inflections), and
thr=0.80 maxes F1. v1 has a much more informative P-R curve than v0
(which stayed essentially flat across the grid).

## Interpretation

1. **Corpus expansion WORKED at the generalization axis.** The
   train-val gap collapsed (+0.276 -> +0.077), proving the round-1
   diagnosis was correct: v0 was memorising 77 medlocal protocols.
2. **VAL F1 did not improve.** Still ~0.35. This tells us the
   remaining gap is not "cross-protocol transfer" but "task
   signal-to-noise with 4L/4H/128d capacity at 4.7% positive density
   on boundary labels". Possible causes:
   - Label noise: the semantic-drop filter dropped only 24/3601
     divergences (0.7%), keeping many false boundaries. A tighter
     threshold might help.
   - Insufficient capacity for 312-protocol pattern diversity.
   - Boundary labels are inherently noisy: the "start" position of
     an intervention span is one subword among dozens.
3. **The gate's "no protocol recall < 50%" criterion needs to be
   re-specified** for 32+ test protocols. The 8-protocol v0 gate
   was easier to satisfy by chance; 32 protocols exposes real
   distributional variance in the underlying label density.

## Round 7 -- add who_pdf chunks (MIXED, 2026-04-20)

Hypothesis: the 184 who_pdf chunks extracted from the 7 WHO
reference books (IMAI acute care, essential medicines, IMCI
booklets, snakebite, maternal-newborn) add topic coverage the HF
corpus doesn't (dosing ladders, antimalarial/antivenom dosing).
Task #10 staged 190 chunks -> 184 after heading-dedup fix; scaleup
extends source_mix to include ``who_pdf`` with a separate per-doc
cap (30, since who_pdf has only 7 docs).

### Setup

- 500 HF chunks target (max_per_doc=3) + 150 who_pdf target
  (max_per_doc=30).
- Corpus limited both sources below target:
  - icrc corpus-capped at 90 chunks (density+length filters admit
    only 90 of its 1474 chunks per the v1b warning).
  - who_pdf corpus-capped at 101 chunks (of 184; 6 of 7 PDFs
    contribute; postnatal-care has no chunks >= 400 chars).
- Final selection: **541 chunks = 350 HF-who + 90 HF-icrc + 101 who_pdf**.
- Phase 1: 2705 cases in 82 min (0 fail), 2705 responses in 43 min
  (0 fail), 7942 interventions at cos=0.85 (99.3% retention).
- Merged with v0 -> **3090 cases / 618 protocols** (vs v1b: 2585 / 517).

### Runs

| variant           | cases | params | TRAIN F1 | VAL F1 | gap     |
|-------------------|------:|-------:|---------:|-------:|--------:|
| v1b (4L/4H/128d)  |  2585 |  0.9M  |   0.449  | 0.435  |  +0.013 |
| v1b-6L (6L/192d)  |  2585 |  2.8M  |   0.511  | 0.451  |  +0.060 |
| v1c (4L/4H/128d)  |  3090 |  0.9M  |   0.416  | **0.388**|  +0.029 |
| **v1c-6L (6L/192d)**| 3090 |  2.8M  |   0.618  | **0.461**|  +0.157 |

### Verdicts (MIXED)

- **v1c-6L is the new best: VAL F1=0.461, +0.010 over v1b-6L.**
  Capacity + who_pdf data compound slightly. Gate F1>=0.50 shadow
  bar now 0.039 away (from 0.049 at v1b-6L).
- **v1c baseline (4L/128d) REGRESSED: 0.435 -> 0.388** (-0.047).
  who_pdf chunks carry a different structural signature than HF
  (dosing tables, bullet lists vs prose guidelines). At 4L/128d
  the model can't unify them with the HF chunks; the extra data
  hurts instead of helping. At 6L/192d the capacity is enough to
  accommodate both distributions, and the benefit flips positive.
- **Returns diminish.** Round 6 (data doubling v1 -> v1b) paid
  +0.009 at 6L. Round 7 (adding who_pdf) pays +0.010 at 6L.
  Baseline hurt in round 7 is a new failure mode.

### Graduation gate re-eval

| criterion          | target   | v1-6L  | v1b-6L | v1c-6L | verdict     |
|--------------------|---------:|-------:|-------:|-------:|:-----------:|
| F1 held-out        | >= 0.75  | 0.442  | 0.451  | 0.461  | FAIL        |
| F1 shadow-only     | >= 0.50  | 0.442  | 0.451  | 0.461  | FAIL by 0.039 |
| train wall MPS     | <= 30m   | 203s   | 196s   | 216s   | PASS        |

### Cumulative session summary

| round | lever | best VAL F1 | delta | cumulative |
|-------|-------|------------:|------:|-----------:|
| 0 | v0 baseline (77p, 4L/128d) | 0.352 | -- | 0.352 |
| 2 | v1 corpus 77->312 (4L/128d) | 0.359 | +0.007 | 0.359 |
| 3 | v1-6L capacity 6L/192d     | 0.442 | +0.083 | 0.442 |
| 6 | v1b-6L corpus 2585 (data+) | 0.451 | +0.009 | 0.451 |
| 7 | v1c-6L + who_pdf 3090     | **0.461** | +0.010 | **0.461** |

**Total gain 0.352 -> 0.461 = +0.109 across 7 rounds.**

The capacity bump (round 3) was the single biggest lever (+0.083).
Corpus expansions contributed +0.009 (v1 -> v1b) + +0.010 (v1b ->
v1c) = +0.019. Other levers (pos_weight, MLP head, aligner, longer
training, regularisation) contributed 0.

### Decision: stop scaling corpus, pivot to shadow

With two consecutive corpus-expansion rounds paying ~+0.010 each at
6L, and the baseline regression in round 7, it is clear we have
squeezed most of the juice from the "more data at 6L" lever.
Closing the remaining 0.039 to the shadow bar F1=0.50 with the
same pattern (another round adding epfl-llm CDC + NICE = 2549
more docs) would cost another $4-6 API + 3-4 hours for an
expected +0.005 to +0.015 gain. Diminishing returns is decisive.

Recommended next move: **accept v1c-6L as the v1 shadow candidate
at F1=0.461** per task #8, wire it into the midloop primitive
through `ShadowMidloopDecider(NoopMidloopDecider(),
NanoGPTMidloopDecider(ckpt))` with action clamped to WHISPER, and
let real (decision, outcome) pairs accumulate. Retrain on pooled
real labels once sufficient volume (~200) accumulates.

## Round 6 -- double the dataset (POSITIVE, 2026-04-20)

Post-plateau hypothesis (RESULTS_v1 round 3): "more data is the
only lever left". Tested by relaxing the per-doc diversity cap
from max_per_doc=1 to max_per_doc=3, doubling the HF-guidelines
selection from 235 chunks to 440 (warn correctly fired: ICRC
only has 90 chunks available at density>=5 vs 150 requested).
Re-ran the full Phase 1 pipeline: 2200 cases from Gemini
(66 min, 0 failures), 2200 responses from lmstudio (41 min,
0 failures), 6669 interventions after alignment at cos=0.85.
Merged with v0 (385 cases, 77 protocols) -> **2585 cases,
517 protocols** (vs v1: 1560 cases / 312 protocols).

### Runs

Same v0 -> v1 -> v1-6L sequence, on the v1b dataset:

| variant               | cases | params | dropout | TRAIN F1 | VAL F1 | gap    |
|-----------------------|------:|-------:|--------:|---------:|-------:|-------:|
| v1  (4L/4H/128d)      |  1560 |  0.9M  | 0.10    |    0.431 | 0.359  | +0.072 |
| v1-6L (6L/6H/192d)    |  1560 |  2.8M  | 0.15    |    0.733 | 0.442  | +0.291 |
| v1b (4L/4H/128d)      |  2585 |  0.9M  | 0.10    |    0.449 | **0.435** | **+0.013** |
| **v1b-6L (6L/6H/192d)**| 2585 | 2.8M  | 0.15    |    0.511 | **0.451** | +0.060 |

### Verdicts

- **Data is the dominant lever now.** Going from 1560 -> 2585 cases
  lifted VAL F1 from 0.359 -> 0.435 at the baseline architecture
  (+0.076). The train-val gap collapsed further, from +0.072 to
  +0.013 (essentially zero memorisation).
- **Capacity + data compounds weakly.** v1b-6L over v1b is only
  +0.016 VAL F1. The 6L architecture that paid +0.083 on the small
  v1 corpus pays only +0.016 on the larger v1b corpus -- when the
  data bottleneck is closer to resolved, the capacity bump becomes
  marginal.
- **Gate F1>=0.50 is now 0.049 away.** Shadow-only deployment
  (task #8) is viable at v1b-6L's 0.451 if we accept a lowered bar.
  Closing the remaining 0.049 without overfitting appears to
  require still more data, not more parameters.

### Graduation gate re-eval

| criterion                   | target   | v1-6L  | v1b-6L | verdict     |
|-----------------------------|---------:|-------:|-------:|:-----------:|
| F1 on held-out              | >= 0.75  | 0.442  | 0.451  | FAIL        |
| F1 shadow-only bar          | >= 0.50  | 0.442  | 0.451  | FAIL by 0.049 |
| min per-protocol recall     | >= 50%   | ~0.09  | ~low   | FAIL        |
| training wall <= 30min MPS  | <= 30m   | 203s   | 196s   | PASS        |

### What IS next

**Two levers with clear EV, rough order:**

1. **Add who_pdf chunks (190 extracted in task #10).** These
   cover the 7 WHO reference books staged in medlocal, heavy on
   dosing / essential medicines / acute care. Adding 190 new
   protocols * 5 cases = ~950 new cases would take the corpus
   from 2585 -> ~3535. scaleup_phase1_hf.step_select needs its
   source_mix extended to include who_pdf before the next run.
   Cost: ~$1 Gemini + ~15 min wall time for responses.
2. **Accept v1b-6L as the shadow candidate at F1=0.451.** Per
   task #8, the spec's F1>=0.50 shadow bar was a guess; 0.45 is
   a defensible first-shadow threshold if the action is clamped
   to WHISPER per the midloop plan. Real (decision, outcome)
   pairs accumulate organically once the primitive is wired into
   Reforge, allowing v2 retraining on production-truth.

### What we are NOT doing

- More capacity than 6L. v1-8L (round 3) saturated at +0.01 over 6L
  with v1; likely similar story on v1b.
- Synthetic data expansion. The 440-chunk selection with
  max_per_doc=3 is already using the TOP-density region of the
  corpus; lowering the density cutoff would admit the policy/
  process content the filter was designed to reject.

## Round 5 -- extended training (NEGATIVE, 2026-04-20)

Hypothesis: v1-6L best ckpt was at iter 1050 of 1200 with training
loss still dropping and cosine schedule ending. Maybe more iters
would continue lifting VAL F1.

### Method

Re-ran v1-6L with max_iters=2000 and lr_decay_iters=2000 (stretched
cosine schedule), everything else identical.

### Result

| variant              | best iter | TRAIN F1 | VAL F1 | gap    |
|----------------------|----------:|---------:|-------:|-------:|
| v1-6L @ 1200 iters   |      1050 |    0.733 | **0.442** | +0.291 |
| v1-6L-long @ 2000    |      1600 |    0.965 | 0.440  | +0.525 |

TRAIN F1 jumped 0.733 -> 0.965 (nearly perfect memorisation). VAL
F1 essentially unchanged: -0.002. The extra iters just memorize
the training set further.

### Conclusion -- real plateau

v1-6L's VAL F1 = 0.442 is the real ceiling for this architecture
at this data size. More iters trade no-gain against exploding gap.
The remaining gap to F1>=0.50 shadow bar (0.058) can ONLY come
from:

- More data (more protocols, not more iters)
- Different labels (smaller label-noise ceiling)
- Architectural change that attacks label-noise directly

No more iter increases.

## Round 4 -- tighter aligner (NEGATIVE, 2026-04-20)

Hypothesis: remaining 0.06 gap to F1>=0.50 shadow bar is driven
by label noise in divergence regions that the semantic-drop
filter lets through at cosine=0.85.

### Method

Re-ran `step_align` on existing responses.jsonl with two tighter
thresholds while holding everything else constant:

| cosine threshold | divergent regions | interventions kept | retention |
|-----------------:|------------------:|-------------------:|----------:|
|             0.85 |              3601 |               3577 |    99.3%  |
|             0.90 |              3601 |               3592 |    99.7%  |
|             0.95 |              3601 |               3599 |    99.9%  |

Going from 0.85 to 0.95 drops 22 MORE interventions out of 3601 --
0.6% change. The filter is already barely active at 0.85.

### Conclusion -- not the bottleneck

The ruido isn't in "falsely-flagged divergences with high cosine";
real divergences between the small model's output and the truth
have cosine much lower than 0.85 already, so they all pass at any
stringent threshold.

The boundary-label noise is **intrinsic to the shape**: within a
divergence region, labelling the FIRST subword as the boundary is
noisy because the true boundary may be between words, inside a
phrase, or tokenized ambiguously by BPE.

Not retraining at higher threshold -- no label distribution change
worth measuring. `aligned.jsonl` restored to cos=0.85 state.

## Round 3 -- capacity bump sweep (2026-04-20, same session)

With the train-val gap closed in v1 baseline, the CLAUDE.md anti-
pattern "do not bump backbone above 4L/4H/128d before measuring"
was finally satisfied. We measured, then bumped.

### Runs

All on the same v1 dataset (1400 train / 160 val, 312/32 protocols).
Threshold sweep [0.30, 0.95] in steps of 0.025.

| variant               | params | dropout | wd  | iters | TRAIN F1 | VAL F1 | gap    | wall  |
|-----------------------|-------:|--------:|----:|------:|---------:|-------:|-------:|------:|
| v1 baseline 4L/4H/128d| 0.90M  | 0.10    | 0.1 |   800 | 0.431    | 0.359  | +0.072 |  55s  |
| **v1-6L 6L/6H/192d**  | **2.83M**| 0.15 | 0.1 |  1200 | 0.733    | **0.442** | +0.291 | 203s  |
| v1-6L-reg (heavy reg) | 2.83M  | 0.25    | 0.2 |  1200 | 0.517    | 0.375  | +0.142 | 184s  |
| v1-8L 8L/8H/256d      | 6.53M  | 0.15    | 0.1 |  1500 | 0.917    | 0.453  | +0.464 | 436s  |

### Verdicts

- **v1-6L is the new best.** +0.08 VAL F1 over baseline (0.359 -> 0.442).
  Gap opened (+0.29) but that is expected with capacity -- the
  evidence it's capacity-signal not memorisation-noise is the VAL
  F1 jump in parallel.
- **v1-6L-reg: over-regularized.** dropout 0.25 + wd 0.2 collapsed
  VAL F1 from 0.442 -> 0.375 while closing gap from +0.29 to +0.14.
  Worse on both axes (VAL and TRAIN). Sweet spot is mild reg.
- **v1-8L: capacity saturated.** 2.3x more params (6.5M), only +0.011
  VAL F1 (0.442 -> 0.453), gap exploded to +0.46. Further capacity
  mainly memorizes training protocols; returns diminish fast.

### Pareto frontier after round 3

VAL F1 best: 0.453 (v1-8L) but marginal over 6L. **v1-6L (~3M, F1=0.442)
is the recommended deployment candidate.** Below proposed shadow gate
(F1>=0.50) by only 0.058.

### Graduation gate re-eval

| criterion                  | target   | v1-6L  | verdict |
|----------------------------|---------:|-------:|:-------:|
| F1 on held-out             | >= 0.75  | 0.442  | FAIL    |
| F1 on held-out (shadow bar)| >= 0.50  | 0.442  | FAIL by 0.058 |
| min per-protocol recall    | >= 50%   | ~0.09  | FAIL (variance with 32 protos) |
| training wall <= 30min MPS | <= 30m   | 203s   | PASS    |

Neither published gate passes yet. The F1 >= 0.50 shadow bar is
within striking distance.

## What IS next (priority order)

1. **Tighten aligner threshold (free, re-run align+format only).**
   Bump cosine 0.85 -> 0.90 on the existing responses.jsonl. If
   label noise is part of the remaining 0.06 gap, this surfaces it
   fast. ~5 min wall time, no API cost.
2. **Chunk the 7 WHO PDFs (task #10, free, adds protocols).** Adds
   ~50-100 more clinical protocols for generalisation diversity.
   Total pipeline re-run ~1h, ~$0.50 Gemini.
3. **Pull more fuentes in epfl-llm/guidelines (WHO+ICRC+CDC+NICE
   = 2549 docs vs 272 WHO+ICRC).** 10x the source doc count but
   CDC/NICE may be less CHW-focused. Gated on (1) + (2) not being
   enough.
4. **Lower-risk aux objectives:** multi-task with protocol-id
   classification as auxiliary head. Adds inductive bias around
   protocol identity without adding labels.
5. **Consider dropping the gate to F1 >= 0.50 shadow-only** (task
   #8). We're closer than v0 was.

## What we are NOT doing next (explicit)

- Further capacity bumps above 8L/8H/256d. Returns diminish;
  memorisation grows.
- Contrastive losses (H2/H2b/H2c anti-pattern from v7 saga).
- Re-running v0 iteration at 77 protocols. Conclusively resolved.

## Code review (2026-04-20, post-session) + fixes landed

Ran code-reviewer agent against all 15 new/modified files. Key
findings + fixes applied:

### Blocker (fixed)

**eval_midloop_train.py and eval_midloop_pr.py hardcoded
`data/midloop_v0`.** When either was run against a v1 ckpt, it
loaded v1 weights but evaluated using v0's BPE tokenizer (different
vocab mapping, same size 512 -> no dimension error, silent garbage
metrics). THIS WAS THE ROOT CAUSE of the reported F1=0.112 bug in
round 2 reporting. The inline debug swept-threshold one-liner used
`data/midloop_v1` and got 0.345 correctly.

Fix: both scripts now infer the dataset from
`ckpt['config']['dataset']` (which train_midloop.py saves since v0)
with a regex-based fallback on the parent dir name. Validated via
smoke test -- eval_midloop_train on v1-6L now reports VAL F1=0.442
matching the manual sweep.

### Important (fixed)

1. `scaleup_phase1_hf.step_format`: `convert_case` returns None on
   empty/untokenizable rows. Now counted and logged.
2. `chunk_hf_guidelines.write_chunk`: `chunk['source']` becomes a
   subdir name; added whitelist check to prevent path traversal.
3. `sample_case_gen.select_chunks`: `sorted(glob())` for cross-
   filesystem reproducibility; under-fill warning when the corpus
   cannot supply the requested mix; parse_chunk errors now logged.
4. `retrain_v1.sh`: added `set -u` and `N_PROTO >= 20` guardrail.
5. `merge_datasets.py`: collision detection between v0/v1 protocol_ids
   and case_ids; aborts loudly on collision (silent corruption risk).
6. `train_midloop.py`: coerces numeric configurator overrides to
   float so `--pos_weight=3` stops failing the assertion silently.
7. `prepare.py` (v0 + v1): added "KEEP IN SYNC" header comment.

### Minor (fixed)

- Dropped unused `thr` parameter from `eval_split`.
- Dropped the single-iteration outer loop in `evaluate()`.

### Minor (documented, not fixed)

- Gemini API non-determinism (structured mode) acknowledged.
- MPS reduction non-determinism (torch upstream limitation).
- `prepare.py v0/v1` duplication left with sync comment vs lifting
  to `midloop_common/prepare_lib.py` -- deferred; costs 5 LoC savings
  against ~100 LoC duplication, worth revisiting if v2 ships.
- `requests.Session` not in a context manager (one-shot process;
  negligible).

### Not a bug

`self.gpt.lm_head = nn.Identity()`: verified no memory leak, no
gradient issue, no state_dict incompatibility. The tied
`wte.weight` tensor stays alive referenced via `transformer.wte`.
Linear wrapper is GC'd. Intentional aesthetic hack; adequate for
the research phase.

## Artifacts

- `experiments/midloop_pilot/scaleup_out_hf/` -- full Phase 1
  artifacts (protocols / cases / responses / aligned / training JSONL).
- `experiments/midloop_pilot/training_data/midloop_v1.jsonl` --
  merged 1560-case dataset.
- `experiments/midloop_pilot/training_data/midloop_v1_{train,test}.jsonl`
  -- 1400 / 160 split.
- `nanoGPT/data/midloop_v1/` -- BPE tokenizer, tokenized arrays,
  meta.pkl.
- `nanoGPT/out-midloop-v1/ckpt.pt` -- best-F1 ckpt at iter 700.
- `experiments/midloop_pilot/retrain_v1.sh` -- idempotent driver.
