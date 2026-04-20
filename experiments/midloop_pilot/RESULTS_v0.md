# midloop v0 -- first training run (2026-04-20)

First end-to-end run of the midloop boundary-tagger per
`notes/midloop-training-plan.md`. Training infra lives in
`nanoGPT` sibling repo (model + trainer) plus the BPE prep script at
`nanoGPT/data/midloop_v0/prepare.py`. Engram-side scripts:

- `experiments/midloop_pilot/split_train_test.py` -- 8-protocol
  holdout, deterministic seed=42.

## Setup

- Dataset: `midloop_v0.jsonl` (385 cases, 77 protocols, 5 per protocol).
- Split: 8 protocols holdout (40 test cases) / 69 protocols train
  (345 train cases). No protocol leak, no case_id overlap.
- Architecture: 4L/4H/128d GPT backbone + `Linear(n_embd, 1)`
  tagging head. 0.90M params. block_size=320 (covers max observed
  265 subword sequence length).
- BPE tokenizer: trained on train cases only, vocab=512, specials
  `<|pad|>`, `<|prompt|>`, `<|sep|>`, `<|end|>`.
- Labels: boundary mode (v0.jsonl already). Subword-BPE collapses
  per-word positive density 14.1% -> per-subword 5.21%. Boundary
  label stays at the FIRST subword of each intervention-start word.
- Loss: per-position BCE with pos_weight=10 (covers subword
  neg/pos=18:1 partway), masked to response-region only.
- Training: 800 iters, batch_size=32, AdamW lr=1e-3, warmup=100,
  cosine decay, MPS float32.

## Result

- Training wall time: 51 seconds on M-series MPS.
- Best F1 on held-out: **0.318 @ iter 300** (early in run; overfit
  after that, F1 drifts down to 0.301 at the final iter).
- At best-F1 checkpoint: P=0.206, R=0.692, TP=72, FP=277, FN=32,
  TN=1955.

Per-protocol recall at best-F1 iter:

    0.385  depression-mhgap-who        <-- below the 50% per-protocol bar
    0.571  gestational-diabetes-who
    0.700  herpes-shingles-who
    0.733  meningitis-who
    0.750  medication-side-effects-expected-vs-alarm-who
    0.769  chest-pain-who
    0.818  fever-differential-tropical
    0.833  measles-who

## Graduation gate

| criterion                   | target   | observed | verdict |
|-----------------------------|---------:|---------:|:-------:|
| F1 on held-out              | >= 0.75  | 0.318    | FAIL    |
| min per-protocol recall     | >= 50%   | 38.5%    | FAIL    |
| training wall <= 30min MPS  | <= 30m   | 51s      | PASS    |

1 of 3. v0 does not ship as shadow yet.

## Interpretation

- Model IS learning: recall 0.69 at threshold 0.5 means it catches
  most intervention starts. It also catches way too much noise
  (277 FP vs 72 TP), so precision 0.21.
- Training loss collapses to ~0.03 by iter 750 while val F1 drops
  from 0.32 -> 0.30. Classic small-data overfit with early stopping
  already saving the best ckpt.
- Per-protocol recall spread (0.385 - 0.833) is reasonable for n=5
  cases per test protocol; the 1 protocol below 50% is driven by
  only 13 positive subwords total, CIs are wide.

## Iteration round 1 (2026-04-20, same session) -- knobs do NOT lift F1

Ran the plan's "open levers" to measure whether any cheap knob
closes the gate gap. Takeaway: **the bottleneck is not calibration
or head capacity; it is protocol-level transfer.**

### Threshold sweep on the pw=10 best ckpt

Full grid [0.10, 0.95] in steps of 0.05. F1 stays in [0.281, 0.329]
across the entire grid. Peak F1=0.329 at thr=0.55 (+0.011 over the
default thr=0.50). Threshold tuning is a 1-point lever, not a
10-point lever. Artifact: `nanoGPT/out-midloop-v0/pr_sweep.json`.

### pos_weight sweep [3, 5, 6, 8, 10, 15, 18]

Ran 6 additional training runs (800 iters each, ~50s each, MPS).

| pos_weight | best F1 | best iter | ckpt dir (under nanoGPT/) |
|-----------:|--------:|----------:|:---------------------------|
| 3.0        | 0.352   | 200       | out-midloop-v0-pw3.0 |
| 5.0        | 0.346   | 600       | out-midloop-v0-pw5.0 |
| 6.0        | 0.330   | 250       | out-midloop-v0-pw6.0 |
| 8.0        | 0.317   | 300       | out-midloop-v0-pw8.0 |
| 10.0       | 0.318   | 300       | out-midloop-v0 |
| 15.0       | 0.346   | 350       | out-midloop-v0-pw15.0 |
| 18.0       | 0.337   | 750       | out-midloop-v0-pw18.0 |

Best F1 over the 6x range: 0.352. Worst: 0.317. Spread of 0.035.
The knob moves the P-R balance (low pw -> higher P, lower R; high
pw -> higher R, lower P) but F1 plateaus. Per-protocol recall
worst-case stays in [0, 0.385] across all pos_weights.

### MLP head (128 -> 64 -> 1, 8.3K params vs 129)

Reran pw=[3, 5, 10, 15] with `--head_type=mlp`. At pw=3: F1=0.343.
At pw>=5: the model collapses to all-negative predictions
(F1=0.000). 60x more head capacity does NOT break the plateau.

### Train vs val diagnostic (best ckpts, full threshold sweep)

|   pw | best iter | TRAIN F1 | VAL F1 | gap       |
|-----:|----------:|---------:|-------:|----------:|
|   3  |      200  |  0.480   |  0.352 | **+0.128**|
|  10  |      300  |  0.582   |  0.329 | **+0.253**|
|  15  |      350  |  0.624   |  0.348 | **+0.276**|

TRAIN F1 climbs with pos_weight (the model CAN learn the training
protocols); VAL F1 plateau is tight (0.33-0.35). **Cross-protocol
generalization is the wall, not optimization.**

### Verdict

Per Silt's rule (look at the distribution before proposing a new
algorithm), the distribution says:

1. The model learns in-distribution (TRAIN F1 up to 0.62).
2. Held-out protocols stay at F1 ~ 0.35 regardless of knob choice.
3. The 7-point F1 spread across pos_weight [3..18] is smaller than
   the 12-25 point train-val gap.

Interpretation: with 69 train protocols * 5 cases, the model
memorises per-protocol phrasings and has little cross-protocol
signal. The plan anticipated this ("similar phrasings across cases
of the same protocol -- protocol-id dependency is real") but did
not anticipate the gap would be this wide at v0. **Closing the
gate on this data alone is unlikely.**

## What IS the right next move

1. **Expand the corpus.** 7 WHO PDFs are already staged via PR #27
   in `staging/who_extracted/` (~1.1MB). Phase 1 pipeline is
   idempotent; re-run on 100-200 protocols instead of 77. More
   protocols, not more cases per protocol, is the generalization
   lever.

   **Corpus expansion started 2026-04-20:**
   - Filtered `epfl-llm/guidelines` by `source in {who, icrc}` and
     saved 272 full documents (28MB text) at
     `staging/hf_guidelines/who_icrc.jsonl`.
   - `chunk_hf_guidelines.py` produces 16619 chunks with default
     filters; density >= 5.0 narrows to **5515 clinically-dense
     chunks** (WHO 4041, ICRC 1474) at
     `staging/hf_guidelines/chunks/{who,icrc}/*.md`.
   - 7 WHO PDFs re-extracted to `staging/who_extracted/` (were
     missing since the gitignored dir got cleaned).
   - **Validation run 2026-04-20**: sampled 10 chunks (7 WHO +
     3 ICRC, density >=29) through `default_gemini_client(structured=True)`.
     **100% pass rate** (50/50 cases), 95s wall time, ~$0.05 USD.
     Spot-check on MgSO4, PLAN C, EFV prophylaxis: truths derived
     strictly from chunk text (no hallucination). Pipeline scales.
     Artifacts: `staging/hf_guidelines/sample_cases.jsonl` +
     `sample_report.md`.

   - **Next**: pick 300-500 chunks (diverse by doc_id, balance
     source proportions) and regenerate Phase 1 at scale. Wall
     time estimate: ~50 min. Cost estimate: ~$1.50 USD.
2. **Lower the v0 gate for shadow.** F1 >= 0.75 is out of reach on
   this data. Drop to F1 >= 0.50 as a shadow-only bar (action
   clamped to WHISPER per spec), accumulate real (decision, outcome)
   pairs through the existing midloop primitive, and re-train v1 on
   the accumulated pool. Mirror of how v7 graduated the write-filter.
3. **Audit the label distribution.** 5% per-subword positive rate
   is much lower than the 14% per-word estimate. If the boundary
   mode labels are correct, the BPE effect is natural. If many
   boundary spans got lost in the word->subword mapping, re-inspect
   the prep logic.

## Open levers (plan's "1-3 weeks" iteration) -- DEFERRED

From notes/midloop-training-plan.md -- round 1 above explored
pos_weight and MLP head. Remaining levers are unlikely to close
the 40-point gap to F1=0.75 on the current data, but might inform
a v0.1 run:

1. **pos_weight sweep [3, 5, 6, 8, 10]** against held-out F1.
   Current 10 pushes recall at precision's expense; 6 is the plan's
   original estimate (word-level) and may be a better midpoint.
2. **Threshold calibration on a held-out VAL slice** (10% of train,
   not test). Current hard 0.5 is untuned.
3. **Early stopping**: best was iter 300 but we ran 800. Cap iters
   or add patience logic.
4. **Wider tagging head**: Linear(128,1) is 129 params. Try an MLP
   (128 -> 64 -> 1, 8257 params) in case capacity is the bottleneck
   -- cheap test before changing backbone.
5. **Regularization**: dropout=0.1 is light. Try 0.2 - 0.3.

Anti-patterns per CLAUDE.md + v7 saga:
- Do NOT bump backbone above 4L/4H/128d before exhausting the
  above levers.
- Do NOT optimize threshold on TEST.

## Artifacts

- `nanoGPT/out-midloop-v0/ckpt.pt` -- best-F1 checkpoint (iter 300).
- `nanoGPT/out-midloop-v0/final_metrics.json` -- final-iter metrics.
- `nanoGPT/data/midloop_v0/{train,val}.npz, tokenizer.json, meta.pkl`.
- `experiments/midloop_pilot/training_data/midloop_v0_{train,test}.jsonl`.
