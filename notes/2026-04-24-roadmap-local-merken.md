2026-04-24 roadmap: merken towards 100% local
=============================================

Context
-------

merken's CONSTITUTION §3 and §8 are explicit: no new vector storage,
no bespoke compression dialect, vstash-only substrate. The implicit
end-state is a memory loop that runs entirely locally -- no Cerebras,
no Gemini, no external LLM dependency for the memory-maintenance
primitives (should_remember / should_consolidate / should_recall /
should_forget).

Today we use external services for bootstrap and validation:

| component                      | external today               | status             |
|--------------------------------|------------------------------|--------------------|
| Embedder                       | bge-small-en-v1.5 (local)    | already local      |
| Writer filter                  | nanoGPT v7 (local)           | academic POC; fragile off-domain |
| Brief synth (pipeline_runner)  | qwen-3-235b on Cerebras      | marginal retrieval value |
| Builder (Mode A/C/pipeline)    | llama3.1-8b on Cerebras      | production dependency |
| Judge (Mode A/C)               | qwen-3-235b on Cerebras      | production dependency |
| Oracle (LongMemEval grading)   | Gemini 2.5 Flash             | EVAL-only; do not replace |

Today's victory
---------------

k=3 -> k=10 episodic on the merken brief_v1 pipeline lifts
seed-robust (3-seed mean) correct-rate from 51.9% to 67.8%
(stdev 1.7pp across seeds 42/43/44). This parity-matches the
historical RAG-k3 pure baseline on a substrate that was previously
rejected at -18.5pp against the same bar.

Locked-in baseline for the next phase: **k=10 episodic, llama-8b
Builder, brief_v1 briefs (marginal but retained) -> 67.8% 3-seed mean
on LongMemEval-s N=30 per seed.**

This is the number a local replacement has to match. Everything from
here is about replacing the external calls without losing that
baseline.

Phase 1 -- close today (2026-04-24)
-----------------------------------

Status: done.

- 3-seed k=10 llama-8b confirmed 67.8% mean.
- Ablation shows briefs are marginal (+3.3pp for 98% of Cerebras burn).
- Production default: drop briefs, keep k=10 episodic.
- `--dump-briefs-to` flag added to pipeline_runner so any future run
  that does enable briefs also collects teacher-signal as a byproduct.
- PR #36 updated with today's commits and artifacts.

Phase 2 -- define the local target (next session)
-------------------------------------------------

Three strategic choices to make before any training starts:

1. **Single model vs stack.**
   - Option A: one unified 1-3B model fine-tuned multi-task
     (should_remember, brief-synth, Builder, Judge).
   - Option B: a stack of 2-3 specialized small models, each
     targeting one primitive.
   - Trade-off: A is simpler to deploy / harder ML; B is more
     engineering / each piece tractable. Pick based on deployment
     constraints (single-process loop vs modular pipeline).

2. **Base model size.** Updated with user's available budget:
   $900 GCP credit = ~240+ hours A100 80GB spot. That puts 7-13B
   models in comfortable range.
   - 0.5B (fast, fragile on OOD given v7/v8 history). Rejected unless
     there's a strong speed argument.
   - 1-3B (MPS-trainable; reasonable capacity; works for inference
     locally at ~25-50 tok/s on M-series). Good if deployment target
     is strictly "runs on Jay's laptop".
   - **7B (Llama-3 8B, Mistral 7B, Qwen-2.5 7B)** (recommended).
     Training fits a single A100 80GB with LoRA; inference on
     M-series via MLX quant runs at 15-30 tok/s; quality close to
     today's Cerebras llama3.1-8b Builder. Most defensible match to
     the 67.8% baseline.
   - 13B: bigger training spend, inference slower on M-series. Only
     if 7B gates fail.

3. **Training strategy.**
   - Distill from Cerebras (teacher-student). Preserves today's
     winning behavior. Cheap to generate data in large volume.
   - LoRA on open-source instruct-tuned base. Fast to train, validated
     on MPS (v5-ft bge) and standard for A100 workloads.
   - From-scratch (nanoGPT-style). Rejected: history shows OOD
     collapse at small scale.

Recommended default (subject to discussion): **LoRA on Llama-3 8B
or Qwen-2.5 7B as a unified multi-task model**, distilling from
Cerebras teachers, training on GCP A100 80GB, inference via MLX
quant on M-series. One base, multi-task LoRA (or per-task adapters
if multi-task proves unstable).

Budget arithmetic:
- LoRA on 7B / A100 80GB / 3 epochs / 30k rows: ~4-6 hours training.
  At $3.67/hr spot = $14-22 per training run.
- Expect 3-5 iterations during Phase 4 gating: $50-100 total.
- Leaves $780+ headroom for longer runs, bigger models if needed,
  or evaluation compute.

Phase 3 -- data collection (blocked on Phase 2)
-----------------------------------------------

Training dataset shape depends on the architecture picked in Phase 2.

If we go unified multi-task, we need roughly:
- 5-10k (question, k=10_context, correct_answer) for Builder behavior.
- 5-20k (session_turns, briefs) for brief synth (if briefs stay).
- 1-5k (question, context, draft, judge_verified_answer) for Judge.
- Existing v7 merken_labels + LongMemEval per-turn labels for
  should_remember.

Total: ~20k-50k rows to hit reasonable capability with LoRA.

Cost estimate with gpt-oss-120b as teacher (cheaper than qwen-235b):
- Builder traces: ~$5 for 5k qids across seeds.
- Brief dataset: ~$8 for 19k briefs (via `--dump-briefs-to` flag
  already in place).
- Judge traces: ~$3 for 3k Judge-verified answers.
- **Total ~$15-20 one-shot budget** for a first training dataset.

Important: this replaces the need to burn Cerebras per-experiment
with a SINGLE up-front spend that produces a reusable dataset.

Phase 4 -- training + validation (GCP A100)
-------------------------------------------

GCP A100 80GB spot = $3.67/hr. $900 credit = 240+ hours runway.
LoRA on 7B model, ~30k rows, 3 epochs fits in 4-6 hours / $14-22
per iteration. Gates mirror today's seed-robust pattern:

- Gate 1: match or exceed k=10 llama-8b Cerebras on LongMemEval
  N=30 seed-robust (> 63% for 3-seed mean).
- Gate 2: match the 4 decision primitives' current behavior on the
  `experiments/loop_quality/` scenarios (no regression below current
  pass rates).
- Gate 3: BEIR no-regression guard (the v5-ft precedent: R@5 delta
  >= -1.0pp vs baseline).

Fail criterion: any gate regressing significantly -> iterate on
training data or arch, do not ship.

Phase 5 -- production integration (after Phase 4 gates)
-------------------------------------------------------

- Add the local Builder as a config option in pipeline_runner
  alongside the Cerebras Builder.
- Shadow-deploy in the merken Python SDK (default still vstash +
  Cerebras; opt-in local via env var).
- After 1-2 weeks of shadow signal, flip default if no regressions.

Constraints and invariants
--------------------------

- vstash remains the only substrate. No new storage.
- No bespoke compression dialect (CONSTITUTION §8).
- Every new local model passes the `experiments/loop_quality/`
  safety net before default-flipping.
- Fail-closed: if local model emits garbage, fall back to Cerebras
  until the issue is understood.
- Measure with real tokenizer / real answers, never mocked.

What's NOT on this roadmap
--------------------------

- A fifth decision primitive.
- Embedding-based consolidation as a retrieval improvement (proven
  structurally limited, `experiments/consolidation/RESULTS.md`).
- A knowledge graph.
- Web UI, shell completions.
- Swapping encoder to v5-lora in production -- the calibration-
  mismatch investigation already proved that encoder ABI includes
  score distribution; drop-ins unsafe.

Decisions due next session
--------------------------

1. Phase 2 architecture pick:
   - Single-model multi-task vs stack of specialists.
   - Recommended: Llama-3 8B or Qwen-2.5 7B + LoRA multi-task.
2. Data-collection shape:
   - Confirm whether briefs stay in pipeline (marginal retrieval
     value; independent data-collection still useful if brief-synth
     is one of the local-replacement tasks).
   - Budget: Cerebras credit remaining (~$10) + option to top up
     if needed; GCP covers the training itself.
3. GCP environment setup:
   - Spin up an A100 80GB spot VM template.
   - Install stack (torch + transformers + peft + trl + mlx for
     inference validation).
   - Test end-to-end LoRA save/load flow on a trivial model before
     burning hours.
4. Gating experiments for Phase 4:
   - Which LongMemEval subsets for gates.
   - Which loop_quality scenarios get run as regression guard.
   - Oracle budget for gating (Gemini 2.5 Flash; ~30 calls per gate
     x 3 seeds x 3 gates = ~270 calls, trivial).
