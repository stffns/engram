# Changelog

All notable changes to merken are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/), loosely,
with empirical findings kept alongside code changes to reflect the
project's benchmark-driven development model.

## [Unreleased]

### Added
- `pipeline_runner.py --dump-briefs-to <path>`: append every synthesized
  brief alongside the session-turn input the teacher saw, with
  provenance (`run_id`, `teacher_model`, `teacher_temperature`,
  `teacher_max_completion_tokens`). Enables expensive Cerebras brief
  runs to double as training-data collection for a future local
  brief-synth student. Code-reviewed pre-merge.
- `run_pipeline_higher_k.py`: wrapper that monkey-patches
  `RAG_TOP_K_EPISODIC`, `RAG_TOP_K_BRIEFS`, `BUILDER` model, and a
  `--skip-briefs` switch so ablations do not require editing
  `pipeline_runner.py`.
- `experiments/retrieval/bge_lme_ft/`: LoRA contrastive fine-tune of
  `BAAI/bge-small-en-v1.5` on LongMemEval -- build / train / eval /
  shim / loader plus three-seed BEIR-mini and LongMemEval evaluation
  JSONs. Adapter weights gitignored as regeneratable from the scripts.
- `experiments/retrieval/longmemeval/judge_post_hoc.py` and
  `claim_extract_post_hoc.py`: two builder-bottleneck post-hoc
  approaches tested this cycle; both produced net +2 on seed=44 and
  were subsumed by the k=10 retrieval-depth finding.
- `experiments/retrieval/longmemeval/dump_failure_analysis.py`: per-qid
  forensics dumper used to generate `notes/2026-04-24-failing-qid-
  forensics.md`.
- `notes/2026-04-23-retrieval-not-bottleneck.md`: seed document that
  started the retrieval-saturation investigation and grew into a day
  of experiments.
- `notes/2026-04-24-failing-qid-forensics.md`: per-qid breakdown of
  the 13 seed=44 baseline failures, a fix taxonomy
  (enumeration / ordering / literal / buried-but-present), and a
  shape-aware pipeline standardization proposal.
- `notes/2026-04-24-roadmap-local-merken.md`: five-phase plan from
  today's Cerebras-dependent baseline (67.8% 3-seed) to a 100% local
  memory loop. Target architecture is a 7B-class model (Llama-3 8B or
  Qwen-2.5 7B) fine-tuned with LoRA, distilled from Cerebras teachers,
  training on GCP A100 80GB (user's $900 credit is 240+ hours runway).

### Changed
- `.gitignore` now excludes `experiments/retrieval/bge_lme_ft/adapters/`,
  `experiments/nanogpt/*.log`, `experiments/nanogpt/v8c_frozen_encoder/`,
  and `experiments/retrieval/longmemeval/pipeline_runs/` (all
  regeneratable from the tracked sources).

### Empirical findings (2026-04-24 EOD+1)

- **vstash.ask default prompt is Pareto-optimal for trust-first RAG.**
  Same Cerebras llama-8b + top-k = 10 as pipeline_runner, scored
  17/30 = 56.7% correct with trust_score +50.0% on seed=44 N=30 --
  the highest trust_score measured across every configuration tested
  this cycle. Pipeline_runner with briefs scored 66.7%/+46.7%
  (different Pareto point, same substrate). See
  `notes/2026-04-24-vstash-ask-prompt-experiment.md`.

- **Prohibitive prompt rules backfire at 8B Builder scale.** A
  hybrid variant layering "respond EXACTLY" + "do not substitute"
  clauses onto extraction rules dropped correct_rate to 33.3%
  and trust to +23.3% with 7 supports regressing to neutral.
  Stripping decorative rules from the default produced identical
  verdicts. Prompt lever ceiling reached at 8B; past this point,
  gains require either a larger Builder or targeted fine-tuning
  on the specific reasoning shapes.

- **Retrieval saturation re-confirmed via `miss_analysis`.** vstash's
  built-in miss_analysis tool shows 13/13 of the default-failing
  qids have the answer-session chunk within top-10. Zero retrieval-
  config wins remain on LongMemEval-s at this k.

- **Four stuck-reasoning-shape taxonomy for future training.** The
  qids that neither prompt nor retrieval could unlock group into
  four behaviorally distinct classes: implicit inference across two
  claims (`a96c20ee`), relative-date comparison on fuzzy temporal
  expressions (`gpt4_213fd887`), identity resolution across pronoun
  shifts (`gpt4_0a05b494`), and strict-category counting
  (`ef66a6e5`). These are training targets, not prompt targets.

### Added

- `experiments/retrieval/longmemeval/run_vstash_ask.py` -- benchmark
  wrapper around `vstash.Memory.ask()` with `--prompt-variant
  default|hybrid` flag and runtime inference-backend override.
- `experiments/retrieval/longmemeval/vstash_miss_diagnostic.py` --
  per-qid retrieval depth diagnostic using `vstash.Memory.search`
  to check answer-session chunk rank at top_k = 30.
- `experiments/phase2_training/fuse_to_mlx.py` -- merges a PEFT
  LoRA adapter into its base HF model and converts to MLX 4-bit
  via `mlx_lm.convert` for local inference on M-series.
- `experiments/phase2_training/data/train_positives_only.jsonl` --
  299-row subset of the 414-row teacher dataset, dropping the 115
  hedge-substituted contradicts rows that caused the first SmolLM3
  LoRA to collapse into 80% neutrals.
- `notes/2026-04-24-vstash-ask-prompt-experiment.md` -- full
  write-up of the prompt-lever experiment.
- `experiments/phase2_training/` directory (LoRA training pipeline,
  Colab-ready, Apache 2.0 base models only).

### Changed

- `papers/dual-channel-memory.md` addendum gets a "The prompt-layer
  ceiling (trust vs correctness)" subsection with the empirical
  numbers and the normative observation that substrate-default
  prompts are deliberately-chosen Pareto points.
- `experiments/phase2_training/train_ministral_colab.ipynb` base
  model swapped to `HuggingFaceTB/SmolLM3-3B` (Apache 2.0,
  text-only, standard transformers `AutoModelForCausalLM`) after
  Ministral 3 3B/8B/14B were all confirmed multimodal
  (`Mistral3ForConditionalGeneration`) and blocked LoRA training
  via the standard path.
- `.gitignore` excludes trained LoRA adapters, merged HF dirs,
  and MLX-quantized fused model outputs under
  `experiments/phase2_training/` (all regeneratable from the
  Colab notebook + fuse script).

### Empirical findings (2026-04-24)

- **k=10 retrieval depth dominates on LongMemEval pipeline.**
  `RAG_TOP_K_EPISODIC = 3 -> 10` lifts merken brief_v1 pipeline from
  14/27 = 51.9% to 20/30 = 66.7% (+14.8pp, +7 gains / -2 losses) on
  seed=44 N=30. Three-seed replication:

  | seed | correct |
  |------|---------|
  | 42   | 21/30 = 70.0% |
  | 43   | 20/30 = 66.7% |
  | 44   | 20/30 = 66.7% |
  | mean | 67.8% (stdev 1.7pp) |

  Parity-matches the historical RAG-k3 clean baseline on a substrate
  previously rejected at -18.5pp.

- **Briefs are marginal on LongMemEval (specific-recall benchmark).**
  Ablation with `--skip-briefs` on seed=44 N=30 gives 19/30 = 63.3%
  (delta -3.3pp vs 20/30 with briefs). Cost/benefit is ~100x cost
  per 3pp, so briefs cease to be the default in the pipeline. This
  does not contradict the brief_v1 paper's trajectory-reasoning
  result -- LongMemEval is a specific-recall benchmark and the brief
  mechanism helps trajectory reasoning (see paper Section 8-9).

- **Post-hoc LLM approaches subsumed by k=10.** Judge post-hoc and
  claim-extraction post-hoc each produced net +2 on seed=44 N=30.
  k=10 produced net +5 at zero additional LLM cost.

- **Bigger Builder is not better on this task.** Swapping the Builder
  from `llama3.1-8b` (Cerebras) to `gpt-oss-120b` with k=10 produced
  18/30 = 60.0% vs 20/30 = 66.7%: the 120B model recovers the two
  precision-at-volume losses of 8B but introduces five new refusal-
  prone losses. Empirical winner for this benchmark is
  `llama3.1-8b + k=10`.

- **Encoder ABI includes score distribution.** v5-lora (LoRA fine-tune
  of bge-small on LongMemEval) regresses Mode C by -13.3pp on default
  thresholds; two-knob recalibration recovers only half. Dropping a
  better encoder into a threshold-calibrated consumer is unsafe.

- **Failure classes on LongMemEval (13-qid forensics).** 8 of 13
  fixable by one of: enumeration discipline, explicit date ordering,
  literal-term matching, or larger k. 5 remain structurally hard
  because the answer fact is not retrievable from the RAG pool at any
  k (arithmetic over dates not in context, aggregation over prices
  scattered across turns, true retrieval misses). These are the
  atomic-fact-layer territory described in the roadmap.

## [0.1.0] - 2026-04 (initial release)

### Added
- Four decision primitives: `should_remember`, `should_consolidate`,
  `should_recall`, `should_forget`, each with a default implementation
  and at least one alternative.
- Three deployment surfaces: Python SDK (`from merken import Memory`),
  CLI (`merken`), and MCP server (`merken-mcp`).
- Claude Code hook integration (`SessionStart`, `PreCompact`,
  `UserPromptSubmit`).
- Five loop-quality scenarios covering synthetic controls and real
  organic content.
- LongMemEval Phase A benchmark runner; n=500 gives R@5 = 0.964.
- `brief_v1` consolidation mechanism -- temporal briefs with typed
  schemas (DECISION, ENTITY, EVENT, FREE), stored in a dedicated layer.
  Validated at 86-100% accuracy on knowledge-update scenarios with
  50-1100 events. Paper draft: `papers/dual-channel-memory.md`.
- nanoGPT v7 write-filter graduated to SHADOW on the merken domain.
