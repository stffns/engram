# v5-ft: contrastive fine-tune of bge-small-en-v1.5 on LongMemEval

Plan: `notes/bge-small-lme-ft-plan.md`. Full training artifacts
gitignored under `data/bge_lme_ft/` (1128 triples) and
`experiments/retrieval/bge_lme_ft/adapters/` (adapter weights).

## Summary

| gate   | metric                            | threshold  | observed     | verdict    |
|--------|-----------------------------------|------------|-------------:|------------|
| Gate 1 | BEIR-mini R@5 delta vs bge        | >= -1.0pp  | -0.97pp      | PASS       |
| Gate 2 | LongMemEval 3-seed R@1 delta      | positive   | +10.0pp      | PASS       |
| Gate 2 | LongMemEval 3-seed R@5 delta      | +5-10pp    | +3.3pp       | WEAK       |
| Gate 3 | RAG-k3 N=30 seed=44 supports      | improve    | flat 60%     | NULL       |
| Gate 3b| Mode C H18 N=30 seed=44 supports  | improve    | 36.7% (-13.3pp) | REGRESSION |

**Net: retrieval gains confirmed, but don't translate to end-to-end
answer accuracy at RAG-k3 (ceiling effect) and actively HURT Mode C
(threshold-calibration mismatch). Do not ship as default; do not drop
into other pipelines without recalibration.**

## Training (selected variant `adapters/v5-lora-lr1e4`)

- Base: `BAAI/bge-small-en-v1.5`
- LoRA: `target_modules=[query, key, value]`, rank=8, alpha=16, dropout=0.05
- Data: 1128 contrastive triples (1 pos + 4 hard negatives each) from
  414 LongMemEval holdout questions, gpt-oss-120b turn-level labels
  (1128 "yes" labels)
- Optimizer: AdamW lr=1e-4, 3 epochs, bs=8, warmup=10%
- Wall: 238s on MPS (bs=16 triggered a ~400x slowdown on MPS -- use bs=8)
- Trainable params: 221,184 (0.66% of 33.5M base)
- val recall@1 on held-out qids: 0.7121

## LoRA rate search

Four configs tried. lr=1e-4 with 3 epochs is the sweet spot:

| config            | BEIR R@5 delta | LME R@1 3-seed | LME R@5 3-seed |
|-------------------|---------------:|---------------:|---------------:|
| bge (zero-shot)   |          -     |         67.8%  |         92.2%  |
| e1 (1 ep, 2e-4)   |         -0.42  |         70.0%  |         93.3%  |
| e2 (2 ep, 2e-4)   |         -1.51  |         76.7%  |         95.6%  |
| e3 (3 ep, 2e-4)   |         -3.10  |         77.8%  |         94.4%  |
| **lr1e4 (3 ep)**  |       **-0.97**|       **76.7%**|       **95.6%**|

## Gate 1: BEIR no-regression

| dataset  | metric  | bge     | v5-lora-lr1e4 | delta  |
|----------|---------|--------:|--------------:|-------:|
| scifact  | NDCG@10 | 0.7215  | 0.7019        | -1.96  |
| scifact  | R@5     | 0.7753  | 0.7553        | -2.00  |
| nfcorpus | NDCG@10 | 0.3403  | 0.3411        | +0.08  |
| nfcorpus | R@5     | 0.1280  | 0.1288        | +0.08  |
| **mean** | R@5     | 0.4517  | 0.4420        | **-0.97** |

BEIR cache reused from `../vex/experiments/data/beir_*`.

## Gate 2: LongMemEval session-level R@k (3-seed, N=30)

| seed | R@1 bge | R@1 v5-lora | R@5 bge | R@5 v5-lora | R@10 bge | R@10 v5-lora |
|------|--------:|------------:|--------:|------------:|---------:|-------------:|
| 42   |  66.7%  |     73.3%   |  96.7%  |     96.7%   |   96.7%  |      96.7%   |
| 43   |  70.0%  |     83.3%   |  93.3%  |     96.7%   |   96.7%  |      96.7%   |
| 44   |  66.7%  |     73.3%   |  86.7%  |     90.0%   |   96.7%  |      96.7%   |
| mean |  67.8%  |     76.7%   |  92.2%  |     94.4%   |   96.7%  |      96.7%   |

Gains across all 3 seeds on R@1 and R@5 (no seed=44 artifact).

## Gate 3: RAG-k3 end-to-end (seed=44 N=30)

Oracle-graded by Gemini. Builder = Cerebras (same prompt as RAG-k3 baseline).

| condition        | supports | contradicts | neutral | rate  |
|------------------|---------:|------------:|--------:|------:|
| RAG-k3 + bge     |       18 |           8 |       4 | 60.0% |
| RAG-k3 + v5-lora |       18 |           7 |       5 | 60.0% |

Per-question diff: 29/30 identical verdicts. Only qid `370a8ff4`
flipped: bge=contradicts -> v5-lora=neutral. Same binary accuracy.

## Gate 3b: Mode C end-to-end (H18 preface, seed=44 N=30)

| condition        | supports | contradicts | neutral | rate  |
|------------------|---------:|------------:|--------:|------:|
| Mode C + bge     |       15 |           4 |      11 | 50.0% |
| Mode C + v5-lora |       11 |           7 |      12 | 36.7% |

**v5-lora regressed Mode C by -13.3pp.** Root cause (post-mortem):
Mode C has calibrated score thresholds (absolute 0.0161 cutoff +
distance filters) tuned for bge-small's cosine distribution.
v5-lora produces different cosines for the same queries/chunks;
many answer-bearing chunks fall below Mode C's vstash threshold,
triggering the FTS-only fallback path (`vec_w=0.00` observed on
most queries after the first vector probe). FTS-only retrieval is
weaker than hybrid on this benchmark.

Implication: encoder-side gains are not free to swap across
pipelines -- each downstream consumer with calibrated thresholds
needs a matched sweep. Ship v5-lora only where retrieval metrics
alone matter (not answer accuracy), or not at all until calibration
is done per consumer.

## Artifacts

- `build_contrastive_pairs.py` -- Step 0: emit training triples
- `train_lora.py` -- Step 1: LoRA fine-tune
- `load_v5_lora.py` -- loader helper for adapter+base composition
- `beir_mini_check.py` -- Gate 1 runner
- `eval_lme_retrieval.py` -- Gate 2 runner (session-level R@k)
- `vstash_v5lora_shim.py` -- monkey-patch bridge for Gate 3
- `run_pipeline_v5lora.py` -- wrapper to run pipeline_runner.py or
  mode_a_eval.py with v5-lora encoder via the shim
- `adapters/v5-lora-lr1e4/` -- shipped adapter weights (base_model:
  BAAI/bge-small-en-v1.5, rank=8)

## What this DOESN'T show

- v5-lora is untested against Mode C / KV-splice (different retrieval
  consumer, likely benefits from top-1 improvements).
- Higher-k RAG (k=5, k=10) may show different deltas than k=3.
- Builder-side failure analysis (why does the pipeline lose 40% of
  questions even with perfect context?) is open.

## What's next

See `notes/bge-small-lme-ft-plan.md` §risk-list. Suggested (not run):

- **Judge post-hoc** over RAG-k3 output: might recover some of the
  12 non-support questions by reranking or reasoning over top-3.
- **Structured claim extraction** (H-F from memory) over top-3: may
  help builder reason about what it retrieved.
- vstash per-collection encoder support: unblocks shipping v5-lora
  opt-in without the monkey-patch shim.
