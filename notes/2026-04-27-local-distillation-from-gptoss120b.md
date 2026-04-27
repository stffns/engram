# Local distillation from gpt-oss-120b — next-session plan

**Status:** plan only. No code changes from this note. Captured at
end of 2026-04-27 session 2 after the gpt-oss-120b 3-seed Builder
result + cost analysis landed (commit `eecf041`).

## Why now

The original "Phase 2 / local-merken" plan (notes/cerebras-writer-loop-plan.md,
notes/2026-04-24-roadmap-local-merken.md) was: use Cerebras llama3.1-8b
as a teacher, distill into a small local model via LoRA. Teacher
ceiling at the time: 56.3% LoCoMo / 56.7% LME.

Today's findings change the picture without invalidating the goal:

- Builder swap to **gpt-oss-120b** lifts the teacher ceiling to
  **70.7% LoCoMo (+14.4pp 3-seed)** and **67.8% LME (+6.7pp 3-seed)**.
- Cost analysis (vstash full top-k=8 prompt path): **per-question
  total tokens 1.02x** between models. Generating a 50K-example
  training corpus with the new teacher costs roughly $145-290
  depending on Cerebras tier.

Net: the local-distillation path is *more* viable than under the
old teacher, not less. But the architecture has to change.

## What changes vs the old plan

| piece | original plan | revised |
|---|---|---|
| teacher          | `cerebras/llama3.1-8b` @ 56.3% | **`cerebras/gpt-oss-120b` @ 70.7%** |
| student size     | `mistralai/Ministral-3-3B-Instruct-2512` | **7-8B class** (Llama-3 8B / Qwen-2.5 7B) |
| training shape   | Q → A direct                   | **Q → reasoning_trace → A** (gpt-oss is a reasoning model) |
| target           | "match teacher"                | "close half the teacher↔baseline gap" — concretely **≥65% LoCoMo** |
| eval anchors     | one (teacher)                  | three: zero-shot baseline (floor) / student (measured) / teacher (ceiling) |

The old branch name `feature/phase2-ministral-lora` is misleading
under the revised plan. Mistral 3 3B is sub-dimensioned for reasoning
distillation from a 120B-class MoE teacher. R1-Distill literature
suggests 7-8B is the minimum-viable student for retaining reasoning
fidelity.

## Why a reasoning student matters

gpt-oss-120b on Cerebras returns `message.reasoning` (hidden CoT)
separate from `message.content` (visible answer). For Q&A workloads
on this benchmark, `reasoning_tokens=0` in our cost sample — the
model didn't seem to use the channel — but on harder shapes
(temporal-reasoning, multi-hop) the reasoning channel does get
populated. The runner currently throws it away.

If we distill Q → A only, the student misses the reasoning signal
that produced the answer. R1-Distill / DeepSeek-R1 papers consistently
show that reasoning-trace-supervised distillation outperforms
direct-answer distillation by a wide margin on downstream reasoning
tasks. We should capture and train on the trace.

## Honest risks

1. **Reasoning distillation is not solved.** Llama-3 8B may capture
   70-90% of teacher reasoning quality, not 100%. Realistic target:
   meet the *baseline+gap/2* line, i.e. ~65% LoCoMo / ~64% LME.
2. **MoE teacher → dense student capacity gap.** gpt-oss-120b is
   ~5-10B active params per token in MoE form. A dense 7-8B has
   smaller effective capacity for reasoning storage. The student
   may saturate below the teacher even with infinite training data.
3. **Teacher failure modes transfer.** LME knowledge-update -11pp
   3-seed regression in gpt-oss-120b will copy into the student.
   Distillation does not fix the teacher's blind spots.
4. **Cost scales with eval iteration count.** ~$2-7 per full 3-seed
   eval; 10 iterations of train-eval = $20-70 in eval alone, on top
   of the $145-290 for the training corpus. Total program cost
   roughly $200-400.

## Step-by-step plan

### Step 1 — capture reasoning channel (~30 min)

Today's runners discard `message.reasoning`. Patch
`vstash.chat.ask` (or write a thin wrapper that bypasses it) to:
- Capture `resp.choices[0].message.reasoning` as a separate field.
- Store in artifact rows as `reasoning_trace` alongside `vstash_answer`.
- Keep capture optional (CLI flag `--capture-reasoning`) to avoid
  bloating non-distillation runs.

Out of scope today; first move next session.

### Step 2 — qualitative inspection (1 hour)

Generate N=200 samples (Q → reasoning_trace → A) with gpt-oss-120b
across LoCoMo + LME. Cost ~$1-2.

Read 30-50 traces by hand. Questions to answer:
- Are the traces coherent and on-topic?
- On temporal-reasoning shapes, does the trace explicitly do the
  date math or shortcut to an answer?
- On knowledge-update shapes (the regression), does the trace flag
  the contradiction-then-refuse pattern? (Hypothesis: yes.)

If traces are generic ("The answer is X. Let me verify..."), the
distillation signal is weak and we'd need to prompt the teacher
harder for explicit CoT. If traces are detailed and structured,
proceed to Step 3.

### Step 3 — target architecture decision (30 min)

Pick one of:
- **Llama-3 8B Instruct** — broad ecosystem, well-studied LoRA
  recipes, moderate reasoning capacity baseline.
- **Qwen-2.5 7B Instruct** — newer, stronger reasoning baseline,
  some Apache-2.0 licensing wrinkles to verify.
- **Llama-3.1 8B** if Llama-3 base is older.

Decide based on (a) zero-shot LoCoMo/LME baseline of each on the
canonical Phase 2 stack, and (b) reasoning-trace following ability.
Spend 30 min on a smoke eval of all three at zero-shot before
committing to one.

### Step 4 — smoke distillation (1 day)

LoRA r=16, 1K examples, 1 epoch. Goal: confirm the pipeline works
end-to-end. Eval gate: student post-LoRA improves ≥5pp over student
zero-shot on LoCoMo seed=44 N=201. If not, debug pipeline before
scaling.

Existing scaffolding: `experiments/phase2_training/train_ministral_lora.py`
+ `gcp_bootstrap.sh` + `build_training_data.py`. Repurpose; don't
write from scratch.

### Step 5 — full pipeline (2-3 days incl. GCP A100 wall time)

LoRA r=32, 50K examples (mix LoCoMo + LME + LongMemEval long-tail
synthetic) sourced from the patched gpt-oss-120b teacher. 3 epochs,
bs=8, gradient accumulation 2, lr=1e-4 (grid first if smoke result
suggests tuning is needed).

Train on the GCP A100 80GB. Wall time: 4-8 hours per epoch at
~50K examples; ~12-24 hours total. $900 GCP credit covers this
many times over.

### Step 6 — eval gate (~30 min compute)

Run student through:
- **LoCoMo 3-seed canonical stack** (per-session, vec=0.5/0.5,
  rerank, --n-per-cat 40, Cerebras oracle).
- **LME 3-seed canonical stack** (per-turn, default weights,
  no-rerank, Cerebras oracle).

Gate criteria (to declare distillation a success):
- LoCoMo 3-seed mean ≥ **65%** (clears the baseline+stdev band of
  56.3+3.4=59.7%, captures ≥half the teacher gap).
- LME 3-seed mean ≥ **64%** (similarly captures ≥half the +6.7pp).
- Knowledge-update LME does NOT regress beyond -15pp (teacher's
  failure mode should transfer cleanly, not amplify).
- Per-shape stdev ≤ teacher's. (Student should be at least as
  seed-stable as the teacher.)

Below those gates → diagnose, possibly grow data or step the LoRA
rank up. Above → declare local-merken viable; cost falls from
~$1/Q on Cerebras to ~$0/Q on local hardware.

## What this plan is NOT

- Not a Mistral-3-3B story. Re-target the branch name on first
  commit of Step 3.
- Not a "100% local" promise — student inference still needs a
  local GPU at runtime; for personal use that's fine, for cloud
  serving it's a separate engineering question.
- Not gated on Gemini quota. Cerebras-only path; no oracle bias
  if we use Cerebras-graded eval throughout.

## Files to read on resume

- `notes/cerebras-writer-loop-plan.md` — original concept, still
  intellectually relevant.
- `notes/2026-04-24-roadmap-local-merken.md` — old 5-phase plan,
  useful for the GCP setup section.
- `experiments/phase2_training/train_ministral_lora.py` — existing
  LoRA scaffolding to repurpose.
- `experiments/retrieval/oracle_health.py` — defensive guard from
  this session; the distillation eval pipeline must use it.
- `experiments/retrieval/locomo/RESULTS_phase2.md` and
  `experiments/retrieval/longmemeval/RESULTS.md` — the teacher
  numbers the student is chasing.

## Branch hygiene

Branch `feature/phase2-ministral-lora` is 3 commits ahead of origin
at session close. The next-session work should either:
- Re-target this branch (rename to `feature/phase2-distillation`)
  and continue, OR
- Merge to develop and start a fresh branch for the new direction.

Decide before Step 1.
