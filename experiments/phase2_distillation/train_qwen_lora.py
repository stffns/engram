"""Phase 2 distillation -- LoRA SFT trainer for Qwen on CUDA.

Reads the chat-template JSONL produced by step4c (mlx_split/) and
fine-tunes a Qwen base model with LoRA on the assistant target.
Adapter weights are saved at the end; the metadata.json captures
the recipe + git sha for provenance.

Designed for an NVIDIA A100 / H100 / RTX 6000 environment; Colab
A100 high-RAM is the canonical target. Apple Silicon (MPS) is
NOT supported here -- use ``mlx_lm.lora`` for that path. See
``step4c_prepare_mlx_train.py`` for the Apple-Silicon path.

The ``{messages}`` row shape (system / user / assistant) is fed to
``tokenizer.apply_chat_template`` so the training tokenization
matches what the model will see at inference. The assistant target
already embeds Qwen3-style ``<think>...</think>`` tags around the
teacher's reasoning trace (see ``step4b_format_sft_corpus.py``);
the trainer treats those tokens as part of the target the student
must learn to produce.

Usage::

    python -m experiments.phase2_distillation.train_qwen_lora \\
        --data experiments/phase2_distillation/sft_corpus_v1/mlx_split \\
        --output-dir experiments/phase2_distillation/sft_corpus_v1/adapter_smoke_v1

    # Or in a Colab cell after `cd /content/merken`:
    !python -m experiments.phase2_distillation.train_qwen_lora \\
        --data experiments/phase2_distillation/sft_corpus_v1/mlx_split \\
        --output-dir /content/adapters/smoke_v1 \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --epochs 1 --batch-size 1 --grad-accum 4 --max-seq-len 8192

    # Resume from a checkpoint if the runtime disconnected:
    !python -m experiments.phase2_distillation.train_qwen_lora \\
        --data experiments/phase2_distillation/sft_corpus_v1/mlx_split \\
        --output-dir /content/adapters/smoke_v1 \\
        --resume-from-checkpoint /content/adapters/smoke_v1/checkpoint-225

Defaults are tuned for the Step 4 smoke (~$1, ~30-60 min on A100).
For the Step 5 full pipeline bump --epochs to 3, --data to a 50K
corpus, and consider --batch-size 2 if memory allows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path,
                    help="Directory with train.jsonl (and optional valid.jsonl). "
                         "Each row: {'messages': [system, user, assistant]}.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF id of the base model. Defaults to "
                         "Qwen/Qwen2.5-7B-Instruct (BF16, ~14 GB on disk).")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where to save adapter + metadata.")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="gradient accumulation steps; effective batch = batch * grad-accum")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Optional checkpoint path or truthy value passed through to "
            "Trainer.train(resume_from_checkpoint=...)."
        ),
    )
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"],
                    help="LoRA target modules. Default matches Qwen attention.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-eval", action="store_true",
                    help="Run mid-training eval on the valid split. "
                         "Disabled by default because eval at this "
                         "seq length OOMs on a 40 GB A100 -- only "
                         "set this on H100 / 80 GB A100.")
    args = ap.parse_args()

    train_path = args.data / "train.jsonl"
    valid_path = args.data / "valid.jsonl"
    if not train_path.exists():
        raise SystemExit(f"missing {train_path}")

    # Heavy imports after CLI parse.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. This trainer targets A100/H100. "
            "For Apple Silicon use mlx_lm.lora -- see step4c output."
        )

    print(f"[train] loading tokenizer: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[train] loading base model: {args.base_model}", flush=True)
    t0 = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"[train] base loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    print(f"[train] loading {train_path}", flush=True)
    train_rows = _load_jsonl(train_path)
    print(f"[train] {len(train_rows)} train rows", flush=True)

    valid_rows: list[dict] = []
    if args.enable_eval and valid_path.exists():
        valid_rows = _load_jsonl(valid_path)
        print(f"[train] {len(valid_rows)} valid rows", flush=True)
    elif valid_path.exists():
        n_valid = sum(1 for _ in valid_path.open())
        print(f"[train] valid.jsonl has {n_valid} rows; "
              f"eval disabled (use --enable-eval to opt in)",
              flush=True)

    def _format(row: dict) -> dict:
        # Apply Qwen's chat template to the {system, user, assistant}
        # message triple. Training tokenization matches inference so
        # the assistant target the student learns is exactly what
        # the model will be asked to emit.
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_rows).map(
        _format, remove_columns=list(train_rows[0].keys())
    )
    eval_ds = None
    if valid_rows:
        eval_ds = Dataset.from_list(valid_rows).map(
            _format, remove_columns=list(valid_rows[0].keys())
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=50 if eval_ds is not None else None,
        # Move eval predictions to CPU per batch so the eval pass
        # does not blow the 40 GB A100 with full-vocab logits in
        # GPU memory; only matters when --enable-eval is set.
        eval_accumulation_steps=1 if eval_ds is not None else None,
        prediction_loss_only=True if eval_ds is not None else False,
        per_device_eval_batch_size=1 if eval_ds is not None else 8,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"[train] starting at {time.strftime('%H:%M:%S')} -- "
          f"effective batch = {args.batch_size * args.grad_accum}, "
          f"epochs = {args.epochs}", flush=True)
    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    dt = time.perf_counter() - t0
    print(f"[train] done in {dt/60:.1f} min", flush=True)

    final_adapter = args.output_dir / "adapter_final"
    model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    metadata = {
        "base_model": args.base_model,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "target_modules": args.target_modules,
        "warmup_ratio": args.warmup_ratio,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "n_train_rows": len(train_rows),
        "n_valid_rows": len(valid_rows),
        "training_wall_s": dt,
        "git_sha": _git_sha(),
        "final_train_loss": float(result.training_loss),
        "log_history": trainer.state.log_history,
    }
    (final_adapter / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"[train] adapter + metadata -> {final_adapter}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
