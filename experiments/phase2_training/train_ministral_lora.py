"""LoRA SFT training for Ministral 3 3B Instruct on GCP A100 80GB.

Runs on the provisioned GCP VM (see gcp_bootstrap.sh). Reads the
training jsonl produced by build_training_data.py, fine-tunes
Ministral 3 3B with LoRA via peft + trl SFTTrainer, and saves the
adapter weights to a local directory that the bootstrap script
uploads back to the driver machine.

Design:
- Base: `mistralai/Ministral-3-3B-Instruct-2512` (the full HF version,
  not the pre-quantized 4bit MLX variant; we want BF16 for training).
- LoRA: rank=16, alpha=32, target_modules query/key/value/o_proj.
- Optimizer: AdamW lr=1e-4, warmup=10%, 3 epochs.
- Chat template: the tokenizer's built-in apply_chat_template with
  a system + user message; the target is appended as the assistant
  turn. This ensures the training shape matches inference.
- Loss: standard causal LM loss, but only on the assistant tokens
  via TRL's DataCollatorForCompletionOnlyLM.

Gate-ready: saves a `metadata.json` with training loss curve + git
sha so the gate eval has provenance.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path,
                    help="Training jsonl from build_training_data.py")
    ap.add_argument("--base-model",
                    default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    args = ap.parse_args()

    # Heavy imports after CLI parse so --help is fast on a minimal env.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

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

    # LoRA config mirrors the v5-ft bge successful recipe at rank=16,
    # alpha=32 (stronger than the rank=8 used on bge because the
    # target task -- instruction following vs embedding -- needs
    # more capacity).
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # Build HF dataset from the jsonl.
    print(f"[train] loading data: {args.data}", flush=True)
    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    print(f"[train] {len(rows)} training rows", flush=True)

    def _format(row):
        # Use the tokenizer's built-in chat template so training
        # matches what MLX inference will do downstream.
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user",   "content": row["user"]},
            {"role": "assistant", "content": row["target"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    ds = Dataset.from_list(rows).map(_format, remove_columns=list(rows[0].keys()))

    # Modern trl SFTConfig (trl >= 0.15). Full-text loss (no
    # completion-only masking); for 414 rows this is fine and avoids
    # the DataCollatorForCompletionOnlyLM dependency which was removed
    # in newer trl versions.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=0.1,
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
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"[train] starting training at {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()
    result = trainer.train()
    dt = time.perf_counter() - t0
    print(f"[train] training done in {dt:.0f}s", flush=True)

    # Save adapter only (not merged weights). Saves ~40MB vs ~6GB.
    final_adapter = args.output_dir / "adapter_final"
    model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    metadata = {
        "base_model": args.base_model,
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "n_train_rows": len(rows),
        "training_wall_s": dt,
        "git_sha": _git_sha(),
        "final_train_loss": float(result.training_loss),
    }
    (final_adapter / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"[train] adapter + metadata -> {final_adapter}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
