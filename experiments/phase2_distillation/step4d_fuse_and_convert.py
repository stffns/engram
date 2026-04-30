"""Phase 2 distillation -- Step 4d: fuse LoRA + convert to MLX.

Takes the peft adapter saved by ``train_qwen_lora.py`` (or the
Colab notebook), merges it into the Qwen base via
``peft.merge_and_unload``, saves the merged HF model, then runs
``mlx_lm.convert`` to produce an MLX q8 (or q4) build that LM
Studio can load on Apple Silicon.

The intermediate merged HF model is ~14 GB on disk (BF16 weights).
The final MLX q8 is ~8 GB. Both are kept by default so the user
can switch between them; pass ``--cleanup`` to delete the merged
HF after MLX conversion.

Usage::

    python -m experiments.phase2_distillation.step4d_fuse_and_convert \\
        --adapter experiments/phase2_distillation/sft_corpus_v1/adapter_smoke_v1/adapter_final \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --merged-path ~/models/Qwen2.5-7B-Instruct-merken-smoke-v1 \\
        --mlx-path ~/models/Qwen2.5-7B-Instruct-merken-smoke-v1-mlx-q8 \\
        --q-bits 8

Requirements: ``peft``, ``transformers``, ``mlx_lm`` available
in the active environment. CPU-only merge works fine; GPU is not
required for this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Path to the unzipped adapter_final directory.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF id of the base model. Must match the model "
                         "the adapter was trained on (see the adapter's "
                         "training_metadata.json).")
    ap.add_argument("--merged-path", type=Path, required=True,
                    help="Where to save the merged HF model.")
    ap.add_argument("--mlx-path", type=Path, required=True,
                    help="Where to save the MLX-converted model.")
    ap.add_argument("--q-bits", type=int, default=8, choices=[4, 8],
                    help="MLX quantization bits. 8 keeps more precision; "
                         "4 halves disk + memory at ~1-2pp accuracy cost.")
    ap.add_argument("--cleanup", action="store_true",
                    help="Delete the merged HF after successful MLX convert.")
    args = ap.parse_args()

    args.merged_path = args.merged_path.expanduser().resolve()
    args.mlx_path = args.mlx_path.expanduser().resolve()
    args.adapter = args.adapter.expanduser().resolve()
    if not args.adapter.exists():
        sys.exit(f"adapter path does not exist: {args.adapter}")

    # Sanity-check the adapter metadata to catch base-model mismatches.
    meta_path = args.adapter / "training_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        recorded_base = meta.get("base_model")
        if recorded_base and recorded_base != args.base_model:
            print(
                f"[fuse] WARNING: adapter was trained on "
                f"{recorded_base!r} but --base-model is "
                f"{args.base_model!r}. Override only if you know "
                f"the architectures are compatible.",
                flush=True,
            )

    # Step 1: merge adapter into base.
    if args.merged_path.exists() and any(args.merged_path.iterdir()):
        print(f"[fuse] merged path already populated, skipping merge: "
              f"{args.merged_path}", flush=True)
    else:
        print(f"[fuse] loading base: {args.base_model}", flush=True)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # merge is CPU-friendly; saves GPU RAM.
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        print(f"[fuse] base loaded in {time.perf_counter()-t0:.1f}s", flush=True)

        print(f"[fuse] applying adapter: {args.adapter}", flush=True)
        t0 = time.perf_counter()
        peft_model = PeftModel.from_pretrained(base, str(args.adapter))
        merged = peft_model.merge_and_unload()
        print(f"[fuse] merged in {time.perf_counter()-t0:.1f}s", flush=True)

        args.merged_path.mkdir(parents=True, exist_ok=True)
        print(f"[fuse] saving merged HF -> {args.merged_path}", flush=True)
        t0 = time.perf_counter()
        merged.save_pretrained(str(args.merged_path), safe_serialization=True)
        tokenizer.save_pretrained(str(args.merged_path))
        print(f"[fuse] saved in {time.perf_counter()-t0:.1f}s", flush=True)

        # Free RAM before MLX convert spins up its own copy.
        del merged, peft_model, base
        import gc
        gc.collect()

    # Step 2: convert to MLX.
    if args.mlx_path.exists() and any(args.mlx_path.iterdir()):
        print(f"[mlx] mlx path already populated, skipping convert: "
              f"{args.mlx_path}", flush=True)
    else:
        from mlx_lm import convert as mlx_convert

        args.mlx_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[mlx] convert -> {args.mlx_path} (q{args.q_bits})", flush=True)
        t0 = time.perf_counter()
        mlx_convert.convert(
            hf_path=str(args.merged_path),
            mlx_path=str(args.mlx_path),
            quantize=True,
            q_bits=args.q_bits,
            q_group_size=64,
        )
        print(f"[mlx] converted in {time.perf_counter()-t0:.1f}s", flush=True)

    # Step 3: optional cleanup of merged HF.
    if args.cleanup and args.merged_path.exists():
        print(f"[cleanup] removing merged HF: {args.merged_path}", flush=True)
        shutil.rmtree(args.merged_path)

    # Sidecar metadata so future runs can trace this MLX model back
    # to the adapter / commit it came from.
    sidecar = args.mlx_path / "merken_provenance.json"
    sidecar.write_text(json.dumps({
        "adapter_path": str(args.adapter),
        "base_model": args.base_model,
        "merged_path_kept": not args.cleanup,
        "merged_path": str(args.merged_path) if not args.cleanup else None,
        "q_bits": args.q_bits,
        "git_sha": _git_sha(),
    }, indent=2), encoding="utf-8")
    print(f"[done] MLX model at {args.mlx_path}", flush=True)
    print(f"[done] sidecar at {sidecar}", flush=True)
    print()
    print("# To load in LM Studio:")
    print(f"#   1. Copy or symlink {args.mlx_path} into ~/.lmstudio/models/")
    print("#   2. In LM Studio -> Model: select the new model")
    print("#   3. Set Context Length >= 8192")
    print("#   4. Re-run step3_zero_shot_smoke.py against it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
