"""Fuse the trained LoRA adapter into the base SmolLM3-3B and convert
to MLX 4-bit for local inference on M-series.

Pipeline:
1. Load base SmolLM3-3B + adapter via peft.AutoPeftModelForCausalLM.
2. merge_and_unload() → full merged HF model.
3. Save merged model to local dir.
4. mlx_lm.convert --quantize --q-bits 4 → MLX 4-bit at final path.

Output: `experiments/phase2_training/smollm3_ft_mlx_q4/` (MLX format,
4-bit quant, ready for `mlx_lm.load()`).

Usage:
    python3 experiments/phase2_training/fuse_to_mlx.py \\
      --adapter experiments/phase2_training/adapter \\
      --out experiments/phase2_training/smollm3_ft_mlx_q4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, type=Path,
                    help="Path to the PEFT adapter directory.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output MLX model dir.")
    ap.add_argument("--merged-dir", type=Path, default=None,
                    help="Intermediate merged HF dir. Defaults to "
                         "<out>_merged_hf.")
    ap.add_argument("--q-bits", type=int, default=4)
    args = ap.parse_args()

    merged_dir = args.merged_dir or Path(str(args.out) + "_merged_hf")
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fuse] adapter={args.adapter}", flush=True)
    print(f"[fuse] merged HF dir={merged_dir}", flush=True)
    print(f"[fuse] MLX out={args.out}", flush=True)

    # --- Step 1: merge adapter into base, save HF format ----------------
    print("[fuse] loading peft model + merging...", flush=True)
    t0 = time.perf_counter()
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(args.adapter),
        dtype=torch.bfloat16,
        device_map="cpu",   # merge on CPU is fine for 3B; avoids MPS OOM
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)

    # Tokenizer: copy from the adapter dir (tokenizer files saved during
    # training), else pull from the HF base.
    try:
        tok = AutoTokenizer.from_pretrained(str(args.adapter))
    except Exception:
        # Fallback to base model id recorded in the adapter's metadata.
        import json
        meta = json.load((args.adapter / "training_metadata.json").open())
        tok = AutoTokenizer.from_pretrained(meta["base_model"])
    tok.save_pretrained(str(merged_dir))
    print(f"[fuse] merged + tokenizer saved in {time.perf_counter()-t0:.1f}s",
          flush=True)

    # --- Step 2: mlx_lm convert + quantize ------------------------------
    print(f"[fuse] converting to MLX q{args.q_bits}...", flush=True)
    t0 = time.perf_counter()
    if args.out.exists():
        shutil.rmtree(args.out)
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(merged_dir),
        "--mlx-path", str(args.out),
        "--quantize",
        "--q-bits", str(args.q_bits),
    ]
    print(f"[fuse] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[fuse] MLX convert FAILED:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print(f"[fuse] MLX convert done in {time.perf_counter()-t0:.1f}s",
          flush=True)
    print(f"[fuse] -> {args.out}")
    print("\n[fuse] disk usage:")
    subprocess.run(["du", "-sh", str(merged_dir), str(args.out)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
