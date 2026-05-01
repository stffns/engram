"""Phase 2 distillation -- Step 4c: prepare mlx-lm LoRA training.

Splits the Step 4b training jsonl into train/valid/test partitions
in the directory layout that ``mlx_lm.lora`` expects::

    <out-dir>/
        train.jsonl
        valid.jsonl
        test.jsonl

Default split is 90/5/5 by row, deterministic shuffle. Each row
keeps the {qid, messages, meta} shape from step4b -- mlx-lm reads
``messages`` natively for chat-template-aware models.

Once the split is on disk, the smoke training run is::

    mlx_lm.lora \\
        --model <path-to-qwen3.5-9b-mlx-base> \\
        --train \\
        --data experiments/phase2_distillation/sft_corpus_v1/mlx_split \\
        --fine-tune-type lora \\
        --num-layers 16 \\
        --batch-size 1 \\
        --iters 1000 \\
        --learning-rate 1e-4 \\
        --max-seq-length 8192 \\
        --grad-checkpoint \\
        --mask-prompt \\
        --adapter-path experiments/phase2_distillation/adapters/smoke_v1 \\
        --steps-per-report 25 \\
        --steps-per-eval 100

Notes on knobs:
  - ``--num-layers 16`` matches the original phase2 plan rank=16.
  - ``--batch-size 1`` because each example has ~7K tokens of
    (system + user + assistant). Apple Silicon unified memory
    cannot fit larger micro-batches on a 9B at this seq length;
    use ``--grad-accumulation-steps`` if effective bs > 1 is
    needed.
  - ``--mask-prompt`` makes the loss only count assistant tokens;
    this is the right behavior for SFT distillation (we don't want
    to penalize the student for not memorizing the chunks).
  - ``--max-seq-length 8192`` covers the canonical ~7K example.
    Bump if a future ETL grows the prompt.
  - ``--iters 1000`` with batch_size=1 ~= 1 epoch on 1K examples.
    Check the actual via the eval loss curve at steps-per-eval.

The base model path: this script does NOT auto-detect the local
mlx-lm cache. Pass it explicitly when invoking mlx_lm.lora -- on
LM Studio installs the path is typically
``~/.cache/lm-studio/models/<org>/<model>`` or ``~/.lmstudio/models/...``.
For the canonical mlx-community Qwen3.5-9B (q8 / bf16 -- not the
TQ3 variant tested in step3), use the HF id directly and let
mlx-lm download::

    --model mlx-community/Qwen3.5-9B-Instruct-bf16

(downloads ~17 GB on first use; subsequent runs use the cached copy.)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _split(rows: list[dict], frac_valid: float, frac_test: float, seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    if frac_valid < 0 or frac_test < 0:
        raise ValueError(f"split fractions must be >= 0, got {frac_valid}, {frac_test}")
    if frac_valid + frac_test >= 1.0:
        raise ValueError(
            f"frac_valid + frac_test = {frac_valid + frac_test:.3f} leaves "
            "no room for training rows; both must sum to < 1.0"
        )
    # Respect 0-valued fractions explicitly. The previous max(1, ...)
    # forced at least one row even when the caller wanted to skip
    # the split entirely.
    n_valid = int(round(n * frac_valid)) if frac_valid > 0 else 0
    n_test = int(round(n * frac_test)) if frac_test > 0 else 0
    n_train = n - n_valid - n_test
    if n_train <= 0:
        raise ValueError(
            f"split would leave train empty: n={n} valid={n_valid} test={n_test}"
        )
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_valid],
        shuffled[n_train + n_valid :],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                    help="Input train.jsonl from step4b.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory; will receive train/valid/test.jsonl")
    ap.add_argument("--frac-valid", type=float, default=0.05)
    ap.add_argument("--frac-test", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.in_path.exists():
        sys.exit(f"missing input: {args.in_path}")

    rows = []
    with args.in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  skipping malformed row: {exc}")

    train, valid, test = _split(rows, args.frac_valid, args.frac_test, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in part:
                # mlx-lm expects "messages" for chat-template
                # training. Drop the meta/qid wrapper -- they don't
                # affect training but inflate row size.
                f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
        print(f"  wrote {len(part):4d} rows -> {path}")

    summary = {
        "n_rows_input": len(rows),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))
    print()
    print("# Suggested mlx_lm.lora command:")
    print(
        "mlx_lm.lora "
        "--model mlx-community/Qwen3.5-9B-Instruct-bf16 "
        "--train "
        f"--data {args.out_dir} "
        "--fine-tune-type lora "
        "--num-layers 16 "
        "--batch-size 1 "
        "--iters 1000 "
        "--learning-rate 1e-4 "
        "--max-seq-length 8192 "
        "--grad-checkpoint "
        "--mask-prompt "
        f"--adapter-path {args.out_dir.parent}/adapter_smoke_v1 "
        "--steps-per-report 25 "
        "--steps-per-eval 100"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
