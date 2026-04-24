"""Step 1 of the v5-ft bge contrastive fine-tune (notes/bge-small-lme-ft-plan.md).

LoRA contrastive fine-tune of `BAAI/bge-small-en-v1.5` on LongMemEval
positives using `sentence-transformers` + `peft`.

Input: `data/bge_lme_ft/train.jsonl` (emitted by build_contrastive_pairs.py).
Each line: {qid, question, positive, hard_negatives[]}.

Loss: MultipleNegativesRankingLoss (InfoNCE variant). For each anchor
question, the positive is matched against (positive + hard_negatives +
all other anchors' positives in the batch as in-batch easies). This is
the standard contrastive recipe and is what bge-small was pretrained
with, so the objective matches the base model's prior.

Output:
  - LoRA adapter weights under `experiments/retrieval/bge_lme_ft/adapters/v5-lora/`
  - `meta.json` with run config + val loss/recall + base model id
  - `st_base/` snapshot of the SentenceTransformer module layout so a
    loader can reconstitute (base + adapter) without hardcoding the
    modules.json path.

Requires (not in current env):
  pip install peft

Run recipe (MPS on Mac may need the fallback env):
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 experiments/retrieval/bge_lme_ft/train_lora.py

Loader recipe (see load_v5_lora.py in this directory).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

BASE_MODEL = "BAAI/bge-small-en-v1.5"


def _load_triples(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _split(
    triples: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by qid, not by triple, so train/val share no questions."""
    rng = random.Random(seed)
    qids = sorted({t["qid"] for t in triples})
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_frac))
    val_qids = set(qids[:n_val])
    train = [t for t in triples if t["qid"] not in val_qids]
    val = [t for t in triples if t["qid"] in val_qids]
    return train, val


def _to_examples(triples: list[dict]) -> list[InputExample]:
    """InputExample with multiple texts: [anchor, positive, neg1, neg2, ...].

    sentence-transformers' MultipleNegativesRankingLoss treats the first
    text as the anchor, the second as the positive, and any remaining as
    hard negatives for THAT anchor (they become additional negatives in
    the softmax denominator along with in-batch positives of other
    anchors).
    """
    return [
        InputExample(texts=[t["question"], t["positive"], *t["hard_negatives"]])
        for t in triples
    ]


def _apply_lora(model: SentenceTransformer, rank: int, alpha: int, dropout: float) -> None:
    """Wrap the underlying HF transformer in a peft LoRA adapter in-place.

    Targets attention projections only (query/key/value). `dense` is NOT
    in the target list because in BertModel it matches the attention
    output dense, both FFN dense layers per block, AND pooler.dense --
    that blows up trainable params far past the "lightweight LoRA" budget
    we planned for. Start with q/k/v; if undertrained, bump to include
    FFN explicitly and re-plan param budget.
    """
    from peft import LoraConfig, get_peft_model

    # In sentence-transformers 5.x, `Transformer.auto_model` is a read-only
    # property that returns `self.model`. We must assign to `.model`, not
    # `.auto_model`, or the LoRA wrap silently lives in a parallel child
    # module and never sees forward passes.
    transformer = model[0]
    hf_model = transformer.model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    transformer.model = get_peft_model(hf_model, cfg)
    transformer.model.print_trainable_parameters()


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/bge_lme_ft/train.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/retrieval/bge_lme_ft/adapters/v5-lora"),
    )
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of qids held out for validation loss.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="Turn length cap (tokens). bge default is 512.")
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device or _pick_device()
    print(f"[device] {device}", flush=True)

    triples = _load_triples(args.train)
    print(f"[load] {len(triples)} triples from {args.train}", flush=True)

    train_triples, val_triples = _split(triples, args.val_frac, args.seed)
    print(f"[split] train={len(train_triples)} val={len(val_triples)} "
          f"(by-qid, val_frac={args.val_frac})", flush=True)

    model = SentenceTransformer(args.base, device=device)
    model.max_seq_length = args.max_seq_length
    print(f"[model] {args.base} max_seq_length={model.max_seq_length}", flush=True)

    _apply_lora(model, args.rank, args.alpha, args.dropout)

    train_examples = _to_examples(train_triples)
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    print(f"[steps] total={total_steps} warmup={warmup_steps}", flush=True)

    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    wall_s = time.perf_counter() - t0
    print(f"[train] {wall_s:.1f}s wall", flush=True)

    # Save artifacts FIRST. If any downstream metric crashes, the
    # weights still land on disk and can be evaluated directly via
    # eval_lme_retrieval.py.
    args.out.mkdir(parents=True, exist_ok=True)
    _save_artifacts(model, args.base, args.out)

    # Validation. Recall@1 on (q, pos, negs) tuples is the direct
    # self-auditing signal: for each val triple, embed q + its own
    # (pos + negs), check whether pos is the top-scored candidate.
    # Gate 2 (LongMemEval R@5) is the real test.
    val_r_at_1 = _compute_val_recall_at_1(model, val_triples, device)
    print(f"[val] recall@1={val_r_at_1:.4f} on {len(val_triples)} triples",
          flush=True)

    import peft as _peft
    import transformers as _tfm
    import sentence_transformers as _st
    import hashlib
    with args.train.open("rb") as f:
        train_sha = hashlib.sha256(f.read()).hexdigest()[:16]

    meta = {
        "base_model": args.base,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": ["query", "key", "value"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "warmup_frac": args.warmup_frac,
        "n_train": len(train_triples),
        "n_val": len(val_triples),
        "val_recall_at_1": float(val_r_at_1),
        "wall_s": float(wall_s),
        "device": device,
        "torch_version": torch.__version__,
        "peft_version": _peft.__version__,
        "transformers_version": _tfm.__version__,
        "sentence_transformers_version": _st.__version__,
        "train_sha256_prefix": train_sha,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[save] artifacts -> {args.out}", flush=True)
    return 0


def _compute_val_recall_at_1(
    model: SentenceTransformer,
    val: list[dict],
    device: str,
) -> float:
    """For each (q, pos, negs): is sim(q, pos) > sim(q, neg_i) for all i?

    Direct measurement, not batch-sensitive -- safe to compare across
    runs and to use as an early-stopping signal if we add one.
    """
    if not val:
        return float("nan")
    import torch.nn.functional as F

    model.eval()
    correct = 0
    with torch.no_grad():
        for t in val:
            texts = [t["question"], t["positive"], *t["hard_negatives"]]
            emb = model.encode(texts, convert_to_tensor=True, device=device,
                               show_progress_bar=False, normalize_embeddings=True)
            q = emb[0]
            cands = emb[1:]
            sims = F.cosine_similarity(q.unsqueeze(0), cands, dim=1)
            if int(torch.argmax(sims).item()) == 0:
                correct += 1
    model.train()
    return correct / len(val)


def _save_artifacts(model: SentenceTransformer, base: str, out: Path) -> None:
    """Save LoRA adapter + loader contract.

    Writes:
      - `out/adapter/` -- peft adapter weights (includes adapter_config.json)
      - `out/load_hint.json` -- base model id (canonical, not HF cache path)
        so downstream loaders don't depend on adapter_config's cached path.

    We do NOT save the full SentenceTransformer module layout because
    the pooling config is fixed by the base model and any loader can
    reconstruct it by instantiating `SentenceTransformer(base)` fresh.
    Saving it risks drift between adapter config and base config.
    """
    transformer = model[0]
    peft_model = transformer.model
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    (out / "load_hint.json").write_text(json.dumps({
        "base_model": base,
        "adapter_subdir": "adapter",
        "loader": "experiments/retrieval/bge_lme_ft/load_v5_lora.py",
        "max_seq_length": int(model.max_seq_length),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
