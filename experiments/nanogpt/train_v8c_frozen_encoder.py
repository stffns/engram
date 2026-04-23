"""v8c write filter: frozen vstash embedder + MLP classification head.

Architectural pivot from v8a/v8b (nanoGPT from-scratch). Next-token-
prediction loss over ``<event>text<label>LABEL<end>`` spends 99% of
gradient on text prediction and 1% on the label token. On a uniform-
conversational corpus (LongMemEval personal) that's not enough
discriminative signal; v8a collapsed to "always NOISE", v8b balanced
1:5 still shows heavy overfit before the label feature is reliably
learned (2026-04-24 session: val loss plateau ~2.51 with P(DEC) ~ 0.05
across both classes).

v8c leverages the existing vstash embedder (bge-small-en-v1.5, 384d)
as a frozen encoder, then trains a small MLP head with binary cross-
entropy on the explicit DECISION/NOISE label. Every gradient step is
100% about the classification task. Starting from pre-trained
embeddings bypasses "learn English from scratch" that was eating v8's
training budget.

Inputs: same `data/merken_bpe_v8_longmemeval/{train,val}.jsonl`
produced by `build_longmemeval_train.py` (balanced 1:5 for train,
honest for val).

Outputs:
- `experiments/nanogpt/v8c_frozen_encoder/head.pt` (MLP weights)
- `experiments/nanogpt/v8c_frozen_encoder/meta.json` (config +
  embedding model id)

Training budget: ~1 min CPU for the MLP head (embeddings cached once).
Embedding pass for the ~5k train examples: ~2-5 min one-time.

Validation: a sibling `FrozenEncoderWriteDecider` in merken is
scaffolded here but not yet in production path; add it once v8c
graduates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGRAM))


def _load_jsonl(path: Path) -> list[tuple[str, int]]:
    """Return [(text, label_int)] with DECISION=1 / NOISE=0."""
    rows = []
    for line in path.open():
        r = json.loads(line)
        label = 1 if r.get("label") == "DECISION" else 0
        text = (r.get("text") or "").strip()
        if text:
            rows.append((text, label))
    return rows


def _embed_batch(texts: list[str], model_name: str, chunk: int = 256) -> "list":
    """Use vstash's embedder. Chunk inputs so MLX/Metal doesn't attempt
    a single monolithic allocation (~34 GB for 45k bge-small embeddings
    triggered OOM at 14.3 GB Metal buffer limit, 2026-04-24).

    Default chunk=256 is conservative; bge-small-en-v1.5 is small so
    chunk=512+ would also fit, but 256 costs nothing and stays well
    below the cap.
    """
    from vstash.embed import embed_texts
    out: list = []
    for i in range(0, len(texts), chunk):
        batch = texts[i : i + chunk]
        out.extend(embed_texts(batch, model_name=model_name, backend="auto"))
    return out


def main() -> int:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/merken_bpe_v8_longmemeval/train.jsonl"),
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("data/merken_bpe_v8_longmemeval/val.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/nanogpt/v8c_frozen_encoder"),
    )
    parser.add_argument(
        "--encoder", default="BAAI/bge-small-en-v1.5",
        help="vstash embedding model id. Default matches production.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cache-embeddings", action="store_true",
        help=(
            "Cache {train,val}_embeddings.npy in --out-dir so a re-run "
            "skips the embedding pass. First run must omit this flag."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] train={args.train}  val={args.val}", flush=True)
    train_rows = _load_jsonl(args.train)
    val_rows = _load_jsonl(args.val)
    print(
        f"[data] train={len(train_rows)}  "
        f"DEC={sum(y for _, y in train_rows)}  "
        f"NOI={sum(1 - y for _, y in train_rows)}",
        flush=True,
    )
    print(
        f"[data] val={len(val_rows)}  "
        f"DEC={sum(y for _, y in val_rows)}  "
        f"NOI={sum(1 - y for _, y in val_rows)}",
        flush=True,
    )

    cache_train = args.out_dir / "train_embeddings.npy"
    cache_val = args.out_dir / "val_embeddings.npy"
    cache_train_y = args.out_dir / "train_y.npy"
    cache_val_y = args.out_dir / "val_y.npy"

    if args.cache_embeddings and cache_train.exists() and cache_val.exists():
        print(f"[cache] loading {cache_train} / {cache_val}", flush=True)
        X_train = np.load(cache_train)
        y_train = np.load(cache_train_y)
        X_val = np.load(cache_val)
        y_val = np.load(cache_val_y)
    else:
        print(f"[embed] encoder={args.encoder}  (~{len(train_rows) + len(val_rows)} texts)", flush=True)
        t0 = time.perf_counter()
        train_texts = [t for t, _ in train_rows]
        val_texts = [t for t, _ in val_rows]
        train_embs = _embed_batch(train_texts, args.encoder)
        val_embs = _embed_batch(val_texts, args.encoder)
        X_train = np.array(train_embs, dtype=np.float32)
        X_val = np.array(val_embs, dtype=np.float32)
        y_train = np.array([y for _, y in train_rows], dtype=np.int64)
        y_val = np.array([y for _, y in val_rows], dtype=np.int64)
        dt = time.perf_counter() - t0
        print(f"[embed] done in {dt:.1f}s  dim={X_train.shape[1]}", flush=True)

        np.save(cache_train, X_train)
        np.save(cache_val, X_val)
        np.save(cache_train_y, y_train)
        np.save(cache_val_y, y_val)
        print(f"[cache] saved embeddings to {args.out_dir}", flush=True)

    dim = X_train.shape[1]

    # Simple 2-layer MLP: dim -> hidden -> 2
    class Head(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 2),
            )

        def forward(self, x):
            return self.net(x)

    model = Head(dim, args.hidden_dim)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                for p, y in zip(pred.tolist(), yb.tolist()):
                    if y == 1 and p == 1:
                        tp += 1
                    elif y == 1 and p == 0:
                        fn += 1
                    elif y == 0 and p == 1:
                        fp += 1
                    else:
                        tn += 1
        acc = correct / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return acc, precision, recall, f1, (tp, fp, fn, tn)

    print(
        f"[train] epochs={args.epochs} lr={args.lr} batch={args.batch_size} "
        f"hidden={args.hidden_dim} dropout=0.1",
        flush=True,
    )
    best_recall = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * yb.size(0)
            total_n += yb.size(0)
        train_loss = total_loss / max(1, total_n)
        acc, pr, rec, f1, cm = evaluate()
        print(
            f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_acc={acc*100:.1f}%  prec={pr*100:.1f}%  "
            f"rec={rec*100:.1f}%  f1={f1*100:.1f}%  cm(tp,fp,fn,tn)={cm}",
            flush=True,
        )
        if rec > best_recall:
            best_recall = rec
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save best
    if best_state is not None:
        torch.save(best_state, args.out_dir / "head.pt")
    meta = {
        "encoder": args.encoder,
        "dim": dim,
        "hidden_dim": args.hidden_dim,
        "best_recall": best_recall,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"\n[out] head -> {args.out_dir / 'head.pt'}\n"
        f"[out] meta -> {args.out_dir / 'meta.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
