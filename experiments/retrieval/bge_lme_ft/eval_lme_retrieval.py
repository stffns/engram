"""Gate 2 runner: LongMemEval R@k for zero-shot bge vs v5-lora.

Standalone comparative eval. Bypasses vstash's model-name-only encoder
hook and uses SentenceTransformer directly so we can feed in the
LoRA-adapted encoder via `load_v5_lora`.

For each question:
  - Build the candidate pool by concatenating each haystack session into
    a single string (session-level retrieval). This matches the
    answer-session-recall framing used by the existing pipeline runner.
  - Embed the question + all candidates with the chosen encoder.
  - Top-k by cosine. Hit if any top-k candidate's session_id is in
    answer_session_ids.

Decontamination: the standard seeds 42/43/44 N=30 split applies. This
script samples one of those three seeds (default seed=44, the one
already reported in the pipeline runs).

Output: single JSON with per-q hits + aggregate R@1 / R@5 / R@10.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402


def _session_text(turns) -> str:
    return "\n".join(f"{t.role}: {t.content}" for t in turns)


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(encoder: str, device: str):
    from sentence_transformers import SentenceTransformer

    if encoder == "bge":
        return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    if encoder.startswith("v5-lora:"):
        adapter_out = encoder.split(":", 1)[1]
        from experiments.retrieval.bge_lme_ft.load_v5_lora import load_v5_lora
        return load_v5_lora(adapter_out, device=device)
    raise ValueError(f"unknown encoder {encoder!r}. use 'bge' or 'v5-lora:<path>'")


def _encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs


def _evaluate(model, convs, k_list: list[int], batch_size: int) -> dict:
    per_q = []
    t0 = time.perf_counter()
    for i, c in enumerate(convs):
        sids = list(c.haystack_sessions.keys())
        candidates = [_session_text(c.haystack_sessions[s]) for s in sids]
        q_vec = _encode(model, [c.question], batch_size=1)[0]
        d_vecs = _encode(model, candidates, batch_size=batch_size)
        sims = d_vecs @ q_vec
        order = np.argsort(-sims)
        answer_set = set(c.answer_session_ids)
        hits_at = {}
        for k in k_list:
            top_sids = [sids[j] for j in order[:k]]
            hits_at[k] = any(s in answer_set for s in top_sids)
        per_q.append({
            "qid": c.question_id,
            "n_sessions": len(sids),
            "hits": hits_at,
            "top5": [sids[j] for j in order[:5]],
        })
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(convs)}] elapsed={time.perf_counter()-t0:.1f}s",
                  flush=True)
    dt = time.perf_counter() - t0

    agg = {k: sum(q["hits"][k] for q in per_q) / len(per_q) for k in k_list}
    return {
        "n_questions": len(per_q),
        "recall_at": {str(k): float(v) for k, v in agg.items()},
        "wall_s": float(dt),
        "per_q": per_q,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        help="'bge' for zero-shot or 'v5-lora:<adapter_out_dir>'")
    parser.add_argument("--seed", type=int, default=44,
                        help="Which held-out seed to evaluate (42, 43, or 44).")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of held-out questions (default 30 matches "
                             "existing pipeline runs).")
    parser.add_argument("--k", type=str, default="1,5,10",
                        help="Comma-separated k values to evaluate.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for candidate encoding. MPS cliff "
                             "observed at bs=16 during training but eval is "
                             "forward-only, so 16 here is usually fine.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.seed not in (42, 43, 44):
        raise SystemExit(f"seed must be in the held-out set {{42,43,44}}; "
                         f"got {args.seed}")

    k_list = [int(x) for x in args.k.split(",")]
    device = _pick_device(args.device)
    print(f"[device] {device}", flush=True)

    convs = load_longmemeval("longmemeval_s", download=True)
    print(f"[load] {len(convs)} total questions", flush=True)

    rnd = random.Random(args.seed)
    subset = rnd.sample(convs, args.n)
    print(f"[subset] seed={args.seed} n={len(subset)}", flush=True)

    print(f"[model] {args.encoder}", flush=True)
    model = _load_model(args.encoder, device)
    model.eval()
    print(f"[model] max_seq_length={model.max_seq_length}", flush=True)

    result = _evaluate(model, subset, k_list, args.batch_size)
    result["encoder"] = args.encoder
    result["seed"] = args.seed
    result["device"] = device

    for k in k_list:
        print(f"  R@{k} = {result['recall_at'][str(k)]:.4f}", flush=True)
    print(f"  wall = {result['wall_s']:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[save] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
