"""Gate 1: BEIR-mini no-regression check.

Runs scifact + nfcorpus head-to-head on zero-shot bge vs v5-lora. Both
are small BEIR datasets that bge-small is benchmarked on.

Metrics: NDCG@10 and R@5 (the plan's stated threshold). Gate: R@5 on
BEIR-mini must NOT drop >= 1.0pp vs zero-shot.

Reuses the BEIR cache under `../vex/experiments/data/beir_*` if
available (same content as the public BEIR download). Falls back to
HF-style fetch if not.

Usage:
    python3 experiments/retrieval/bge_lme_ft/beir_mini_check.py \\
        --encoder bge --out .../bge.json
    python3 experiments/retrieval/bge_lme_ft/beir_mini_check.py \\
        --encoder v5-lora:experiments/retrieval/bge_lme_ft/adapters/v5-lora \\
        --out .../v5lora.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))

VEX_BEIR_CACHE = Path.home() / "Desktop" / "Personal" / "Projects" / "vex" / "experiments" / "data"
DATASETS = ["scifact", "nfcorpus"]


def _load_beir(ds: str) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    """Return (corpus, queries, qrels) for a BEIR dataset.

    qrels: {query_id -> {doc_id -> relevance_int}}
    """
    root = VEX_BEIR_CACHE / f"beir_{ds}"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} missing; run vex's beir_benchmark.py to populate cache"
        )
    corpus = []
    with (root / "corpus.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("title") or "") + ". " + (d.get("text") or "")
            corpus.append({"id": d["_id"], "text": text.strip()})
    queries = []
    with (root / "queries.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            queries.append({"id": d["_id"], "text": d["text"]})

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    test_tsv = root / "qrels" / "test.tsv"
    with test_tsv.open() as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels[qid][did] = int(score)
    return corpus, queries, dict(qrels)


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
        path = encoder.split(":", 1)[1]
        from experiments.retrieval.bge_lme_ft.load_v5_lora import load_v5_lora
        return load_v5_lora(path, device=device)
    raise ValueError(f"unknown encoder {encoder!r}")


def _encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _ndcg_at_k(ranked_ids: list[str], rels: dict[str, int], k: int) -> float:
    # Standard BEIR/TREC NDCG with binary-graded relevance.
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k]):
        r = rels.get(did, 0)
        if r > 0:
            dcg += r / math.log2(i + 2)
    ideal_rels = sorted(rels.values(), reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels[:k]) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_ids: list[str], rels: dict[str, int], k: int) -> float:
    total_rel = sum(1 for r in rels.values() if r > 0)
    if total_rel == 0:
        return 0.0
    hit = sum(1 for did in ranked_ids[:k] if rels.get(did, 0) > 0)
    return hit / total_rel


def _eval_one(model, corpus, queries, qrels, batch_size: int) -> dict:
    # Skip queries with no qrels (common in BEIR test splits).
    q_with_rels = [q for q in queries if q["id"] in qrels]
    corpus_ids = [d["id"] for d in corpus]
    corpus_texts = [d["text"] for d in corpus]

    t0 = time.perf_counter()
    d_vecs = _encode(model, corpus_texts, batch_size=batch_size)
    d_wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    q_vecs = _encode(model, [q["text"] for q in q_with_rels], batch_size=batch_size)
    q_wall = time.perf_counter() - t1

    ndcg10s = []
    r5s = []
    for i, q in enumerate(q_with_rels):
        sims = d_vecs @ q_vecs[i]
        order = np.argsort(-sims)
        ranked_ids = [corpus_ids[j] for j in order[:50]]
        ndcg10s.append(_ndcg_at_k(ranked_ids, qrels[q["id"]], 10))
        r5s.append(_recall_at_k(ranked_ids, qrels[q["id"]], 5))

    return {
        "n_queries": len(q_with_rels),
        "n_corpus": len(corpus),
        "ndcg_at_10": float(np.mean(ndcg10s)),
        "recall_at_5": float(np.mean(r5s)),
        "corpus_encode_s": float(d_wall),
        "query_encode_s": float(q_wall),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        help="'bge' or 'v5-lora:<adapter_out_dir>'")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = _pick_device(args.device)
    print(f"[device] {device}", flush=True)
    print(f"[model]  {args.encoder}", flush=True)
    model = _load_model(args.encoder, device)
    model.eval()
    print(f"[model]  max_seq_length={model.max_seq_length}", flush=True)

    out = {"encoder": args.encoder, "device": device, "by_dataset": {}}
    for ds in args.datasets:
        print(f"[ds]     {ds} ...", flush=True)
        corpus, queries, qrels = _load_beir(ds)
        res = _eval_one(model, corpus, queries, qrels, args.batch_size)
        out["by_dataset"][ds] = res
        print(f"  n_corpus={res['n_corpus']} n_queries={res['n_queries']} "
              f"NDCG@10={res['ndcg_at_10']:.4f} R@5={res['recall_at_5']:.4f}",
              flush=True)

    n = len(args.datasets)
    out["mean_ndcg_at_10"] = sum(out["by_dataset"][d]["ndcg_at_10"] for d in args.datasets) / n
    out["mean_recall_at_5"] = sum(out["by_dataset"][d]["recall_at_5"] for d in args.datasets) / n
    print(f"[mean]   NDCG@10={out['mean_ndcg_at_10']:.4f} "
          f"R@5={out['mean_recall_at_5']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[save]   {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
