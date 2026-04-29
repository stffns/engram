"""Probe whether the residual geometric metrics from #38 (PR, EVR_1,
AngDisp, NormVar) carry signal at *production* ingest shape.

#38's NO_GO was reached on three short-event scenarios (mediana 86-97
chars) -- ``knowledge_update_hard``, ``products`` (generated), and
``medical`` (generated). The merken pipeline ingests:

- LoCoMo per-session (mediana 2808 chars, ~30x longer)
- LME oracle per-turn (mediana 482 chars, ~5-7x longer)

This probe reuses the *exact* metric implementation from
``experiments/role_geometry/directional_residual_geometry.py:cluster_metrics``
on three embedding pools at three different shape regimes:

1. ``short_events``: 200 events from #38's ``medical.json`` (the
   distribution that produced the NO_GO).
2. ``lme_turn``: 200 LME oracle haystack turns (production shape A).
3. ``locomo_session``: 200 LoCoMo per-session blobs (production shape B).

For each pool we form K=50 random clusters of size 4 (same n_fixed_4
setup #38 used) and report the distribution of each metric. If the
metric ranges look qualitatively similar across pools, #38's NO_GO
inherits to production. If they shift -- particularly if production
collapses to flat distributions or, conversely, opens up a range that
short-events compressed -- then the NO_GO does not transfer and the
geometric methods deserve a re-eval at production shape.

Run: ``python -m notes.2026-04-28-geometric-shape-probe``
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import numpy as np

from experiments.role_geometry.directional_residual_geometry import cluster_metrics
from vstash.embed import embed_texts

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # same as #38

POOL_SIZE = 200      # texts per pool
N_CLUSTERS = 50      # number of random clusters per pool
CLUSTER_N = 4        # cluster size (matches #38 n_fixed_4)
SEED = 42

SHORT_EVENTS_PATH = Path("experiments/role_geometry/scenarios_disjoint/generated/medical.json")
LOCOMO_PATH = Path("experiments/retrieval/locomo/data/locomo10.json")
LME_PATH = Path("experiments/retrieval/longmemeval/.cache/longmemeval_oracle.json")


def _load_short_events() -> list[str]:
    with SHORT_EVENTS_PATH.open() as f:
        data = json.load(f)
    return [e["text"] for e in data["events"] if isinstance(e, dict)]


def _load_lme_turns() -> list[str]:
    with LME_PATH.open() as f:
        raw = json.load(f)
    out: list[str] = []
    for q in raw:
        for sess in q.get("haystack_sessions") or []:
            if not isinstance(sess, list):
                continue
            for t in sess:
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                role = t.get("role", "?")
                out.append(f"{role}: {content}")
    return out


def _load_locomo_sessions() -> list[str]:
    with LOCOMO_PATH.open() as f:
        raw = json.load(f)
    out: list[str] = []
    for entry in raw:
        conv = entry["conversation"]
        i = 1
        while f"session_{i}" in conv:
            dt = conv.get(f"session_{i}_date_time", "")
            turns = conv[f"session_{i}"]
            lines = [f"{t['speaker']}: {t['text']}" for t in turns]
            text = f"[{dt}]\n" + "\n".join(lines)
            out.append(text)
            i += 1
    return out


def _summarize_pool(name: str, texts: list[str], rng: random.Random) -> dict:
    sample = rng.sample(texts, k=min(POOL_SIZE, len(texts)))
    char_lens = [len(t) for t in sample]
    print(
        f"  {name:18s}: n={len(sample)}  chars min/median/max = "
        f"{min(char_lens)}/{int(statistics.median(char_lens))}/{max(char_lens)}"
    )
    print(f"    embedding {len(sample)} texts...")
    vectors = np.asarray(embed_texts(sample, model_name=EMBEDDING_MODEL), dtype=np.float64)
    if vectors.shape[0] != len(sample):
        raise RuntimeError(f"embed shape mismatch: {vectors.shape}")

    metrics: list[dict[str, float]] = []
    for _ in range(N_CLUSTERS):
        idx = rng.sample(range(len(sample)), k=CLUSTER_N)
        m = cluster_metrics(vectors, idx)
        m["_indices"] = idx
        metrics.append(m)

    return {
        "name": name,
        "sample": sample,
        "vectors": vectors,
        "metrics": metrics,
        "char_lens": char_lens,
    }


def _print_distribution_table(pools: list[dict]) -> None:
    metric_keys = ["pr", "evr_1", "ang_disp", "norm_var", "baseline_cos_mean"]
    print("\n" + "=" * 92)
    print("Metric distributions per pool (n=50 random clusters of 4)")
    print("=" * 92)
    header = (
        f"{'metric':18s} "
        + " ".join(f"{p['name']:>22s}" for p in pools)
    )
    print(header)
    print("-" * len(header))
    for mk in metric_keys:
        cells = []
        for p in pools:
            vals = [m[mk] for m in p["metrics"]]
            cell = (
                f"{statistics.median(vals):.3f} "
                f"[{min(vals):.3f}-{max(vals):.3f}]"
            )
            cells.append(cell)
        print(f"{mk:18s} " + " ".join(f"{c:>22s}" for c in cells))


def _print_representative_cluster(pool: dict, n_show: int = 1) -> None:
    print("\n  Representative cluster (median EVR_1):")
    metrics = pool["metrics"]
    sample = pool["sample"]
    sorted_metrics = sorted(metrics, key=lambda m: m["evr_1"])
    pick = sorted_metrics[len(sorted_metrics) // 2]
    print(
        f"    PR={pick['pr']:.3f}  EVR_1={pick['evr_1']:.3f}  "
        f"AngDisp={pick['ang_disp']:.3f}  NormVar={pick['norm_var']:.4f}  "
        f"baseline_cos_mean={pick['baseline_cos_mean']:.3f}"
    )
    for j, idx in enumerate(pick["_indices"][:n_show + 2]):
        snippet = sample[idx][:140].replace("\n", " | ")
        print(f"      [m{j}] {snippet}")


def main() -> None:
    rng = random.Random(SEED)

    print(f"embedder: {EMBEDDING_MODEL}")
    print(f"cluster size: {CLUSTER_N}, n_clusters per pool: {N_CLUSTERS}, pool size: {POOL_SIZE}")
    print(f"seed: {SEED}\n")

    print("Loading text pools...")
    short_events = _load_short_events()
    lme_turns = _load_lme_turns()
    locomo_sessions = _load_locomo_sessions()
    print(f"  short_events available:    {len(short_events)}")
    print(f"  lme turns available:       {len(lme_turns)}")
    print(f"  locomo sessions available: {len(locomo_sessions)}")
    print()

    print("Building pools (sample, embed, cluster, score)...")
    pools = [
        _summarize_pool("short_events", short_events, rng),
        _summarize_pool("lme_turn", lme_turns, rng),
        _summarize_pool("locomo_session", locomo_sessions, rng),
    ]

    _print_distribution_table(pools)

    for p in pools:
        print(f"\n[ {p['name']} ]")
        _print_representative_cluster(p)


if __name__ == "__main__":
    main()
