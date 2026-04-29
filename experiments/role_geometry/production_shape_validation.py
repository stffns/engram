"""Production-shape validation of #38's residual geometry hypothesis.

#38's NO_GO was reached on three sentence-level scenarios (events with
median 86-97 chars). The 2026-04-28 audit flagged that the follow-up
shape probe used random clusters (no pos/neg labels) and only N=4, so
the "NO_GO survives at production shape" claim was not actually tested.

This script repairs that gap. It builds *labeled* pos/neg clusters
directly at production shape and runs the same `cluster_metrics` and
`bootstrap_auc` from `directional_residual_geometry.py`.

Construction:

- ``locomo_session`` pool: each cluster member is one full per-session
  blob (`[date_time]\\n{speaker}: {text}\\n...`), the unit merken
  ingests for LoCoMo. Pos = N sessions sampled from the SAME conv
  (same persona pair, topically related). Neg = N sessions sampled
  from N DISTINCT convs.
- ``lme_turn`` pool: each cluster member is one LME haystack turn
  (`{role}: {content}`), the unit merken ingests for LME at per-turn
  granularity. Pos = N turns sampled from the SAME haystack session
  (one topic). Neg = N turns sampled from N DISTINCT sessions across
  DISTINCT qids.

Sweep: ``N in {4, 20}`` x ``seed in {42, 43, 44}``. N=4 mirrors #38's
original cluster size and rank-deficient regime (rank <= 3 in d=384).
N=20 tests whether the residual geometry escapes saturation when the
cluster is large enough for the rank to surface.

Metrics: PR, EVR_1, AngDisp, NormVar (vs baseline neg-cos). Verdict
per metric via ``decide_primary``: lower-CI > 0.80 = STRONG, >= 0.65 =
PARTIAL, < 0.65 = NO_GO. Anti-relabeling gate: metric AUC must beat
``baseline_neg_cos`` AUC by >= 0.10.

Run: ``python -m experiments.role_geometry.production_shape_validation``
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.role_geometry.directional_residual_geometry import (
    bootstrap_auc,
    cluster_metrics,
    decide_baseline_beat,
    decide_primary,
)

logger = logging.getLogger("role_geometry_prod_shape")

LOCOMO_PATH = Path("experiments/retrieval/locomo/data/locomo10.json")
LME_PATH = Path("experiments/retrieval/longmemeval/.cache/longmemeval_oracle.json")
# Pool-specific N. LoCoMo has only 10 convs (10 distinct group_ids), so neg
# clusters can only be of size <= 10; we cap at 8 to keep slack. LME has 940
# namespaced sessions, but only 7 of them have >= 20 turns, so N=20 would
# bootstrap pos from 7 templates; N=12 (the median session length) keeps the
# pos pool large enough for an honest test.
DEFAULT_N_BY_POOL = {
    "lme_turn": (4, 12),
    "locomo_session": (4, 8),
}
DEFAULT_K_TARGET = 30  # pos clusters and neg clusters per (pool, N) cell
DEFAULT_SEEDS = (42, 43, 44)
METRIC_NAMES = ("pr", "evr_1", "ang_disp", "norm_var")


@dataclass(frozen=True)
class TextItem:
    text: str
    group_id: str  # conv id (LoCoMo) or session id (LME)
    item_id: str  # session id (LoCoMo) or turn id (LME)


# ----------------------------------------------------------------- pool builders


def build_locomo_session_pool() -> list[TextItem]:
    """One TextItem per LoCoMo session blob. ``group_id`` = conv sample_id."""
    raw = json.loads(LOCOMO_PATH.read_text())
    items: list[TextItem] = []
    for entry in raw:
        sample_id = entry["sample_id"]
        conv = entry["conversation"]
        i = 1
        while f"session_{i}" in conv:
            dt = conv.get(f"session_{i}_date_time", "")
            turns = conv[f"session_{i}"]
            lines = [f"{t['speaker']}: {t['text']}" for t in turns]
            text = f"[{dt}]\n" + "\n".join(lines)
            items.append(
                TextItem(
                    text=text,
                    group_id=str(sample_id),
                    item_id=f"{sample_id}::session_{i}",
                )
            )
            i += 1
    return items


def build_lme_turn_pool() -> list[TextItem]:
    """One TextItem per LME haystack turn. ``group_id`` = haystack session id
    namespaced by qid so that turns from sessions that happen to share an id
    across questions are not collapsed."""
    raw = json.loads(LME_PATH.read_text())
    items: list[TextItem] = []
    for q in raw:
        qid = q.get("question_id") or q.get("id")
        if not qid:
            raise KeyError(
                "LME entry missing both 'question_id' and 'id'; refusing "
                "to silently collapse multiple entries into a single group"
            )
        sessions = q.get("haystack_sessions") or []
        sids = q.get("haystack_session_ids") or [str(i) for i in range(len(sessions))]
        for sid, turns in zip(sids, sessions):
            if not isinstance(turns, list):
                continue
            ns_sid = f"{qid}::{sid}"
            for i, t in enumerate(turns):
                role = t.get("role", "?")
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                text = f"{role}: {content}"
                items.append(
                    TextItem(
                        text=text,
                        group_id=ns_sid,
                        item_id=f"{ns_sid}::{i}",
                    )
                )
    return items


# ----------------------------------------------------------------- cluster builders


def _group_by_group_id(items: list[TextItem]) -> dict[str, list[int]]:
    """Map ``group_id`` -> list of indices into ``items``."""
    out: dict[str, list[int]] = {}
    for idx, it in enumerate(items):
        out.setdefault(it.group_id, []).append(idx)
    return out


def build_clusters_for_n(
    items: list[TextItem],
    *,
    n: int,
    k_target: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[list[int]], dict]:
    """Return (pos_clusters, neg_clusters, diagnostics).

    Pos cluster: N items drawn without replacement from a single group.
    Only groups with >= N items are eligible. Multiple pos clusters may
    be drawn from the same group with independent without-replacement
    samples (the samples can overlap across pos clusters from the same
    group; this is intentional bootstrapping within a topic-coherent
    pool, the same pattern #38 used for ``mixed_topic_decisions``).

    Neg cluster: pick N distinct groups, sample 1 item from each that is
    NOT already in the union of all pos cluster items for this cell. This
    keeps the AUC unconfounded by item-level overlap on small-group pools
    (LoCoMo has 10 distinct convs; without this guard, pos and neg
    cluster items would collide frequently).

    Returns a third element with diagnostics: n_eligible_pos_groups,
    n_distinct_groups, n_pos_unique_items, n_neg_excluded_items.
    """
    by_group = _group_by_group_id(items)
    eligible_pos_groups = [g for g, idxs in by_group.items() if len(idxs) >= n]
    all_groups = list(by_group.keys())

    if len(eligible_pos_groups) == 0:
        raise RuntimeError(
            f"no group has >= {n} items; eligible pos groups = 0"
        )
    if len(all_groups) < n:
        raise RuntimeError(
            f"pool has only {len(all_groups)} distinct groups; cannot build "
            f"a neg cluster of N={n}"
        )

    pos_clusters: list[list[int]] = []
    pos_used: set[int] = set()
    for _ in range(k_target):
        g = eligible_pos_groups[int(rng.integers(0, len(eligible_pos_groups)))]
        pool = by_group[g]
        chosen = rng.choice(len(pool), size=n, replace=False)
        cluster_idxs = [pool[c] for c in chosen]
        pos_clusters.append(cluster_idxs)
        pos_used.update(cluster_idxs)

    # Neg: pick N distinct groups, sample 1 item per group from the
    # group's items minus pos_used. If a chosen group has no remaining
    # items, retry the group draw with a fresh permutation. Worst case:
    # if pos_used has consumed an entire group, we retry until we find N
    # groups that each have >= 1 unused item; with k_target * n=4 << total
    # items per group on both pools, this is bounded.
    neg_clusters: list[list[int]] = []
    max_retries_per_cluster = 50
    for _ in range(k_target):
        for retry in range(max_retries_per_cluster):
            gs = rng.choice(len(all_groups), size=n, replace=False)
            cluster: list[int] = []
            ok = True
            for gi in gs:
                pool = by_group[all_groups[gi]]
                available = [j for j in pool if j not in pos_used]
                if not available:
                    ok = False
                    break
                cluster.append(available[int(rng.integers(0, len(available)))])
            if ok:
                neg_clusters.append(cluster)
                break
        else:
            raise RuntimeError(
                f"could not find a neg cluster disjoint from pos_used after "
                f"{max_retries_per_cluster} retries; pos_used={len(pos_used)} "
                f"of {sum(len(v) for v in by_group.values())} items"
            )

    diagnostics = {
        "n_eligible_pos_groups": len(eligible_pos_groups),
        "n_distinct_groups": len(all_groups),
        "n_pos_unique_items": len(pos_used),
        "n_total_pool_items": sum(len(v) for v in by_group.values()),
    }
    return pos_clusters, neg_clusters, diagnostics


# ----------------------------------------------------------------- embedder


def resolve_embedder() -> str:
    from vstash.config import EmbeddingsConfig
    model = EmbeddingsConfig().model
    if not model:
        raise RuntimeError("vstash EmbeddingsConfig().model empty")
    return model


EMBED_BATCH_SIZE = 256


def embed_pool(items: list[TextItem], model_name: str) -> np.ndarray:
    """Batched wrapper over vstash.embed_texts.

    The vstash MLX path does not auto-batch; passing 10k+ texts in a
    single call OOMs Metal on 24GB unified memory. Batching in chunks of
    ``EMBED_BATCH_SIZE`` produces identical embeddings up to floating
    point and stays under the device buffer limit.
    """
    from vstash.embed import embed_texts
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    for i in range(0, len(items), EMBED_BATCH_SIZE):
        batch = [it.text for it in items[i : i + EMBED_BATCH_SIZE]]
        vecs = embed_texts(batch, model_name=model_name)
        arr = np.asarray(vecs, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != len(batch):
            raise RuntimeError(
                f"embed_texts shape {arr.shape!r} for batch of {len(batch)}"
            )
        chunks.append(arr)
    full = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0))
    if full.shape[0] != len(items):
        raise RuntimeError(
            f"batched embed total {full.shape[0]} != {len(items)} input"
        )
    logger.info(
        "embedded %d items in %.2fs across %d batches, dim=%d, model=%s",
        len(items), time.perf_counter() - t0, len(chunks),
        full.shape[1] if full.size else 0, model_name,
    )
    return full


# ----------------------------------------------------------------- driver


def _seed_for_cell(pool_name: str, n: int, seed: int) -> np.random.Generator:
    """Return an RNG independent across (pool, N, seed) tuples.

    Without this, ``default_rng(seed)`` would produce overlapping draws
    across different N within the same seed (the n=4 cell's first draws
    would coincide with the first draws of the n=12 cell), so a 3-seed
    sweep across 2 N values would not be 6 independent realizations.
    Hashing pool_name + n + seed into the SeedSequence fixes this.
    """
    pool_hash = int.from_bytes(pool_name.encode(), "big", signed=False) & 0xFFFF
    return np.random.default_rng([int(seed), int(n), pool_hash])


def evaluate_cell(
    pool_name: str,
    items: list[TextItem],
    vectors: np.ndarray,
    *,
    n: int,
    k_target: int,
    seed: int,
    bootstrap_resamples: int,
) -> dict:
    """Run a (pool, N, seed) cell. Returns per-metric AUCs + verdict."""
    rng_root = _seed_for_cell(pool_name, n, seed)
    rng_clusters, rng_boot = rng_root.spawn(2)

    pos, neg, diag = build_clusters_for_n(
        items, n=n, k_target=k_target, rng=rng_clusters
    )
    logger.info(
        "[%s N=%d seed=%d] built pos=%d neg=%d clusters; "
        "eligible_pos_groups=%d distinct_groups=%d pos_unique_items=%d/%d",
        pool_name, n, seed, len(pos), len(neg),
        diag["n_eligible_pos_groups"], diag["n_distinct_groups"],
        diag["n_pos_unique_items"], diag["n_total_pool_items"],
    )

    pos_metrics = [cluster_metrics(vectors, idx) for idx in pos]
    neg_metrics = [cluster_metrics(vectors, idx) for idx in neg]
    all_metrics = pos_metrics + neg_metrics
    # label = 1 means "neg" (topic-incoherent), the contamination class.
    labels = np.array([0] * len(pos) + [1] * len(neg))

    cell = {
        "pool": pool_name,
        "n": n,
        "seed": seed,
        "k_target": k_target,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "diagnostics": diag,
        "metrics_aucs": {},
    }

    metric_aucs: dict[str, dict] = {}
    for metric in METRIC_NAMES:
        scores = np.array([m[metric] for m in all_metrics])
        metric_aucs[metric] = bootstrap_auc(
            scores=scores, labels=labels,
            n_resamples=bootstrap_resamples, rng=rng_boot,
        )

    baseline_scores = -np.array([m["baseline_cos_mean"] for m in all_metrics])
    metric_aucs["baseline_neg_cos"] = bootstrap_auc(
        scores=baseline_scores, labels=labels,
        n_resamples=bootstrap_resamples, rng=rng_boot,
    )

    cell["metrics_aucs"] = metric_aucs

    verdicts: dict[str, dict] = {}
    baseline_mean = metric_aucs["baseline_neg_cos"]["roc_auc_mean"]
    for metric in METRIC_NAMES:
        auc = metric_aucs[metric]
        primary = decide_primary(auc["roc_auc_lo"])
        beat = decide_baseline_beat(auc["roc_auc_mean"], baseline_mean)
        verdicts[metric] = {
            "primary": primary,
            "baseline_beat": beat,
            "roc_auc_mean": auc["roc_auc_mean"],
            "roc_auc_lo": auc["roc_auc_lo"],
            "roc_auc_hi": auc["roc_auc_hi"],
        }
    cell["verdicts"] = verdicts
    cell["baseline_neg_cos_mean"] = baseline_mean
    return cell


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pools", nargs="+",
        default=["lme_turn", "locomo_session"],
        choices=["lme_turn", "locomo_session"],
        help="which production pools to run",
    )
    p.add_argument(
        "--n-list", type=int, nargs="+", default=None,
        help=(
            "cluster sizes to sweep, applied uniformly across all pools. "
            "If unset, falls back to pool-specific defaults: "
            f"{DEFAULT_N_BY_POOL}."
        ),
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="seeds to sweep (default 42 43 44)",
    )
    p.add_argument(
        "--k-target", type=int, default=DEFAULT_K_TARGET,
        help="pos clusters and neg clusters per (pool, N, seed) cell (default 30)",
    )
    p.add_argument(
        "--bootstrap-resamples", type=int, default=1000,
        help="bootstrap resamples for AUC CI (default 1000)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="default experiments/role_geometry/runs/prod_shape_<ts>",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("experiments/role_geometry/runs") / f"prod_shape_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name = resolve_embedder()
    logger.info("embedder: %s", model_name)

    pool_specs: dict[str, list[TextItem]] = {}
    if "lme_turn" in args.pools:
        pool_specs["lme_turn"] = build_lme_turn_pool()
    if "locomo_session" in args.pools:
        pool_specs["locomo_session"] = build_locomo_session_pool()

    pool_vectors: dict[str, np.ndarray] = {}
    for name, items in pool_specs.items():
        by_group = _group_by_group_id(items)
        group_sizes = [len(idxs) for idxs in by_group.values()]
        logger.info(
            "[%s] pool size=%d distinct_groups=%d "
            "group sizes min/median/max=%d/%d/%d",
            name, len(items), len(by_group),
            min(group_sizes), int(np.median(group_sizes)), max(group_sizes),
        )
        pool_vectors[name] = embed_pool(items, model_name)

    cells: list[dict] = []
    for pool_name, items in pool_specs.items():
        vectors = pool_vectors[pool_name]
        n_list_for_pool = (
            list(args.n_list) if args.n_list is not None
            else list(DEFAULT_N_BY_POOL[pool_name])
        )
        logger.info("[%s] N sweep: %s", pool_name, n_list_for_pool)
        for n in n_list_for_pool:
            for seed in args.seeds:
                cell = evaluate_cell(
                    pool_name, items, vectors,
                    n=n, k_target=args.k_target, seed=seed,
                    bootstrap_resamples=args.bootstrap_resamples,
                )
                cells.append(cell)
                primary_table = " ".join(
                    f"{m}={cell['verdicts'][m]['roc_auc_mean']:.3f}"
                    f"[{cell['verdicts'][m]['roc_auc_lo']:.3f}-"
                    f"{cell['verdicts'][m]['roc_auc_hi']:.3f}]"
                    for m in METRIC_NAMES
                )
                logger.info(
                    "[%s N=%d seed=%d] %s baseline_neg_cos=%.3f",
                    pool_name, n, seed, primary_table,
                    cell["baseline_neg_cos_mean"],
                )

    aggregated = _aggregate_per_pool_n(cells)
    import vstash
    n_list_for_summary = (
        list(args.n_list) if args.n_list is not None
        else {k: list(v) for k, v in DEFAULT_N_BY_POOL.items() if k in pool_specs}
    )
    summary = {
        "embedding_model": model_name,
        "vstash_version": getattr(vstash, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "pools": list(pool_specs.keys()),
        "n_list": n_list_for_summary,
        "seeds": list(args.seeds),
        "k_target": args.k_target,
        "bootstrap_resamples": args.bootstrap_resamples,
        "pool_sizes": {
            name: {
                "n_items": len(items),
                "n_groups": len(_group_by_group_id(items)),
            }
            for name, items in pool_specs.items()
        },
        "cells": cells,
        "aggregated": aggregated,
    }
    summary_path = args.output_dir / "auc_summary.json"
    summary_path.write_text(json.dumps(_scrub(summary), indent=2, allow_nan=False))
    logger.info("summary -> %s", summary_path)

    print("\n=== production-shape #38 verdict (3-seed mean per metric) ===")
    print(f"output dir : {args.output_dir}")
    print(f"embedder   : {model_name}")
    print()
    for record in aggregated:
        print(
            f"--- pool={record['pool']} N={record['n']} "
            f"(n_pos={record['n_pos']} n_neg={record['n_neg']} per seed, "
            f"k_target={record['k_target']})"
        )
        for metric in METRIC_NAMES:
            row = record["per_metric"][metric]
            print(
                f"  {metric:<10} mean(AUC)={row['mean_auc']:.3f} "
                f"+- {row['stderr_3seed']:.3f}  "
                f"worst_seed_lo={row['worst_seed_lo']:.3f}  "
                f"best_seed_hi={row['best_seed_hi']:.3f}  "
                f"verdicts={row['verdicts']}  beat={row['baseline_beat']}"
            )
        print(f"  baseline   mean(AUC)={record['baseline_mean_auc']:.3f}")
    print("==============================================================")
    return 0


def _aggregate_per_pool_n(cells: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, int], list[dict]] = {}
    for c in cells:
        by_key.setdefault((c["pool"], c["n"]), []).append(c)

    records: list[dict] = []
    for (pool, n), entries in by_key.items():
        per_metric: dict[str, dict] = {}
        n_seeds = len(entries)
        for metric in METRIC_NAMES:
            aucs = [e["verdicts"][metric]["roc_auc_mean"] for e in entries]
            los = [e["verdicts"][metric]["roc_auc_lo"] for e in entries]
            his = [e["verdicts"][metric]["roc_auc_hi"] for e in entries]
            verdicts = [e["verdicts"][metric]["primary"] for e in entries]
            beat = [e["verdicts"][metric]["baseline_beat"] for e in entries]
            stdev = float(np.std(aucs, ddof=1)) if n_seeds > 1 else 0.0
            per_metric[metric] = {
                "mean_auc": float(np.mean(aucs)),
                "stdev_auc": stdev,
                "stderr_3seed": stdev / max(np.sqrt(n_seeds), 1.0),
                "worst_seed_lo": float(np.min(los)),
                "best_seed_hi": float(np.max(his)),
                "per_seed_aucs": aucs,
                "verdicts": verdicts,
                "baseline_beat": beat,
            }
        baseline_aucs = [e["baseline_neg_cos_mean"] for e in entries]
        records.append(
            {
                "pool": pool,
                "n": n,
                "n_pos": entries[0]["n_pos"],
                "n_neg": entries[0]["n_neg"],
                "k_target": entries[0]["k_target"],
                "per_metric": per_metric,
                "baseline_mean_auc": float(np.mean(baseline_aucs)),
            }
        )
    return records


def _scrub(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
