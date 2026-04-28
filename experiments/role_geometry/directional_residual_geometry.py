"""Directional residual geometry as a role-contamination signal.

Tests merken issue #38 hypothesis: when embeddings of a cluster are centered
(residuals = vector - centroid), clusters whose members share the same role
have residuals concentrated in fewer directions than role-mixed clusters.
If the hypothesis holds, the participation ratio of the residual covariance
becomes a free cluster-quality gate -- no training, no labels, just
post-processing of vectors vstash already produces.

Design deviations from issue #38, documented here so the writeup can cite
them:

1. The issue assumes scenarios under ``experiments/scenarios/`` with explicit
   role labels per event. The actual scenarios live in
   ``experiments/loop_quality/scenarios/`` and only have ``{id, topic, text}``.
   Roles are derived deterministically from the id pattern:

     * ``*_v1``      -> "initial"
     * ``*_v2``      -> "refinement"
     * ``*_v3``      -> "reversal"
     * ``noise_*``   -> "noise"

2. Each (topic, fine_role) cell contains exactly 1 event in every available
   scenario, so "clean same-(topic, role)" clusters at N >= 4 are
   structurally impossible. Cluster-level analysis builds clusters
   synthetically from cross-topic same-role samples. Pairwise analysis
   substitutes the empty "same-topic same-fine-role" stratum with a
   "diff-topic same-fine-role" control class for the secondary gate.

3. The mixed_role_in_topic class (the target positive for the contamination
   hypothesis) is forced to N=4 by adding one random noise event to the
   v1+v2+v3 trio of one decision topic. The added noise event is sampled
   uniformly from the noise pool, NOT topic-conditioned, to avoid
   confounding role-mismatch with topic-relevance.

The hypothesis is binary and pre-registered:

  * Cluster-level ROC-AUC of PR (clean vs contaminated), bootstrap 95% CI:

      lower CI > 0.80              -> strong, integration design (#41)
      0.65 <= lower CI <= 0.80     -> partial, combined-gate input (#40)
      lower CI < 0.65              -> no-go, pivot to semantic markers (#39)

  * Mandatory baseline-beat: delta-AUC over mean intra-cluster cosine
    >= 0.10 to count as a real geometric finding rather than a relabeling.

  * Pairwise secondary gate: Mann-Whitney U on the residual cosine
    distribution between same-topic-diff-role pairs (target) and
    diff-topic-same-role pairs (control), p < 0.01 with |Cliff delta| > 0.3.

If the experiment fails: framing is "even higher-order geometry does not
recover role structure; external structure is necessary," which strengthens
the merken paper's argument for graph-based or marker-based approaches and
pre-commits the pivot to issue #39.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger("role_geometry")

DEFAULT_SCENARIO = "experiments/loop_quality/scenarios/knowledge_update_50topics.json"
N_MIN_CLUSTER = 4
K_MIN_CLUSTER = 4
K_MAX_CLUSTER = 6


@dataclass
class Event:
    idx: int
    eid: str
    text: str
    topic: str
    fine_role: str  # initial / refinement / reversal / noise
    binary_role: str  # decision / noise


@dataclass
class Cluster:
    cluster_id: str
    cluster_class: str
    is_clean: bool
    indices: list[int] = field(default_factory=list)


def derive_fine_role(event_id: str) -> str:
    if event_id.startswith("noise"):
        return "noise"
    if event_id.endswith("_v1"):
        return "initial"
    if event_id.endswith("_v2"):
        return "refinement"
    if event_id.endswith("_v3"):
        return "reversal"
    raise ValueError(
        f"event id {event_id!r} did not match any known role pattern "
        "(*_v1 / *_v2 / *_v3 / noise_*); update derive_fine_role if "
        "the scenario schema has changed"
    )


def load_scenario(path: Path) -> list[Event]:
    raw = json.loads(path.read_text())
    events: list[Event] = []
    for i, e in enumerate(raw["events"]):
        fine = derive_fine_role(e["id"])
        binary = "noise" if fine == "noise" else "decision"
        events.append(
            Event(
                idx=i,
                eid=e["id"],
                text=e["text"],
                topic=e["topic"],
                fine_role=fine,
                binary_role=binary,
            )
        )
    return events


def resolve_embedder() -> str:
    """Return the embedding model name vstash would use for fresh stores.

    Fails loudly if vstash's config does not surface a model name. We do
    not silently fall back to a hardcoded default because the geometry
    we measure is model-dependent; a silent swap would invalidate the
    pre-registered AUC numbers.
    """
    from vstash.config import EmbeddingsConfig

    model = EmbeddingsConfig().model
    if not model:
        raise RuntimeError(
            "vstash.config.EmbeddingsConfig().model is empty; cannot "
            "proceed with a known embedding space"
        )
    return model


def embed_all(texts: list[str], model_name: str) -> np.ndarray:
    from vstash.embed import embed_texts

    t0 = time.perf_counter()
    vectors = embed_texts(texts, model_name=model_name)
    elapsed = time.perf_counter() - t0
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise RuntimeError(
            f"embed_texts returned shape {arr.shape!r} for {len(texts)} inputs"
        )
    logger.info(
        "embedded %d texts in %.2fs, dim=%d, model=%s",
        len(texts),
        elapsed,
        arr.shape[1],
        model_name,
    )
    return arr


def build_synthetic_clusters(
    events: list[Event],
    *,
    rng: np.random.Generator,
    k_target: int,
    seed: int,
    k_min: int = K_MIN_CLUSTER,
    k_max: int = K_MAX_CLUSTER,
) -> list[Cluster]:
    """Build stratified clusters of known composition.

    Seven classes, each of N in [k_min, k_max] events. Four are "clean"
    (same fine role across the cluster) and three are "contaminated"
    (multiple fine roles or random across topics).

    The two contaminated decision-only classes (mixed_role_in_topic and
    mixed_role_decisions_no_noise) bracket H3: the former mixes roles
    with a noise event added (confounds role and noise contamination),
    the latter mixes roles using only decision events (isolates role
    mismatch from noise contamination). Comparing their PR distributions
    in the writeup tells whether the signal is role-driven or
    noise-driven.
    """
    by_topic_role: dict[tuple[str, str], list[int]] = {}
    for e in events:
        by_topic_role.setdefault((e.topic, e.fine_role), []).append(e.idx)

    decision_topics = sorted({e.topic for e in events if e.binary_role == "decision"})
    noise_indices = [e.idx for e in events if e.fine_role == "noise"]

    clusters: list[Cluster] = []

    def _add(cluster_class: str, is_clean: bool, indices: list[int]) -> None:
        cid = f"{cluster_class}_seed{seed}_{len(clusters):04d}"
        clusters.append(
            Cluster(
                cluster_id=cid,
                cluster_class=cluster_class,
                is_clean=is_clean,
                indices=list(indices),
            )
        )

    # 1-3. clean_role_{initial, refinement, reversal}: K events of same role
    # drawn from K distinct topics.
    for role in ("initial", "refinement", "reversal"):
        topics_with_role = [
            t for t in decision_topics if (t, role) in by_topic_role
        ]
        if len(topics_with_role) < k_min:
            logger.warning(
                "skipping clean_role_%s: only %d topics have this role "
                "(< k_min=%d)",
                role,
                len(topics_with_role),
                k_min,
            )
            continue
        max_k = min(k_max, len(topics_with_role))
        for _ in range(k_target):
            k = int(rng.integers(k_min, max_k + 1))
            chosen = rng.choice(len(topics_with_role), size=k, replace=False)
            indices = [by_topic_role[(topics_with_role[c], role)][0] for c in chosen]
            _add(f"clean_role_{role}", is_clean=True, indices=indices)

    # 4. clean_noise: K random noise events.
    if len(noise_indices) >= k_min:
        for _ in range(k_target):
            k = int(rng.integers(k_min, k_max + 1))
            chosen = rng.choice(len(noise_indices), size=k, replace=False)
            indices = [noise_indices[c] for c in chosen]
            _add("clean_noise", is_clean=True, indices=indices)

    # 5. mixed_role_in_topic (TARGET POSITIVE with noise): v1+v2+v3 of one
    # topic + 1 uniformly random noise event. Fixed N=4 because each topic
    # only has one event per fine role. NB: confounds role and noise
    # contamination -- compare against #6 (no-noise variant) in the writeup.
    eligible = [
        t
        for t in decision_topics
        if all((t, r) in by_topic_role for r in ("initial", "refinement", "reversal"))
    ]
    if eligible and noise_indices:
        for _ in range(k_target):
            t = eligible[int(rng.integers(0, len(eligible)))]
            indices = [
                by_topic_role[(t, "initial")][0],
                by_topic_role[(t, "refinement")][0],
                by_topic_role[(t, "reversal")][0],
                int(noise_indices[int(rng.integers(0, len(noise_indices)))]),
            ]
            _add("mixed_role_in_topic", is_clean=False, indices=indices)
    else:
        logger.warning(
            "skipping mixed_role_in_topic: eligible_topics=%d, noise=%d",
            len(eligible),
            len(noise_indices),
        )

    # 6. mixed_role_decisions_no_noise (TARGET POSITIVE without noise):
    # v1+v2+v3 of topic A + one random decision from topic B (any role).
    # All decision events; isolates role-mismatch from noise contamination.
    # If both #5 and #6 score high on PR vs clean classes, the signal is
    # role-driven. If only #5 does, the signal is noise-driven.
    if len(eligible) >= 2:
        for _ in range(k_target):
            t_a = eligible[int(rng.integers(0, len(eligible)))]
            other = [t for t in decision_topics if t != t_a]
            t_b = other[int(rng.integers(0, len(other)))]
            roles_b = [
                r
                for r in ("initial", "refinement", "reversal")
                if (t_b, r) in by_topic_role
            ]
            role_b = roles_b[int(rng.integers(0, len(roles_b)))]
            indices = [
                by_topic_role[(t_a, "initial")][0],
                by_topic_role[(t_a, "refinement")][0],
                by_topic_role[(t_a, "reversal")][0],
                by_topic_role[(t_b, role_b)][0],
            ]
            _add("mixed_role_decisions_no_noise", is_clean=False, indices=indices)
    else:
        logger.warning(
            "skipping mixed_role_decisions_no_noise: only %d eligible 3-version "
            "topics, need >= 2",
            len(eligible),
        )

    # 7. mixed_topic_decisions: K events from K different decision topics
    # with role chosen randomly from the available {initial, refinement,
    # reversal} per topic. Mixes both axes by construction. The
    # `is_role_pure` flag distinguishes accidental same-role realizations
    # (which conflate topic and role contamination) so the writeup can
    # report AUC excluding them; see H2 in the code review.
    if len(decision_topics) >= k_min:
        max_k = min(k_max, len(decision_topics))
        for _ in range(k_target):
            k = int(rng.integers(k_min, max_k + 1))
            chosen_topics = rng.choice(len(decision_topics), size=k, replace=False)
            indices: list[int] = []
            chosen_roles: list[str] = []
            for ct in chosen_topics:
                topic = decision_topics[ct]
                roles_for_topic = [
                    r
                    for r in ("initial", "refinement", "reversal")
                    if (topic, r) in by_topic_role
                ]
                role = roles_for_topic[int(rng.integers(0, len(roles_for_topic)))]
                chosen_roles.append(role)
                indices.append(by_topic_role[(topic, role)][0])
            # If by chance the K random draws all picked the same role, the
            # cluster is genuinely role-clean (different topics, same role)
            # and should be labeled is_clean=True. Otherwise it is the
            # intended mixed_role mixed_topic case. Bug audit 2026-04-28:
            # original code labeled the role-pure subclass is_clean=False,
            # injecting clean clusters into the contaminated class. The
            # effect on the headline AUC was < 1pp because the rate is
            # ~3% (3 distinct roles, K~5 events, P(all same)~3/3^5 = 0.04),
            # but the labeling is now semantically correct.
            role_pure = len(set(chosen_roles)) == 1
            cluster_class = (
                "mixed_topic_decisions_role_pure"
                if role_pure
                else "mixed_topic_decisions"
            )
            _add(cluster_class, is_clean=role_pure, indices=indices)

    logger.info(
        "built %d synthetic clusters across %d classes",
        len(clusters),
        len({c.cluster_class for c in clusters}),
    )
    return clusters


def cluster_metrics(vectors: np.ndarray, indices: list[int]) -> dict[str, float]:
    """Return PR, EVR_1, AngDisp, NormVar, baseline_cos for one cluster."""
    sub = vectors[indices]  # (N, d)
    n, _d = sub.shape
    centroid = sub.mean(axis=0)
    residuals = sub - centroid  # (N, d)

    # SVD of residual matrix; singular values^2 are the eigenvalues of
    # the residual covariance (up to scaling). PR works directly off the
    # singular spectrum and is robust to rank deficiency.
    sigma = np.linalg.svd(residuals, compute_uv=False)
    eigs = sigma**2  # length min(N, d)

    eig_sum = float(eigs.sum())
    if eig_sum > 0.0:
        pr = (eig_sum**2) / float((eigs**2).sum())
        evr_1 = float(eigs[0]) / eig_sum
    else:
        # All vectors identical; degenerate cluster.
        pr = 1.0
        evr_1 = 1.0

    # Angular dispersion on L2-normalized residuals (independent of
    # eigenvalue magnitudes; tests pure direction structure).
    norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    unit_residuals = residuals / safe
    cos_matrix = unit_residuals @ unit_residuals.T
    iu = np.triu_indices(n, k=1)
    if iu[0].size > 0:
        ang_disp = float(np.mean(1.0 - cos_matrix[iu]))
    else:
        ang_disp = 0.0

    # Magnitude scatter (auxiliary).
    norm_var = float(np.var(norms.ravel()))

    # Baseline: mean pairwise cosine on the RAW vectors (the metric the
    # motivation says fails). Higher means tighter cluster.
    raw_norms = np.linalg.norm(sub, axis=1, keepdims=True)
    raw_safe = np.where(raw_norms > 0.0, raw_norms, 1.0)
    raw_unit = sub / raw_safe
    raw_cos = raw_unit @ raw_unit.T
    baseline_cos_mean = float(np.mean(raw_cos[iu])) if iu[0].size > 0 else 1.0

    return {
        "n": float(n),
        "pr": float(pr),
        "evr_1": float(evr_1),
        "ang_disp": ang_disp,
        "norm_var": norm_var,
        "baseline_cos_mean": baseline_cos_mean,
    }


def pairwise_analysis(
    events: list[Event],
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays for all pairs (i<j).

    Returns
    -------
    cos_raw : (P,) cosine on raw vectors
    cos_residual : (P,) cosine on residuals (vec - global_centroid)
    same_topic : (P,) bool
    same_fine_role : (P,) bool
    pair_index : (P, 2) i, j indices
    """
    n = len(events)
    centroid = vectors.mean(axis=0)
    residuals = vectors - centroid

    raw_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    raw_unit = vectors / np.where(raw_norms > 0.0, raw_norms, 1.0)

    res_norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    res_unit = residuals / np.where(res_norms > 0.0, res_norms, 1.0)

    cos_raw_mat = raw_unit @ raw_unit.T
    cos_res_mat = res_unit @ res_unit.T

    iu = np.triu_indices(n, k=1)
    cos_raw = cos_raw_mat[iu]
    cos_residual = cos_res_mat[iu]

    topics = np.array([e.topic for e in events])
    fine_roles = np.array([e.fine_role for e in events])
    same_topic = topics[iu[0]] == topics[iu[1]]
    same_fine_role = fine_roles[iu[0]] == fine_roles[iu[1]]
    pair_index = np.stack([iu[0], iu[1]], axis=1)

    return cos_raw, cos_residual, same_topic, same_fine_role, pair_index


def bootstrap_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Stratified bootstrap of ROC-AUC and PR-AUC.

    Returns mean and 95% CI (2.5/97.5 percentiles).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if pos.size == 0 or neg.size == 0:
        return {
            "roc_auc_mean": float("nan"),
            "roc_auc_lo": float("nan"),
            "roc_auc_hi": float("nan"),
            "pr_auc_mean": float("nan"),
            "pr_auc_lo": float("nan"),
            "pr_auc_hi": float("nan"),
            "n_pos": float(pos.size),
            "n_neg": float(neg.size),
        }

    point_roc = float(roc_auc_score(labels, scores))
    point_pr = float(average_precision_score(labels, scores))

    rocs = np.empty(n_resamples)
    prs = np.empty(n_resamples)
    for b in range(n_resamples):
        ip = rng.choice(pos, size=pos.size, replace=True)
        ineg = rng.choice(neg, size=neg.size, replace=True)
        idx = np.concatenate([ip, ineg])
        rocs[b] = roc_auc_score(labels[idx], scores[idx])
        prs[b] = average_precision_score(labels[idx], scores[idx])

    return {
        "roc_auc_mean": point_roc,
        "roc_auc_lo": float(np.percentile(rocs, 2.5)),
        "roc_auc_hi": float(np.percentile(rocs, 97.5)),
        "pr_auc_mean": point_pr,
        "pr_auc_lo": float(np.percentile(prs, 2.5)),
        "pr_auc_hi": float(np.percentile(prs, 97.5)),
        "n_pos": float(pos.size),
        "n_neg": float(neg.size),
    }


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = P(X>Y) - P(X<Y), in [-1, 1]."""
    if x.size == 0 or y.size == 0:
        return float("nan")
    # Vectorized via sort + searchsorted: O((n+m) log m) instead of O(nm).
    y_sorted = np.sort(y)
    gt = np.searchsorted(y_sorted, x, side="left")  # x > y count per element
    lt = y_sorted.size - np.searchsorted(y_sorted, x, side="right")  # x < y
    return float((gt.sum() - lt.sum()) / (x.size * y_sorted.size))


def decide_primary(roc_auc_lo: float) -> str:
    if np.isnan(roc_auc_lo):
        return "INSUFFICIENT_DATA"
    if roc_auc_lo > 0.80:
        return "STRONG_GO"
    if roc_auc_lo >= 0.65:
        return "PARTIAL_COMBINED_GATE_INPUT"
    return "NO_GO_PIVOT_TO_SEMANTIC_MARKERS"


def decide_baseline_beat(metric_auc: float, baseline_auc: float) -> str:
    if np.isnan(metric_auc) or np.isnan(baseline_auc):
        return "INSUFFICIENT_DATA"
    delta = metric_auc - baseline_auc
    if delta >= 0.10:
        return "BEATS_BASELINE"
    return "DOES_NOT_BEAT_BASELINE"


def write_metrics_csv(
    out: Path,
    clusters: list[Cluster],
    metrics: list[dict[str, float]],
    events: list[Event],
) -> None:
    cols = [
        "cluster_id",
        "cluster_class",
        "is_clean",
        "n",
        "pr",
        "evr_1",
        "ang_disp",
        "norm_var",
        "baseline_cos_mean",
        "topics",
        "fine_roles",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for cl, m in zip(clusters, metrics):
            row_topics = sorted({events[i].topic for i in cl.indices})
            row_roles = sorted({events[i].fine_role for i in cl.indices})
            w.writerow(
                {
                    "cluster_id": cl.cluster_id,
                    "cluster_class": cl.cluster_class,
                    "is_clean": int(cl.is_clean),
                    "n": int(m["n"]),
                    "pr": f"{m['pr']:.6f}",
                    "evr_1": f"{m['evr_1']:.6f}",
                    "ang_disp": f"{m['ang_disp']:.6f}",
                    "norm_var": f"{m['norm_var']:.6f}",
                    "baseline_cos_mean": f"{m['baseline_cos_mean']:.6f}",
                    "topics": "|".join(row_topics),
                    "fine_roles": "|".join(row_roles),
                }
            )


def write_pairwise_csv(
    out: Path,
    pair_index: np.ndarray,
    cos_raw: np.ndarray,
    cos_res: np.ndarray,
    same_topic: np.ndarray,
    same_fine_role: np.ndarray,
    events: list[Event],
    *,
    sample_cap: int | None = 200_000,
    rng: np.random.Generator | None = None,
) -> None:
    """Write pairwise rows. For 1100 events that's 605k rows; subsample
    to keep the CSV under a few hundred MB unless the caller overrides.
    The arrays themselves stay in memory in full for the AUC/MW-U tests.
    """
    n = pair_index.shape[0]
    if sample_cap is not None and n > sample_cap and rng is not None:
        idx = rng.choice(n, size=sample_cap, replace=False)
    else:
        idx = np.arange(n)

    cols = [
        "i",
        "j",
        "topic_i",
        "topic_j",
        "fine_role_i",
        "fine_role_j",
        "same_topic",
        "same_fine_role",
        "cos_raw",
        "cos_residual",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in idx:
            i, j = int(pair_index[k, 0]), int(pair_index[k, 1])
            w.writerow(
                {
                    "i": i,
                    "j": j,
                    "topic_i": events[i].topic,
                    "topic_j": events[j].topic,
                    "fine_role_i": events[i].fine_role,
                    "fine_role_j": events[j].fine_role,
                    "same_topic": int(bool(same_topic[k])),
                    "same_fine_role": int(bool(same_fine_role[k])),
                    "cos_raw": f"{cos_raw[k]:.6f}",
                    "cos_residual": f"{cos_res[k]:.6f}",
                }
            )


def render_plots(
    out_dir: Path,
    clusters: list[Cluster],
    metrics: list[dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    is_clean = np.array([1 if c.is_clean else 0 for c in clusters])
    pr = np.array([m["pr"] for m in metrics])
    evr = np.array([m["evr_1"] for m in metrics])
    ang = np.array([m["ang_disp"] for m in metrics])
    classes = np.array([c.cluster_class for c in clusters])

    def hist(metric: np.ndarray, name: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(metric[is_clean == 1], bins=20, alpha=0.6, label="clean")
        ax.hist(metric[is_clean == 0], bins=20, alpha=0.6, label="contaminated")
        ax.set_title(f"{name} -- clean vs contaminated")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{name}.png", dpi=120)
        plt.close(fig)

    hist(pr, "pr")
    hist(evr, "evr_1")
    hist(ang, "ang_disp")

    def scatter(x: np.ndarray, y: np.ndarray, xn: str, yn: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cls in sorted(set(classes)):
            mask = classes == cls
            ax.scatter(x[mask], y[mask], s=18, alpha=0.7, label=cls)
        ax.set_xlabel(xn)
        ax.set_ylabel(yn)
        ax.set_title(f"{xn} vs {yn}")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{xn}_vs_{yn}.png", dpi=120)
        plt.close(fig)

    scatter(pr, evr, "pr", "evr_1")
    scatter(pr, ang, "pr", "ang_disp")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--scenario",
        type=Path,
        default=Path(DEFAULT_SCENARIO),
        help=f"path to scenario JSON (default: {DEFAULT_SCENARIO})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; default experiments/role_geometry/runs/<ts>",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for cluster construction and bootstrap (default 42)",
    )
    p.add_argument(
        "--k-target",
        type=int,
        default=20,
        help="target clusters per class (default 20)",
    )
    p.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="cluster-level bootstrap resamples (default 1000)",
    )
    p.add_argument(
        "--pairwise-bootstrap-resamples",
        type=int,
        default=1000,
        help=(
            "pairwise bootstrap resamples (default 1000); separate from "
            "cluster-level so the spec-promised 1000 is honored at both "
            "scales without requiring the cluster sweep cost"
        ),
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="skip matplotlib plot generation",
    )
    p.add_argument(
        "--pairwise-csv-cap",
        type=int,
        default=200_000,
        help="max rows in pairwise_role_geometry.csv (sampling cap)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
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
        args.output_dir = Path("experiments/role_geometry/runs") / ts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Spawn one independent child generator per consumer so that adding or
    # reordering stages does not silently shift the bootstrap CIs. Without
    # this, a CSV-cap change downstream would mutate the upstream AUC
    # numbers via the shared draw counter.
    rng_root = np.random.default_rng(args.seed)
    rng_clusters, rng_cluster_boot, rng_pair_csv, rng_pair_boot = rng_root.spawn(4)

    logger.info("loading scenario %s", args.scenario)
    events = load_scenario(args.scenario)
    logger.info(
        "loaded %d events; topics=%d; roles=%s",
        len(events),
        len({e.topic for e in events}),
        sorted({e.fine_role for e in events}),
    )

    model_name = resolve_embedder()
    vectors = embed_all([e.text for e in events], model_name=model_name)

    clusters = build_synthetic_clusters(
        events, rng=rng_clusters, k_target=args.k_target, seed=args.seed
    )
    metrics: list[dict[str, float]] = []
    excluded = 0
    kept_clusters: list[Cluster] = []
    for cl in clusters:
        if len(cl.indices) < N_MIN_CLUSTER:
            excluded += 1
            continue
        metrics.append(cluster_metrics(vectors, cl.indices))
        kept_clusters.append(cl)
    logger.info(
        "cluster-level: kept %d / %d clusters (excluded %d for N<%d)",
        len(kept_clusters),
        len(clusters),
        excluded,
        N_MIN_CLUSTER,
    )

    write_metrics_csv(
        args.output_dir / "metrics_by_cluster.csv",
        kept_clusters,
        metrics,
        events,
    )

    is_clean = np.array([1 if c.is_clean else 0 for c in kept_clusters])
    is_contam = 1 - is_clean
    pr_vals = np.array([m["pr"] for m in metrics])
    evr_vals = np.array([m["evr_1"] for m in metrics])
    ang_vals = np.array([m["ang_disp"] for m in metrics])
    norm_vals = np.array([m["norm_var"] for m in metrics])
    baseline_vals = np.array([m["baseline_cos_mean"] for m in metrics])

    cluster_aucs: dict[str, dict[str, float]] = {}
    for name, scores in [
        ("pr", pr_vals),
        ("evr_1", evr_vals),
        ("ang_disp", ang_vals),
        ("norm_var", norm_vals),
        ("baseline_neg_cos", -baseline_vals),  # contaminated -> lower cosine
    ]:
        cluster_aucs[name] = bootstrap_auc(
            scores=scores,
            labels=is_contam,
            n_resamples=args.bootstrap_resamples,
            rng=rng_cluster_boot,
        )

    primary = cluster_aucs["pr"]
    baseline = cluster_aucs["baseline_neg_cos"]
    primary_verdict = decide_primary(primary["roc_auc_lo"])
    baseline_verdict = decide_baseline_beat(
        primary["roc_auc_mean"], baseline["roc_auc_mean"]
    )

    logger.info(
        "cluster-level PR ROC-AUC: mean=%.3f CI=[%.3f, %.3f] (n_pos=%d, n_neg=%d) -> %s",
        primary["roc_auc_mean"],
        primary["roc_auc_lo"],
        primary["roc_auc_hi"],
        int(primary["n_pos"]),
        int(primary["n_neg"]),
        primary_verdict,
    )
    logger.info(
        "cluster-level baseline (mean cosine, sign-flipped) ROC-AUC: mean=%.3f -> %s",
        baseline["roc_auc_mean"],
        baseline_verdict,
    )

    cos_raw, cos_res, same_topic, same_role, pair_idx = pairwise_analysis(
        events, vectors
    )
    write_pairwise_csv(
        args.output_dir / "pairwise_role_geometry.csv",
        pair_idx,
        cos_raw,
        cos_res,
        same_topic,
        same_role,
        events,
        sample_cap=args.pairwise_csv_cap,
        rng=rng_pair_csv,
    )

    target_mask = same_topic & ~same_role
    control_mask = (~same_topic) & same_role
    logger.info(
        "pairwise: target same-topic-diff-role n=%d; control diff-topic-same-role n=%d",
        int(target_mask.sum()),
        int(control_mask.sum()),
    )

    pair_aucs: dict[str, dict[str, float]] = {}
    pair_mw: dict[str, dict[str, float]] = {}
    if target_mask.any() and control_mask.any():
        from scipy.stats import mannwhitneyu

        # AUC: contamination = "same-topic diff-role" pair (label=1).
        # Control = "diff-topic same-role" pair (label=0).
        labels = np.concatenate(
            [np.ones(target_mask.sum()), np.zeros(control_mask.sum())]
        )
        for name, arr in [("cos_residual", cos_res), ("cos_raw", cos_raw)]:
            # For contamination, larger angular distance == 1 - cos == lower cosine
            # so we invert (negate) cosine to get "contamination score".
            scores = np.concatenate([-arr[target_mask], -arr[control_mask]])
            pair_aucs[name] = bootstrap_auc(
                scores=scores,
                labels=labels.astype(int),
                n_resamples=args.pairwise_bootstrap_resamples,
                rng=rng_pair_boot,
            )

        # Mann-Whitney U on residual cosine.
        # Pre-registered direction: target (same-topic diff-role) cos <
        # control (diff-topic same-role) cos -- if the residual axis carries
        # role mismatch, role-mismatched pairs are *less* aligned in
        # residual space than role-matched cross-topic pairs.
        # We report both the directional p (alternative="less") and the
        # two-sided p so a "wrong direction but significant" outcome is
        # not silently classified as null. See H4 in the code review.
        for name, arr in [("cos_residual", cos_res), ("cos_raw", cos_raw)]:
            target_vals = arr[target_mask]
            control_vals = arr[control_mask]
            stat_dir = mannwhitneyu(
                target_vals, control_vals, alternative="less"
            )
            stat_two = mannwhitneyu(
                target_vals, control_vals, alternative="two-sided"
            )
            delta = cliff_delta(target_vals, control_vals)
            pair_mw[name] = {
                "U": float(stat_dir.statistic),
                "p_value_directional_less": float(stat_dir.pvalue),
                "p_value_two_sided": float(stat_two.pvalue),
                "cliff_delta": float(delta),
                "median_target": float(np.median(target_vals)),
                "median_control": float(np.median(control_vals)),
                "n_target": float(target_vals.size),
                "n_control": float(control_vals.size),
            }
        logger.info(
            "pairwise residual cos -- AUC mean=%.3f CI=[%.3f, %.3f]; "
            "MW-U p_less=%.3g p_two=%.3g cliff=%.3f",
            pair_aucs["cos_residual"]["roc_auc_mean"],
            pair_aucs["cos_residual"]["roc_auc_lo"],
            pair_aucs["cos_residual"]["roc_auc_hi"],
            pair_mw["cos_residual"]["p_value_directional_less"],
            pair_mw["cos_residual"]["p_value_two_sided"],
            pair_mw["cos_residual"]["cliff_delta"],
        )
    else:
        logger.warning(
            "pairwise: target or control stratum empty; skipping AUC/MW-U"
        )

    import hashlib

    import vstash

    summary = {
        "scenario": str(args.scenario),
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "embedding_model": model_name,
        "embedding_dim": int(vectors.shape[1]),
        "seed": args.seed,
        "k_target": args.k_target,
        "bootstrap_resamples_cluster": args.bootstrap_resamples,
        "bootstrap_resamples_pair": args.pairwise_bootstrap_resamples,
        "vstash_version": getattr(vstash, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "n_events": len(events),
        "n_clusters_total": len(clusters),
        "n_clusters_kept": len(kept_clusters),
        "n_clusters_excluded": excluded,
        "cluster_class_counts": {
            cls: sum(1 for c in kept_clusters if c.cluster_class == cls)
            for cls in sorted({c.cluster_class for c in kept_clusters})
        },
        "cluster_aucs": cluster_aucs,
        "pair_aucs": pair_aucs,
        "pair_mw": pair_mw,
        "primary_verdict": primary_verdict,
        "baseline_verdict": baseline_verdict,
    }
    # Recursively replace NaN/inf with None so the JSON is strictly valid.
    def _scrub(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_scrub(x) for x in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    (args.output_dir / "auc_summary.json").write_text(
        json.dumps(_scrub(summary), indent=2, allow_nan=False)
    )
    logger.info("summary written to %s", args.output_dir / "auc_summary.json")

    if not args.no_plot:
        render_plots(args.output_dir / "plots", kept_clusters, metrics)
        logger.info("plots written to %s", args.output_dir / "plots")

    print("\n=== role_geometry verdict ===")
    print(f"scenario       : {args.scenario}")
    print(f"embedding model: {model_name} (dim={vectors.shape[1]})")
    print(f"seed           : {args.seed}")
    print(
        f"clusters       : kept={len(kept_clusters)} (clean={int(is_clean.sum())}, "
        f"contam={int(is_contam.sum())}), excluded_below_n_min={excluded}"
    )
    print(
        f"PR AUC         : {primary['roc_auc_mean']:.3f} "
        f"[{primary['roc_auc_lo']:.3f}, {primary['roc_auc_hi']:.3f}] "
        f"(n_pos={int(primary['n_pos'])}, n_neg={int(primary['n_neg'])})"
    )
    print(
        f"baseline AUC   : {baseline['roc_auc_mean']:.3f} "
        f"(neg-cos cluster mean)"
    )
    print(f"primary gate   : {primary_verdict}")
    print(f"baseline beat  : {baseline_verdict}")
    if pair_mw:
        m = pair_mw["cos_residual"]
        print(
            f"pairwise MW-U  : p_less={m['p_value_directional_less']:.3g} "
            f"p_two={m['p_value_two_sided']:.3g} "
            f"cliff_delta={m['cliff_delta']:.3f} "
            f"(target_med={m['median_target']:.3f} vs "
            f"control_med={m['median_control']:.3f})"
        )
    print("=============================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
