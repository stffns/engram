"""Phase 2 diagnostic: marker-prefix conditioning on the embedding manifold.

Pre-registered Phase 2 of #39 was gated on Phase 1 reaching at least
PARTIAL. Phase 1 landed borderline (3-seed mean macro-F1 0.753 in the
PARTIAL band, but worst lower CI 0.633 below the strict 0.70 threshold;
2/3 seeds NO_GO). Rather than running the gated Phase 2 as written
(which would inherit Phase 1's DECISION-PREFERENCE collision via
predicted-role prefixes), this script runs Phase 2 as a *diagnostic*:

- Cohort A: vanilla embeddings (no prefix). Baseline.
- Cohort B: prefix with **ground-truth role**. Tests whether
  prefix-conditioning helps the embedding manifold separate roles
  when the prefix is correct.
- Cohort C: prefix with **predicted role from Phase 1**. The
  pre-registered Phase 2 cohort. Tests the realistic deployment
  scenario.
- Cohort D: prefix with **random role-shaped tag**. Pre-registered
  control. Isolates prefix-induced clustering from semantic role
  effect.

Hypothesis the diagnostic answers: does prepending `[role: decision]`
to a DECISION event push its embedding into a DECISION subspace,
separating it from PREFERENCE events that get `[role: preference]`?

If yes (cohort B separates more than D): the embedding manifold can
host role information when explicitly cued; Phase 1's prediction is
the bottleneck, and taxonomy redefinition (replacing DECISION with a
more lexically-distinct role) could rescue.

If no (cohort B does not separate more than D): prefix-conditioning
does not help; the failure is structural in BGE-small itself.

This script does NOT count as a gated Phase 2 result. The gated
Phase 2 (cohort C) is still subject to the spec's >= 30% contamination
drop with role-prefix - random-prefix gap >= 20pp; this script just
reports the underlying separation numbers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.role_markers.few_shot_classifier import (
    ROLES,
    DEFAULT_PROTOTYPES,
    DEFAULT_EVAL,
    classify_with_indices,
    embed_texts_array,
    l2_normalize,
    load_eval,
    load_prototypes,
    resolve_embedder,
    shuffle_proto_orders,
)

logger = logging.getLogger("marker_conditioned_eval")

# Role-shaped tags for the random-prefix control. Pre-committed:
# 4 NATO phonetic words chosen alphabetically. They have no role
# semantic but share the [role: XXX] structural shape.
RANDOM_TAGS = ("alpha", "bravo", "charlie", "delta")

PREFIX_TEMPLATE = "[role: {tag}] "


def apply_prefix(events: list[dict], tag_per_event: list[str]) -> list[str]:
    if len(tag_per_event) != len(events):
        raise ValueError("tag_per_event length mismatch")
    return [PREFIX_TEMPLATE.format(tag=tag) + e["text"] for e, tag in zip(events, tag_per_event)]


def cohort_vanilla(events: list[dict]) -> tuple[list[str], list[str]]:
    """No prefix. Returns (texts, descriptive_tags)."""
    return [e["text"] for e in events], ["(none)"] * len(events)


def cohort_ground_truth(events: list[dict]) -> tuple[list[str], list[str]]:
    """Prefix with the event's ground-truth role."""
    tags = [e["role"] for e in events]
    return apply_prefix(events, tags), tags


def cohort_predicted(events: list[dict], predicted_roles: list[str]) -> tuple[list[str], list[str]]:
    """Prefix with Phase 1's predicted role per event."""
    return apply_prefix(events, predicted_roles), predicted_roles


def cohort_random(
    events: list[dict],
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Prefix with a random tag drawn uniformly per event from RANDOM_TAGS."""
    tags = [RANDOM_TAGS[int(rng.integers(0, len(RANDOM_TAGS)))] for _ in events]
    return apply_prefix(events, tags), tags


def pair_role_means(
    vectors: np.ndarray,
    roles: np.ndarray,
) -> dict[tuple[str, str], dict[str, float]]:
    """For each ordered (role_a, role_b) pair, mean and stdev of
    pairwise cosine across all (i, j) with role_i == role_a, role_j == role_b
    and i < j (or i != j when a != b).
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms > 0.0, norms, 1.0)
    cos = unit @ unit.T  # (N, N), cos[i,j] = dot

    for r_a in ROLES:
        for r_b in ROLES:
            mask_a = roles == r_a
            mask_b = roles == r_b
            if r_a == r_b:
                # Upper triangle, k=1, restricted to role r_a
                idx_a = np.where(mask_a)[0]
                if idx_a.size < 2:
                    out[(r_a, r_b)] = {"mean": float("nan"), "n": 0}
                    continue
                sub = cos[np.ix_(idx_a, idx_a)]
                iu = np.triu_indices(idx_a.size, k=1)
                vals = sub[iu]
            else:
                idx_a = np.where(mask_a)[0]
                idx_b = np.where(mask_b)[0]
                if idx_a.size == 0 or idx_b.size == 0:
                    out[(r_a, r_b)] = {"mean": float("nan"), "n": 0}
                    continue
                sub = cos[np.ix_(idx_a, idx_b)]
                vals = sub.flatten()
            out[(r_a, r_b)] = {
                "mean": float(np.mean(vals)),
                "stdev": float(np.std(vals)),
                "n": int(vals.size),
            }
    return out


def compute_separation(
    pair_means: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """Summary statistics: same-role mean vs cross-role mean, plus
    the specific DECISION-PREFERENCE distance that drives Phase 1
    failure.
    """
    same = []
    cross = []
    for (a, b), v in pair_means.items():
        if v["n"] == 0 or np.isnan(v["mean"]):
            continue
        if a == b:
            same.append(v["mean"])
        else:
            cross.append(v["mean"])
    out = {
        "same_role_mean": float(np.mean(same)) if same else float("nan"),
        "cross_role_mean": float(np.mean(cross)) if cross else float("nan"),
    }
    out["separation"] = out["same_role_mean"] - out["cross_role_mean"]
    out["dec_pref_cross"] = pair_means.get(("decision", "preference"), {}).get(
        "mean", float("nan")
    )
    out["dec_dec_same"] = pair_means.get(("decision", "decision"), {}).get(
        "mean", float("nan")
    )
    out["pref_pref_same"] = pair_means.get(("preference", "preference"), {}).get(
        "mean", float("nan")
    )
    return out


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    p.add_argument("--eval", dest="eval_path", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default experiments/role_markers/runs/phase2_diag_<ts>")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-primary", type=int, default=7)
    p.add_argument("--embed-model", type=str, default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("experiments/role_markers/runs") / f"phase2_diag_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prototypes = load_prototypes(args.prototypes)
    events = load_eval(args.eval_path)
    sizes = {role: len(prototypes[role]) for role in ROLES}

    model_name = args.embed_model or resolve_embedder()
    logger.info("embedding model: %s", model_name)

    # Phase 1 prediction first (we need predicted roles for cohort C).
    proto_texts = [p["text"] for role in ROLES for p in prototypes[role]]
    proto_vec = embed_texts_array(proto_texts, model_name=model_name)
    proto_vec = l2_normalize(proto_vec)

    proto_vectors_by_role: dict[str, np.ndarray] = {}
    cursor = 0
    for role in ROLES:
        size = sizes[role]
        proto_vectors_by_role[role] = proto_vec[cursor:cursor + size]
        cursor += size

    rng_root = np.random.default_rng(args.seed)
    rng_shuffle, rng_random = rng_root.spawn(2)
    proto_orders = shuffle_proto_orders(sizes, rng_shuffle)
    indices_per_role = {role: proto_orders[role][:args.k_primary] for role in ROLES}

    eval_vec_vanilla = embed_texts_array([e["text"] for e in events],
                                          model_name=model_name)
    eval_vec_vanilla_n = l2_normalize(eval_vec_vanilla)

    p1 = classify_with_indices(
        proto_vectors_by_role, eval_vec_vanilla_n,
        indices_per_role=indices_per_role,
    )
    predicted_roles = [ROLES[int(i)] for i in p1["predicted_idx"]]

    # Build the four cohorts. Each cohort produces its own (texts, tags).
    cohorts: dict[str, tuple[list[str], list[str]]] = {}
    cohorts["vanilla"] = cohort_vanilla(events)
    cohorts["ground_truth_prefix"] = cohort_ground_truth(events)
    cohorts["predicted_prefix"] = cohort_predicted(events, predicted_roles)
    cohorts["random_prefix"] = cohort_random(events, rng_random)

    # Embed each cohort.
    cohort_vectors: dict[str, np.ndarray] = {}
    for name, (texts, tags) in cohorts.items():
        logger.info("embedding cohort %s (%d events)...", name, len(texts))
        v = embed_texts_array(texts, model_name=model_name)
        cohort_vectors[name] = v

    # Compute pairwise role means per cohort.
    role_array = np.array([e["role"] for e in events])
    cohort_pair_means: dict[str, dict] = {}
    cohort_separation: dict[str, dict[str, float]] = {}
    for name, vec in cohort_vectors.items():
        pm = pair_role_means(vec, role_array)
        sep = compute_separation(pm)
        cohort_pair_means[name] = {
            f"{a}__{b}": pm[(a, b)] for (a, b) in pm
        }
        cohort_separation[name] = sep
        logger.info(
            "%s: same=%.3f cross=%.3f sep=%.3f | dec-pref=%.3f dec-dec=%.3f pref-pref=%.3f",
            name, sep["same_role_mean"], sep["cross_role_mean"],
            sep["separation"], sep["dec_pref_cross"],
            sep["dec_dec_same"], sep["pref_pref_same"],
        )

    # Per-event CSV: vanilla cosine to each role's centroid + predicted prefix vs vanilla
    # diff per event. (Smaller table than the role-pair JSON above.)
    csv_rows = []
    for i, e in enumerate(events):
        row = {
            "id": e["id"],
            "topic": e["topic"],
            "true_role": e["role"],
            "predicted_role": predicted_roles[i],
            "predicted_correct": int(predicted_roles[i] == e["role"]),
        }
        for name in cohorts:
            v = cohort_vectors[name][i]
            n = np.linalg.norm(v)
            row[f"{name}_norm"] = f"{n:.6f}"
        csv_rows.append(row)

    csv_cols = ["id", "topic", "true_role", "predicted_role", "predicted_correct"] + [
        f"{name}_norm" for name in cohorts
    ]
    with (args.output_dir / "phase2_per_event.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_cols)
        w.writeheader()
        w.writerows(csv_rows)

    # Pre-registered diagnostic interpretation.
    base_sep = cohort_separation["vanilla"]["separation"]
    gt_sep = cohort_separation["ground_truth_prefix"]["separation"]
    pred_sep = cohort_separation["predicted_prefix"]["separation"]
    rand_sep = cohort_separation["random_prefix"]["separation"]
    base_dp = cohort_separation["vanilla"]["dec_pref_cross"]
    gt_dp = cohort_separation["ground_truth_prefix"]["dec_pref_cross"]
    pred_dp = cohort_separation["predicted_prefix"]["dec_pref_cross"]
    rand_dp = cohort_separation["random_prefix"]["dec_pref_cross"]

    # Verdict logic (pre-registered before run):
    # - "manifold can host role with prefix" if ground-truth-prefix
    #   improves overall separation by >= 0.05 over vanilla AND beats
    #   random-prefix improvement by >= 0.05
    # - "DEC-PREF specifically rescued" if ground-truth-prefix DEC-PREF
    #   cross drops by >= 0.10 below vanilla DEC-PREF cross
    manifold_helps = (gt_sep - base_sep >= 0.05) and (gt_sep - rand_sep >= 0.05)
    dec_pref_rescued = (base_dp - gt_dp) >= 0.10

    if manifold_helps and dec_pref_rescued:
        diagnostic = "MANIFOLD_HOSTS_ROLE_AND_DEC_PREF_RESCUED"
    elif manifold_helps and not dec_pref_rescued:
        diagnostic = "MANIFOLD_HOSTS_ROLE_BUT_DEC_PREF_STILL_COLLIDES"
    elif not manifold_helps and dec_pref_rescued:
        diagnostic = "DEC_PREF_RESCUED_BUT_OVERALL_SEPARATION_FLAT"
    else:
        diagnostic = "PREFIX_CONDITIONING_INEFFECTIVE"

    import vstash

    summary = {
        "embedding_model": model_name,
        "embedding_model_overridden": args.embed_model is not None,
        "vstash_version": getattr(vstash, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "seed": args.seed,
        "k_primary": args.k_primary,
        "n_events": len(events),
        "prototypes_path": str(args.prototypes),
        "prototypes_sha256": hashlib.sha256(args.prototypes.read_bytes()).hexdigest(),
        "eval_path": str(args.eval_path),
        "eval_sha256": hashlib.sha256(args.eval_path.read_bytes()).hexdigest(),
        "predicted_role_distribution": {
            r: predicted_roles.count(r) for r in ROLES
        },
        "predicted_accuracy": float(
            sum(1 for i, e in enumerate(events) if predicted_roles[i] == e["role"])
            / len(events)
        ),
        "cohort_separation": cohort_separation,
        "cohort_pair_means": cohort_pair_means,
        "diagnostic_verdict": diagnostic,
        "deltas_vs_vanilla": {
            "ground_truth_prefix_sep_delta": gt_sep - base_sep,
            "predicted_prefix_sep_delta": pred_sep - base_sep,
            "random_prefix_sep_delta": rand_sep - base_sep,
            "ground_truth_prefix_dec_pref_delta": gt_dp - base_dp,
            "predicted_prefix_dec_pref_delta": pred_dp - base_dp,
            "random_prefix_dec_pref_delta": rand_dp - base_dp,
        },
    }

    def _scrub(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_scrub(x) for x in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    (args.output_dir / "phase2_metrics.json").write_text(
        json.dumps(_scrub(summary), indent=2, allow_nan=False)
    )

    print("\n=== #39 Phase 2 diagnostic ===")
    print(f"output dir     : {args.output_dir}")
    print(f"embedding model: {model_name}")
    print(f"seed           : {args.seed}")
    print(f"\n{'cohort':<22} {'same_role':>10} {'cross_role':>11} {'separation':>11} "
          f"{'dec-pref':>10} {'dec-dec':>9} {'pref-pref':>10}")
    for name in ("vanilla", "ground_truth_prefix", "predicted_prefix", "random_prefix"):
        sep = cohort_separation[name]
        print(f"  {name:<20} {sep['same_role_mean']:>10.3f} "
              f"{sep['cross_role_mean']:>11.3f} {sep['separation']:>11.3f} "
              f"{sep['dec_pref_cross']:>10.3f} {sep['dec_dec_same']:>9.3f} "
              f"{sep['pref_pref_same']:>10.3f}")
    print(f"\nDeltas vs vanilla (separation = same - cross, higher = better):")
    print(f"  ground_truth_prefix : {gt_sep - base_sep:+.3f}")
    print(f"  predicted_prefix    : {pred_sep - base_sep:+.3f}")
    print(f"  random_prefix       : {rand_sep - base_sep:+.3f}")
    print(f"\nDEC-PREF cross-role cosine deltas (lower = more separation):")
    print(f"  ground_truth_prefix : {gt_dp - base_dp:+.3f}")
    print(f"  predicted_prefix    : {pred_dp - base_dp:+.3f}")
    print(f"  random_prefix       : {rand_dp - base_dp:+.3f}")
    print(f"\nDiagnostic verdict: {diagnostic}")
    print("===============================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
