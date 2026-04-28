"""Few-shot semantic role classifier (#39 Phase 1).

Loads prototypes.json + eval.json, embeds via vstash.embed.embed_texts,
classifies each eval event by argmax mean-cosine similarity to each
role's prototypes. Reports macro-F1, per-class accuracy, confusion
matrix, margin distribution, and a K-saturation curve at
K in {1, 3, 5, 7, 10}.

Pre-registered gate per `notes/2026-04-28-semantic-markers-issue.md`:

    macro-F1 >= 0.85 AND per-class >= 0.75 -> STRONG (proceed Phase 2)
    macro-F1 0.70 - 0.85 OR one class < 0.75 with others >= 0.85
                                            -> PARTIAL (Phase 2 marker-assisted)
    macro-F1 < 0.70                         -> NO_GO

Anti-leak: prototypes are sourced from the prototype topic set
(logging, dashboards, build_system, code_review, oncall) and eval
events are sourced from a disjoint topic set (payment, search, mobile,
etc.). Disjointness is enforced at run time by an exact-text overlap
check; the script aborts if any prototype text matches any eval text.
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

logger = logging.getLogger("few_shot_classifier")

ROLES = ("decision", "investigation", "observation", "preference")
DEFAULT_PROTOTYPES = Path("experiments/role_markers/prototypes.json")
DEFAULT_EVAL = Path("experiments/role_markers/eval.json")
DEFAULT_K_LIST = (1, 3, 5, 7, 10)


def resolve_embedder() -> str:
    """Return the embedding model name vstash would use, fail-fast on empty."""
    from vstash.config import EmbeddingsConfig

    model = EmbeddingsConfig().model
    if not model:
        raise RuntimeError("vstash.config.EmbeddingsConfig().model is empty")
    return model


def embed_texts_array(texts: list[str], model_name: str) -> np.ndarray:
    from vstash.embed import embed_texts

    t0 = time.perf_counter()
    vectors = embed_texts(texts, model_name=model_name)
    elapsed = time.perf_counter() - t0
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise RuntimeError(
            f"embed_texts returned shape {arr.shape!r} for {len(texts)} inputs"
        )
    logger.info("embedded %d texts in %.2fs (model=%s, dim=%d)",
                len(texts), elapsed, model_name, arr.shape[1])
    return arr


def load_prototypes(path: Path) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text())
    if "roles" not in raw:
        raise ValueError(f"{path} missing 'roles' key")
    out: dict[str, list[dict]] = {}
    for role in ROLES:
        if role not in raw["roles"]:
            raise ValueError(f"{path} missing role {role!r}")
        out[role] = list(raw["roles"][role])
        if not out[role]:
            raise ValueError(f"{path} has empty pool for role {role!r}")
    return out


def load_eval(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if "events" not in raw:
        raise ValueError(f"{path} missing 'events' key")
    events = list(raw["events"])
    for e in events:
        if e.get("role") not in ROLES:
            raise ValueError(f"event {e.get('id')!r} has invalid role: {e.get('role')!r}")
    return events


def assert_disjoint(prototypes: dict[str, list[dict]], events: list[dict]) -> None:
    proto_texts = {p["text"].strip() for role in ROLES for p in prototypes[role]}
    eval_texts = {e["text"].strip() for e in events}
    overlap = proto_texts & eval_texts
    if overlap:
        raise RuntimeError(
            f"anti-leak violated: {len(overlap)} prototype texts also appear in "
            f"eval set. Aborting. Sample: {next(iter(overlap))!r}"
        )
    proto_topics = {p["topic"] for role in ROLES for p in prototypes[role]}
    eval_topics = {e["topic"] for e in events}
    topic_overlap = proto_topics & eval_topics
    if topic_overlap:
        logger.warning(
            "prototype and eval pools share topics: %s. Spec says topics "
            "should be disjoint; check vocabulary if this is unintended.",
            sorted(topic_overlap),
        )


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


def shuffle_proto_orders(
    sizes: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    """Return one shuffled permutation of prototype indices per role.

    The K-saturation curve uses NESTED subsets (K=10 is superset of K=7
    is superset of K=5, ...) so the curve isolates the effect of K
    from the variance of which prototypes were drawn. Independent draws
    per K would conflate "more prototypes" with "different prototypes."
    """
    out: dict[str, list[int]] = {}
    for role, size in sizes.items():
        order = list(range(size))
        rng.shuffle(order)
        out[role] = order
    return out


def classify_with_indices(
    proto_vectors: dict[str, np.ndarray],  # role -> (P_role, d)
    eval_vectors: np.ndarray,  # (E, d), L2-normalized
    *,
    indices_per_role: dict[str, list[int]],
) -> dict:
    """Return per-event predicted role + per-role mean similarity + margin.

    Uses a pre-determined subset of prototypes per role (passed as
    `indices_per_role`) so the K-saturation curve is reproducible
    and nested across K values.
    """
    e_count = eval_vectors.shape[0]
    role_means = np.zeros((e_count, len(ROLES)), dtype=np.float64)

    for r_idx, role in enumerate(ROLES):
        pv = proto_vectors[role]
        idx = indices_per_role[role]
        if not idx:
            raise RuntimeError(f"role {role!r} got empty index list")
        if max(idx) >= pv.shape[0]:
            raise RuntimeError(
                f"role {role!r}: max idx {max(idx)} >= proto count {pv.shape[0]}"
            )
        sub = pv[idx]  # (k, d)
        # eval_vectors are normalized; sub is normalized; cosine = dot.
        sims = eval_vectors @ sub.T  # (E, k)
        role_means[:, r_idx] = sims.mean(axis=1)

    pred_idx = np.argmax(role_means, axis=1)
    sorted_means = -np.sort(-role_means, axis=1)
    top1 = sorted_means[:, 0]
    top2 = sorted_means[:, 1]
    margin = top1 - top2

    return {
        "predicted_idx": pred_idx,
        "role_means": role_means,
        "top1_sim": top1,
        "margin": margin,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0 or tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        n_true = int((y_true == c).sum())
        if n_true == 0:
            out.append(float("nan"))
            continue
        n_correct = int(((y_pred == c) & (y_true == c)).sum())
        out.append(n_correct / n_true)
    return out


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[list[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Stratified bootstrap of macro-F1 (resample within each class)."""
    by_class = [np.where(y_true == c)[0] for c in range(n_classes)]
    if any(arr.size == 0 for arr in by_class):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = macro_f1(y_true, y_pred, n_classes)
    resamples = np.empty(n_resamples)
    for b in range(n_resamples):
        idx_parts = [rng.choice(arr, size=arr.size, replace=True) for arr in by_class]
        idx = np.concatenate(idx_parts)
        resamples[b] = macro_f1(y_true[idx], y_pred[idx], n_classes)
    return {
        "mean": point,
        "lo": float(np.percentile(resamples, 2.5)),
        "hi": float(np.percentile(resamples, 97.5)),
    }


def decide(macro_f1_lo: float, per_class: list[float]) -> str:
    """Pre-registered gate per notes/2026-04-28-semantic-markers-issue.md.

    The spec table has three rows but leaves a hole when macro-F1
    >= 0.85 with multiple per-class < 0.75. Pre-committed (post-code-
    review 2026-04-28) to mapping that hole to
    NO_GO_PER_CLASS_GAPS_EXCEED_PARTIAL: macro-F1 alone passing the
    strong threshold while two-or-more classes are weak means the
    embedder lacks signal for those specific classes, which fits the
    spec's "embedder lacks the functional role signal we hoped for"
    framing better than a generous PARTIAL extension.
    """
    if np.isnan(macro_f1_lo):
        return "INSUFFICIENT_DATA"
    pc_finite = [p for p in per_class if not np.isnan(p)]
    if not pc_finite:
        return "INSUFFICIENT_DATA"
    pc_min = min(pc_finite)
    pc_count_lt_75 = sum(1 for p in pc_finite if p < 0.75)
    pc_count_ge_85 = sum(1 for p in pc_finite if p >= 0.85)
    if macro_f1_lo >= 0.85 and pc_min >= 0.75:
        return "STRONG_PROCEED_PHASE_2"
    # PARTIAL: macro-F1 in [0.70, 0.85] OR (one class < 0.75 with others >= 0.85)
    if (
        0.70 <= macro_f1_lo < 0.85
        or (pc_count_lt_75 == 1 and pc_count_ge_85 == len(pc_finite) - 1)
    ):
        return "PARTIAL_MARKER_ASSISTED"
    if macro_f1_lo < 0.70:
        return "NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL"
    # macro-F1 >= 0.85 but per-class fails to qualify for STRONG and the
    # weak-class pattern does not fit PARTIAL. Treat as NO_GO conservatively:
    # high aggregate F1 carried by 2-3 classes does not justify shipping
    # markers when the remaining classes are not separable.
    return "NO_GO_PER_CLASS_GAPS_EXCEED_PARTIAL"


def write_classification_csv(
    out: Path,
    events: list[dict],
    result: dict,
    role_to_idx: dict[str, int],
) -> None:
    cols = [
        "id", "topic", "true_role", "predicted_role", "correct",
        "top1_sim", "margin",
        "sim_decision", "sim_investigation", "sim_observation", "sim_preference",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, e in enumerate(events):
            true_idx = role_to_idx[e["role"]]
            pred_idx = int(result["predicted_idx"][i])
            row = {
                "id": e["id"],
                "topic": e["topic"],
                "true_role": e["role"],
                "predicted_role": ROLES[pred_idx],
                "correct": int(true_idx == pred_idx),
                "top1_sim": f"{result['top1_sim'][i]:.6f}",
                "margin": f"{result['margin'][i]:.6f}",
            }
            for r_idx, role in enumerate(ROLES):
                row[f"sim_{role}"] = f"{result['role_means'][i, r_idx]:.6f}"
            w.writerow(row)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    p.add_argument("--eval", dest="eval_path", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default experiments/role_markers/runs/<ts>")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-primary", type=int, default=7,
                   help="primary K for the gate decision")
    p.add_argument("--k-list", type=int, nargs="+",
                   default=list(DEFAULT_K_LIST),
                   help="K values for the saturation curve")
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--embed-model", type=str, default=None,
                   help="override vstash default for cross-embedder check")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("experiments/role_markers/runs") / f"phase1_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prototypes = load_prototypes(args.prototypes)
    events = load_eval(args.eval_path)
    assert_disjoint(prototypes, events)

    sizes = {role: len(prototypes[role]) for role in ROLES}
    logger.info("prototype sizes: %s", sizes)
    logger.info("eval set: %d events; class balance: %s", len(events),
                {role: sum(1 for e in events if e["role"] == role) for role in ROLES})

    max_k = max(args.k_list)
    if any(sz < max_k for sz in sizes.values()):
        raise RuntimeError(
            f"prototype pool too small for k_list max {max_k}: sizes={sizes}"
        )

    model_name = args.embed_model or resolve_embedder()
    logger.info("embedding model: %s", model_name)

    proto_texts: list[str] = []
    proto_role_ids: list[tuple[str, str]] = []
    for role in ROLES:
        for p in prototypes[role]:
            proto_texts.append(p["text"])
            proto_role_ids.append((role, p["id"]))
    proto_vec_all = embed_texts_array(proto_texts, model_name=model_name)
    proto_vec_all = l2_normalize(proto_vec_all)

    eval_texts = [e["text"] for e in events]
    eval_vec = embed_texts_array(eval_texts, model_name=model_name)
    eval_vec = l2_normalize(eval_vec)

    proto_vectors_by_role: dict[str, np.ndarray] = {}
    proto_id_lists: dict[str, list[str]] = {}
    cursor = 0
    for role in ROLES:
        size = sizes[role]
        proto_vectors_by_role[role] = proto_vec_all[cursor:cursor + size]
        proto_id_lists[role] = [p["id"] for p in prototypes[role]]
        cursor += size

    role_to_idx = {role: i for i, role in enumerate(ROLES)}
    y_true = np.array([role_to_idx[e["role"]] for e in events])

    # Single per-role shuffle gives NESTED prototype subsets for K-saturation.
    # K=10 is a superset of K=7 is a superset of K=5, etc. -- isolates the
    # effect of K from prototype-draw variance.
    rng_root = np.random.default_rng(args.seed)
    rng_shuffle, rng_boot = rng_root.spawn(2)
    proto_orders = shuffle_proto_orders(sizes, rng_shuffle)
    boot_rng = rng_boot

    k_results: dict[int, dict] = {}
    for k in args.k_list:
        if any(k > sz for sz in sizes.values()):
            raise RuntimeError(f"K={k} exceeds prototype pool size in some role: {sizes}")
        indices_per_role = {role: proto_orders[role][:k] for role in ROLES}
        result = classify_with_indices(
            proto_vectors_by_role, eval_vec, indices_per_role=indices_per_role
        )
        result["used_proto_ids_idx"] = indices_per_role
        y_pred = result["predicted_idx"]
        f1 = macro_f1(y_true, y_pred, len(ROLES))
        pc = per_class_recall(y_true, y_pred, len(ROLES))
        cm = confusion_matrix(y_true, y_pred, len(ROLES))
        accuracy = float((y_pred == y_true).mean())

        margins_by_class: dict[str, dict[str, float]] = {}
        for c, role in enumerate(ROLES):
            mask = y_true == c
            if mask.any():
                margins = result["margin"][mask]
                margins_by_class[role] = {
                    "median": float(np.median(margins)),
                    "q25": float(np.percentile(margins, 25)),
                    "q75": float(np.percentile(margins, 75)),
                    "min": float(margins.min()),
                    "max": float(margins.max()),
                    "n": int(mask.sum()),
                }

        k_results[k] = {
            "macro_f1_point": f1,
            "accuracy": accuracy,
            "per_class_recall": dict(zip(ROLES, pc)),
            "confusion_matrix": cm,
            "margin_by_class": margins_by_class,
            "used_proto_ids": {
                role: [proto_id_lists[role][i] for i in result["used_proto_ids_idx"][role]]
                for role in ROLES
            },
            "y_pred": y_pred.tolist(),
            "_full_result": result,
        }
        logger.info("K=%d macro-F1=%.3f accuracy=%.3f per-class=%s",
                    k, f1, accuracy, [f"{p:.3f}" for p in pc])

    primary = k_results.get(args.k_primary)
    if primary is None:
        raise RuntimeError(
            f"k_primary={args.k_primary} not in k_list={args.k_list}"
        )
    primary_pred = np.array(primary["y_pred"])
    boot = bootstrap_macro_f1(
        y_true, primary_pred,
        n_classes=len(ROLES),
        n_resamples=args.bootstrap_resamples,
        rng=boot_rng,
    )
    decision = decide(boot["lo"], list(primary["per_class_recall"].values()))

    write_classification_csv(
        args.output_dir / "phase1_classification.csv",
        events,
        primary["_full_result"],
        role_to_idx,
    )

    import vstash
    summary = {
        "embedding_model": model_name,
        "embedding_model_overridden": args.embed_model is not None,
        "embedding_dim": int(eval_vec.shape[1]),
        "vstash_version": getattr(vstash, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "seed": args.seed,
        "k_primary": args.k_primary,
        "k_list": list(args.k_list),
        "bootstrap_resamples": args.bootstrap_resamples,
        "n_events": len(events),
        "class_balance": {
            role: sum(1 for e in events if e["role"] == role) for role in ROLES
        },
        "prototypes_path": str(args.prototypes),
        "prototypes_sha256": hashlib.sha256(args.prototypes.read_bytes()).hexdigest(),
        "eval_path": str(args.eval_path),
        "eval_sha256": hashlib.sha256(args.eval_path.read_bytes()).hexdigest(),
        "k_saturation": {
            str(k): {
                "macro_f1": k_results[k]["macro_f1_point"],
                "accuracy": k_results[k]["accuracy"],
                "per_class_recall": k_results[k]["per_class_recall"],
            }
            for k in args.k_list
        },
        "primary": {
            "k": args.k_primary,
            "macro_f1_bootstrap": boot,
            "accuracy": primary["accuracy"],
            "per_class_recall": primary["per_class_recall"],
            "confusion_matrix": primary["confusion_matrix"],
            "margin_by_class": primary["margin_by_class"],
            "used_proto_ids": primary["used_proto_ids"],
        },
        "decision": decision,
    }

    def _scrub(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_scrub(x) for x in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    (args.output_dir / "phase1_metrics.json").write_text(
        json.dumps(_scrub(summary), indent=2, allow_nan=False)
    )

    print("\n=== #39 Phase 1 verdict ===")
    print(f"output dir     : {args.output_dir}")
    print(f"embedding model: {model_name}")
    print(f"seed           : {args.seed}")
    print(f"K primary      : {args.k_primary}")
    print(f"\nK-saturation curve (macro-F1):")
    for k in sorted(args.k_list):
        f1 = k_results[k]["macro_f1_point"]
        acc = k_results[k]["accuracy"]
        print(f"  K={k:<3} macro-F1={f1:.3f} accuracy={acc:.3f}")
    print(f"\nPrimary K={args.k_primary} bootstrap macro-F1: "
          f"mean={boot['mean']:.3f} CI=[{boot['lo']:.3f}, {boot['hi']:.3f}]")
    print(f"Per-class recall:")
    for role, pc in primary["per_class_recall"].items():
        print(f"  {role:<14} {pc:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred): {ROLES}")
    cm = primary["confusion_matrix"]
    for r_idx, role in enumerate(ROLES):
        row = " ".join(f"{cm[r_idx][c]:>3}" for c in range(len(ROLES)))
        print(f"  {role:<14} {row}")
    print(f"\nDecision      : {decision}")
    print("===========================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
