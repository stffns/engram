"""V2 SCR classifier subset probe at production shape.

The 4-class V3-prod run gave macro-F1 0.794 with INV->OBS collision
(recall 0.40) and SCR bleeding 20% to OBS. OBS and PREF were 100%
recall in the 4-class setting. This probe asks the orthogonal
question: if we restrict the taxonomy to the roles whose markers
DO survive at production shape, do we recover a production-grade
classifier?

Two configurations:
- ``2-class``: {observation, preference}. The two roles whose markers
  pass cleanly in 4-class.
- ``3-class``: {state_change_report, observation, preference}. Drops
  the broken role (INV) but keeps SCR; tests whether SCR's bleed to
  OBS was a 4-class artifact or a real failure.

Reuses prototypes_v3_production.json and eval_v3_production.json
unchanged. Same prototype-mean-cosine classification, same K=5
primary, same bootstrap CI on macro-F1, 3 seeds.

Run: ``python -m notes.2026-04-28-role-classifier-subset-probe``
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PROTOTYPES_PATH = Path("experiments/role_markers/prototypes_v3_production.json")
EVAL_PATH = Path("experiments/role_markers/eval_v3_production.json")
SEEDS = (42, 43, 44)
K_PRIMARY = 5
BOOTSTRAP_RESAMPLES = 1000

CONFIGS = {
    "2-class_obs_pref": ("observation", "preference"),
    "3-class_drop_inv": ("state_change_report", "observation", "preference"),
    "4-class_full_baseline": (
        "state_change_report", "investigation", "observation", "preference",
    ),
}


def resolve_embedder() -> str:
    from vstash.config import EmbeddingsConfig
    model = EmbeddingsConfig().model
    if not model:
        raise RuntimeError("vstash EmbeddingsConfig().model empty")
    return model


def embed_array(texts: list[str], model_name: str) -> np.ndarray:
    from vstash.embed import embed_texts
    arr = np.asarray(embed_texts(texts, model_name=model_name), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise RuntimeError(f"embed shape {arr.shape!r} for {len(texts)}")
    return arr


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s: list[float] = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0 or tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    out: list[float] = []
    for c in range(n_classes):
        n_true = int((y_true == c).sum())
        if n_true == 0:
            out.append(float("nan"))
            continue
        out.append(int(((y_pred == c) & (y_true == c)).sum()) / n_true)
    return out


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[list[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    by_class = [np.where(y_true == c)[0] for c in range(n_classes)]
    if any(arr.size == 0 for arr in by_class):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    resamples = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = np.concatenate([rng.choice(arr, size=arr.size, replace=True) for arr in by_class])
        resamples[b] = macro_f1(y_true[idx], y_pred[idx], n_classes)
    return {
        "mean": macro_f1(y_true, y_pred, n_classes),
        "lo": float(np.percentile(resamples, 2.5)),
        "hi": float(np.percentile(resamples, 97.5)),
    }


def run_config(
    config_name: str,
    roles: tuple[str, ...],
    prototypes_raw: dict,
    events_raw: list[dict],
    model_name: str,
    seed: int,
) -> dict:
    proto_texts: list[str] = []
    proto_role: list[str] = []
    sizes: dict[str, int] = {}
    for role in roles:
        items = prototypes_raw["roles"][role]
        sizes[role] = len(items)
        for p in items:
            proto_texts.append(p["text"])
            proto_role.append(role)

    events = [e for e in events_raw if e["role"] in roles]
    eval_texts = [e["text"] for e in events]
    role_to_idx = {role: i for i, role in enumerate(roles)}
    y_true = np.array([role_to_idx[e["role"]] for e in events])

    proto_vec = l2_normalize(embed_array(proto_texts, model_name))
    eval_vec = l2_normalize(embed_array(eval_texts, model_name))

    by_role_indices: dict[str, list[int]] = {role: [] for role in roles}
    for i, role in enumerate(proto_role):
        by_role_indices[role].append(i)

    rng_root = np.random.default_rng(seed)
    rng_shuffle, rng_boot = rng_root.spawn(2)
    proto_orders = {role: list(by_role_indices[role]) for role in roles}
    for role in roles:
        rng_shuffle.shuffle(proto_orders[role])

    k = min(K_PRIMARY, *(sizes[r] for r in roles))
    n_classes = len(roles)
    role_means = np.zeros((eval_vec.shape[0], n_classes), dtype=np.float64)
    for r_idx, role in enumerate(roles):
        idx = proto_orders[role][:k]
        sub = proto_vec[idx]
        role_means[:, r_idx] = (eval_vec @ sub.T).mean(axis=1)
    y_pred = np.argmax(role_means, axis=1)

    f1 = macro_f1(y_true, y_pred, n_classes)
    pc = per_class_recall(y_true, y_pred, n_classes)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    boot = bootstrap_macro_f1_ci(
        y_true, y_pred, n_classes=n_classes,
        n_resamples=BOOTSTRAP_RESAMPLES, rng=rng_boot,
    )

    return {
        "config": config_name,
        "roles": list(roles),
        "n_events": int(eval_vec.shape[0]),
        "k": k,
        "seed": seed,
        "macro_f1_point": f1,
        "macro_f1_bootstrap": boot,
        "accuracy": float((y_pred == y_true).mean()),
        "per_class_recall": dict(zip(roles, pc)),
        "confusion_matrix": cm,
    }


def main() -> int:
    print(f"prototypes: {PROTOTYPES_PATH}")
    print(f"eval:       {EVAL_PATH}")
    prototypes_raw = json.loads(PROTOTYPES_PATH.read_text())
    events_raw = json.loads(EVAL_PATH.read_text())["events"]
    model_name = resolve_embedder()
    print(f"embedder:   {model_name}")
    print(f"K primary:  {K_PRIMARY}")
    print(f"seeds:      {list(SEEDS)}")

    out_dir = Path("experiments/role_markers/runs") / f"subset_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for config_name, roles in CONFIGS.items():
        print(f"\n=== {config_name}  roles={roles}")
        for seed in SEEDS:
            r = run_config(
                config_name, roles, prototypes_raw, events_raw,
                model_name, seed,
            )
            all_results.append(r)
            print(
                f"  seed={seed}  macro-F1={r['macro_f1_point']:.3f} "
                f"CI=[{r['macro_f1_bootstrap']['lo']:.3f}, "
                f"{r['macro_f1_bootstrap']['hi']:.3f}]  "
                f"acc={r['accuracy']:.3f}  "
                f"per-class={ {k: f'{v:.2f}' for k, v in r['per_class_recall'].items()} }"
            )
            print(f"    confusion (rows=true, cols=pred):")
            for ri, role in enumerate(roles):
                row = " ".join(f"{r['confusion_matrix'][ri][ci]:>3}" for ci in range(len(roles)))
                print(f"      {role:<20s} {row}")

    # 3-seed aggregation per config.
    print("\n=== 3-seed aggregation ===")
    by_config: dict[str, list[dict]] = {}
    for r in all_results:
        by_config.setdefault(r["config"], []).append(r)
    summary: list[dict] = []
    for config_name, rows in by_config.items():
        f1s = [r["macro_f1_point"] for r in rows]
        los = [r["macro_f1_bootstrap"]["lo"] for r in rows]
        his = [r["macro_f1_bootstrap"]["hi"] for r in rows]
        per_class_3seed: dict[str, list[float]] = {}
        for r in rows:
            for role, val in r["per_class_recall"].items():
                per_class_3seed.setdefault(role, []).append(val)
        agg = {
            "config": config_name,
            "roles": rows[0]["roles"],
            "n_events": rows[0]["n_events"],
            "k": rows[0]["k"],
            "macro_f1_3seed_mean": float(np.mean(f1s)),
            "macro_f1_3seed_stdev": float(np.std(f1s, ddof=1)),
            "macro_f1_per_seed": f1s,
            "ci_lo_per_seed": los,
            "ci_hi_per_seed": his,
            "per_class_recall_3seed_mean": {
                role: float(np.mean(vals)) for role, vals in per_class_3seed.items()
            },
        }
        summary.append(agg)
        print(
            f"  {config_name:<28s} macro-F1 3-seed = {agg['macro_f1_3seed_mean']:.3f} "
            f"+- {agg['macro_f1_3seed_stdev']:.3f}  "
            f"per-seed CI lo: {[f'{x:.3f}' for x in los]}"
        )
        for role, mean_recall in agg["per_class_recall_3seed_mean"].items():
            print(f"    {role:<20s} recall (3-seed mean) = {mean_recall:.3f}")

    (out_dir / "subset_results.json").write_text(
        json.dumps({"per_seed": all_results, "aggregated": summary}, indent=2)
    )
    print(f"\nresults -> {out_dir / 'subset_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
