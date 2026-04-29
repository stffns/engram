"""AC#2 extension test for issue #47 (closed NO_GO 2026-04-29).

Status: this script ran on 2026-04-29 with N=200 (100 LoCoMo
per-session + 100 LME per-turn), oracle gpt-oss-120b on Cerebras.
Result: pooled in-taxonomy macro-F1 = 0.471 vs the AC#2 0.85 gate
-> FAIL. See ``notes/2026-04-29-issue-47-ac2-failure.md`` and
``runs/extension_20260429_083338/`` for the original artifacts.

Preserved as a reproducer. The classifier is constructed from the
research-only obs_pref bundle at
``experiments/role_markers/bundles_research_only/role_prototypes_obs_pref.json``
via the public ``RoleClassifier(prototypes=...)`` constructor --
not via a ``default(taxonomy=...)`` factory, since #47 closed
NO_GO and the taxonomy registry was reverted.

Procedure
---------
1. Build two streams of ingestables at the merken-canonical
   granularity for each benchmark:
   - LoCoMo: per-session (date_time + speaker:text turns)
   - LME-oracle: per-turn (role: content)
2. Deterministically sample 100 from each pool (seed=42).
3. Oracle-label every sampled text with gpt-oss-120b on Cerebras
   into one of {observation, preference, neither}. Labels saved
   incrementally so a mid-run crash does not lose API spend.
4. Classify the same texts with the obs_pref ``RoleClassifier``.
5. Compute the AC#2 gates:
   - macro-F1 on the in-taxonomy subset (oracle != neither)
     >= 0.85.
   - At most 10% of "neither" items receive a high-confidence
     (margin > 0.5) tag.

Cost guard
----------
- 200 calls at ~1500 prompt tokens / ~250 completion tokens.
- Cerebras gpt-oss-120b at posted rates: a few dollars at most.
- Resume-from-checkpoint via ``--checkpoint`` (default: per-run
  dir). Re-running the same checkpoint dir resumes from the last
  saved label; unrelated runs use a fresh ``runs/<timestamp>``.

Usage
-----
    python -m experiments.role_markers.extension_test_obs_pref \
        --n-locomo 100 --n-lme 100 [--dry-run-first 5]

The ``--dry-run-first`` flag oracle-labels and classifies the
first N items from each pool only, prints metrics, then exits.
Always run the dry-run before committing to the full run.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.midloop_concept.medlocal.cerebras_midloop import cerebras_chat
from experiments.retrieval.oracle_health import (
    OracleHealthError,
    OracleHealthGuard,
    is_oracle_error,
)
from merken.role_classifier import RoleClassifier

# Same pattern canonical longmemeval.runner._format_turn uses to strip
# tiktoken special tokens (e.g. <|endoftext|>) that appear in some LME
# haystacks and crash vstash chunking. We mirror it here so the
# classifier sees the exact text production would ingest.
_LME_SPECIAL_TOKENS = re.compile(r"<\|[a-z_]+\|>")

LOCOMO_PATH = REPO_ROOT / "experiments/retrieval/locomo/data/locomo10.json"
LME_PATH = REPO_ROOT / "experiments/retrieval/longmemeval/.cache/longmemeval_oracle.json"
OBS_PREF_PROTOTYPES = (
    REPO_ROOT
    / "experiments/role_markers/bundles_research_only/role_prototypes_obs_pref.json"
)

ORACLE_MODEL = "gpt-oss-120b"
# gpt-oss-120b is a reasoning model: it spends most of its budget on
# hidden reasoning before producing the visible JSON. A 200-token cap
# (the canonical setting for non-reasoning oracles like
# qwen-3-235b-a22b-instruct-2507) leaves the model truncated mid-
# reasoning with empty content -- observed empirically in a 5-per-pool
# dry run, 8/10 calls returned no JSON. 4096 leaves comfortable
# headroom for the reasoning trace plus the ~50-token JSON.
ORACLE_MAX_TOKENS = 4096
ORACLE_TEMPERATURE = 0.0  # deterministic-ish; Cerebras still has tail noise.

# AC#2 thresholds.
GATE_MACRO_F1 = 0.85
GATE_NEITHER_HIGH_CONF_FRACTION = 0.10
HIGH_CONF_MARGIN = 0.5

# Sampling determinism.
SAMPLE_SEED = 42

ORACLE_LABELS = ("observation", "preference", "neither")

ORACLE_SYSTEM = """You classify chat / dialogue text into exactly one of three labels:

- observation: a factual report of state, events, metrics, or what
  the speaker (or someone they describe) has experienced. Includes
  past events, biographical facts, numerical observations, and
  descriptive statements. The speaker is reporting what is the case,
  not what should be the case.

- preference: a subjective stance, recommendation, opinion, like /
  dislike, or expression of taste / desire. Includes value
  judgments, suggestions, and statements of "I prefer / I would
  rather / I recommend / my favorite is X." Subjective stance about
  a thing, not a factual claim about it.

- neither: anything that is not clearly an observation or
  preference. Includes pure questions, greetings, planning /
  instructions for future action, chitchat with no factual content,
  empty or trivial messages, system or assistant turns asking the
  user something, code snippets, or messages where the role is
  ambiguous between the two.

Composite turns: if a turn contains BOTH a factual report and a
preference, label by the dominant clause; if balanced, prefer
``observation``. A biographical fact stated with evaluative tone
(e.g. "My favorite cat, Mittens, has been sick for 3 weeks") is
``observation`` because the load-bearing clause is factual.

Respond ONLY with a JSON object of the form {"label": "<one of the
three labels>", "rationale": "<one short sentence>"}. Do not include
any prose outside the JSON object.
"""

ORACLE_USER_TEMPLATE = """Classify the following text.

TEXT:
{text}

Respond with the JSON object only.
"""


# ----------------------------------------------------------------- data builders

def build_locomo_per_session() -> list[tuple[str, str]]:
    with LOCOMO_PATH.open() as f:
        raw = json.load(f)
    items: list[tuple[str, str]] = []
    for entry in raw:
        sample_id = entry["sample_id"]
        conv = entry["conversation"]
        i = 1
        while f"session_{i}" in conv:
            dt = conv.get(f"session_{i}_date_time", "")
            turns = conv[f"session_{i}"]
            lines = [f"{t['speaker']}: {t['text']}" for t in turns]
            text = f"[{dt}]\n" + "\n".join(lines)
            items.append((text, f"{sample_id}::session_{i}"))
            i += 1
    return items


def build_lme_per_turn() -> list[tuple[str, str]]:
    if not LME_PATH.exists():
        raise RuntimeError(
            f"LME oracle cache not found at {LME_PATH}. "
            f"Run the canonical LME loader first."
        )
    with LME_PATH.open() as f:
        raw = json.load(f)
    items: list[tuple[str, str]] = []
    for q in raw:
        qid = q.get("question_id") or q.get("id")
        if not qid:
            raise RuntimeError(f"LME entry missing question_id: {q}")
        sessions = q.get("haystack_sessions") or []
        sids = q.get("haystack_session_ids") or [str(i) for i in range(len(sessions))]
        if len(sids) != len(sessions):
            raise RuntimeError(
                f"LME {qid}: haystack_session_ids ({len(sids)}) and "
                f"haystack_sessions ({len(sessions)}) length mismatch"
            )
        for sid, turns in zip(sids, sessions):
            if not isinstance(turns, list):
                continue
            for i, t in enumerate(turns):
                role = t.get("role", "?")
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                # Mirror canonical longmemeval.runner._format_turn:
                # strip tiktoken special tokens before ingest. Keeps
                # the AC#2 sample matched apples-to-apples with what
                # vstash would actually see in production.
                text = _LME_SPECIAL_TOKENS.sub("", f"{role}: {content}")
                items.append((text, f"{qid}::{sid}::{i}"))
    return items


def deterministic_sample(
    items: list[tuple[str, str]],
    n: int,
    *,
    pool_name: str,
) -> list[tuple[str, str]]:
    if n > len(items):
        raise ValueError(
            f"requested {n} samples from {pool_name} but pool has only "
            f"{len(items)} items"
        )
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(items, k=n)


# ----------------------------------------------------------------- oracle

def _parse_oracle_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        # Strip code-fence wrapper.
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"label": "_parse_error_", "rationale": "no JSON object", "raw": raw[:200]}
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return {"label": "_parse_error_", "rationale": str(exc), "raw": raw[:200]}
    label = obj.get("label", "_missing_label_")
    if label not in ORACLE_LABELS:
        return {
            "label": "_invalid_label_",
            "rationale": f"oracle returned {label!r}, not in {ORACLE_LABELS}",
            "raw": raw[:200],
        }
    return {"label": label, "rationale": obj.get("rationale", "")}


def oracle_label(text: str) -> dict:
    messages = [
        {"role": "system", "content": ORACLE_SYSTEM},
        {
            "role": "user",
            "content": ORACLE_USER_TEMPLATE.format(text=text[:4000]),
        },
    ]
    t0 = time.perf_counter()
    try:
        raw, dt, usage = cerebras_chat(
            ORACLE_MODEL,
            messages,
            ORACLE_MAX_TOKENS,
            temperature=ORACLE_TEMPERATURE,
        )
    except Exception as exc:  # noqa: BLE001 -- per-item fail-soft
        # Use the canonical 'oracle_error' marker so OracleHealthGuard
        # / is_oracle_error can spot transient outages and abort the
        # run before it silently corrupts AC#2 numbers.
        return {
            "label": "_oracle_error_",
            "rationale": f"oracle_error: {type(exc).__name__}: {exc}",
            "wall_s": time.perf_counter() - t0,
            "usage": {},
        }
    parsed = _parse_oracle_json(raw)
    parsed["wall_s"] = dt
    parsed["usage"] = usage or {}
    parsed["raw"] = raw[:1000]  # for forensic debugging
    return parsed


def _preflight_score(question: str, ground_truth: str, candidate: str) -> dict:
    """Adapter from OracleHealthGuard's score_fn(q, gt, ans) signature
    to our oracle_label(text) signature. The pre-flight only needs to
    return a verdict dict with a 'rationale' that is_oracle_error can
    inspect. We synthesise one trivial classification call.
    """
    verdict = oracle_label("My favorite color is blue.")
    return {
        "verdict": verdict.get("label", ""),
        "rationale": verdict.get("rationale", ""),
    }


# ----------------------------------------------------------------- metrics

def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> float:
    f1s: list[float] = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp == 0 or tp + fn == 0 or tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_recall(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in labels:
        n_true = sum(1 for t in y_true if t == c)
        if n_true == 0:
            out[c] = float("nan")
            continue
        out[c] = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c) / n_true
    return out


def confusion(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


# ----------------------------------------------------------------- run

def _save_checkpoint(
    path: Path,
    pool_name: str,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pool": pool_name, "rows": rows}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("rows", [])


def label_pool(
    pool_name: str,
    sampled: list[tuple[str, str]],
    checkpoint: Path,
    health: OracleHealthGuard,
) -> list[dict]:
    existing = _load_checkpoint(checkpoint)
    by_title = {r["title"]: r for r in existing}
    rows: list[dict] = []
    n_total = len(sampled)
    n_resumed = 0
    n_called = 0
    for i, (text, title) in enumerate(sampled, start=1):
        if title in by_title:
            rows.append(by_title[title])
            n_resumed += 1
            continue
        oracle = oracle_label(text)
        row = {
            "pool": pool_name,
            "title": title,
            "text": text,
            "char_len": len(text),
            "oracle": oracle,
        }
        rows.append(row)
        n_called += 1
        # Save after every call so a crash loses at most one in-flight call.
        _save_checkpoint(checkpoint, pool_name, rows)
        # Health-guard the live calls only (resumed rows were validated
        # in their original run). Adapt the dict to the rationale-bearing
        # shape the guard expects.
        health.record({"rationale": oracle.get("rationale", "")})
        if i % 10 == 0 or i == n_total:
            print(
                f"    [{pool_name}] {i}/{n_total} "
                f"(resumed={n_resumed}, called={n_called}, "
                f"oracle_errors={health.errors})"
            )
    return rows


def metrics_block(
    rows: list[dict],
    pool_name: str,
) -> dict:
    """Compute the AC#2 gate decomposition for one pool."""
    valid_rows = [r for r in rows if r["oracle"]["label"] in ORACLE_LABELS]
    bad_rows = [r for r in rows if r["oracle"]["label"] not in ORACLE_LABELS]
    in_taxonomy = [r for r in valid_rows if r["oracle"]["label"] != "neither"]
    neither = [r for r in valid_rows if r["oracle"]["label"] == "neither"]

    # In-taxonomy macro-F1 (binary obs vs pref).
    y_true = [r["oracle"]["label"] for r in in_taxonomy]
    y_pred = [r["pred_role"] for r in in_taxonomy]
    macro = macro_f1(y_true, y_pred, ("observation", "preference"))
    pcr = per_class_recall(y_true, y_pred, ("observation", "preference"))
    cm = confusion(y_true, y_pred, ("observation", "preference"))

    # "neither" high-confidence rate.
    high_conf_neither = [r for r in neither if r["confidence"] > HIGH_CONF_MARGIN]
    neither_high_conf_frac = (
        len(high_conf_neither) / len(neither) if neither else 0.0
    )

    return {
        "pool": pool_name,
        "n_rows": len(rows),
        "n_valid": len(valid_rows),
        "n_bad_oracle": len(bad_rows),
        "n_in_taxonomy": len(in_taxonomy),
        "n_neither": len(neither),
        "in_taxonomy_macro_f1": macro,
        "in_taxonomy_per_class_recall": pcr,
        "in_taxonomy_confusion": cm,
        "neither_high_conf_count": len(high_conf_neither),
        "neither_high_conf_fraction": neither_high_conf_frac,
        "char_len_min": min(r["char_len"] for r in rows) if rows else None,
        "char_len_median": (
            sorted(r["char_len"] for r in rows)[len(rows) // 2] if rows else None
        ),
        "char_len_max": max(r["char_len"] for r in rows) if rows else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-locomo", type=int, default=100)
    ap.add_argument("--n-lme", type=int, default=100)
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory; defaults to a fresh runs/extension_<ts>/."
    )
    ap.add_argument(
        "--dry-run-first", type=int, default=0,
        help="If >0, label only this many items per pool and exit (smoke test).",
    )
    args = ap.parse_args()

    if "CEREBRAS_API_KEY" not in os.environ:
        print("CEREBRAS_API_KEY not set", file=sys.stderr)
        return 2

    out_dir = args.out_dir or (
        REPO_ROOT / "experiments/role_markers/runs"
        / f"extension_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out: {out_dir}")

    print("Building ingest streams...")
    locomo_pool = build_locomo_per_session()
    lme_pool = build_lme_per_turn()
    print(f"  LoCoMo per-session pool: {len(locomo_pool)}")
    print(f"  LME per-turn pool:       {len(lme_pool)}")

    n_locomo = min(args.dry_run_first or args.n_locomo, len(locomo_pool))
    n_lme = min(args.dry_run_first or args.n_lme, len(lme_pool))
    print(f"  sampling: locomo={n_locomo}, lme={n_lme}")

    locomo_sample = deterministic_sample(locomo_pool, n_locomo, pool_name="locomo")
    lme_sample = deterministic_sample(lme_pool, n_lme, pool_name="lme")

    print("\nLoading obs_pref prototypes (research-only bundle)...")
    raw_protos = json.loads(OBS_PREF_PROTOTYPES.read_text())
    prototypes = {role: list(items) for role, items in raw_protos["roles"].items()}
    if set(prototypes.keys()) != {"observation", "preference"}:
        print(
            f"prototype JSON has unexpected roles: {sorted(prototypes.keys())}",
            file=sys.stderr,
        )
        return 4

    # Build the embedder the same way RoleClassifier.default() would,
    # then construct the classifier with our 2-class prototypes.
    from vstash.config import EmbeddingsConfig
    from vstash.embed import embed_texts as _embed_texts
    model_name = EmbeddingsConfig().model
    if not model_name:
        print("vstash EmbeddingsConfig().model is empty", file=sys.stderr)
        return 5

    def _embed(texts: list[str], model: str) -> np.ndarray:
        arr = np.asarray(_embed_texts(texts, model_name=model), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise RuntimeError(
                f"embed_texts returned shape {arr.shape!r} for {len(texts)} inputs"
            )
        return arr

    clf = RoleClassifier(
        embed_fn=_embed,
        model_name=model_name,
        prototypes=prototypes,
    )
    print(f"  embedder: {clf.model_name}")
    print(f"  roles:    {clf.roles}")
    print(f"  protos:   {clf.prototype_count}")

    def classify_pool(sample: list[tuple[str, str]]) -> list[tuple[str, float]]:
        results = clf.classify_batch([t for t, _ in sample])
        return [(r.role, r.confidence) for r in results]

    locomo_pred = classify_pool(locomo_sample)
    lme_pred = classify_pool(lme_sample)

    print("\nOracle pre-flight (gpt-oss-120b on Cerebras)...")
    health = OracleHealthGuard(
        first_window=10,
        max_errors_in_first=2,
        max_total_pct=0.05,
        min_total_for_pct_check=30,
    )
    try:
        health.preflight(_preflight_score)
    except OracleHealthError as exc:
        print(f"oracle pre-flight FAILED: {exc}", file=sys.stderr)
        return 3
    print("  pre-flight OK")

    print("\nOracle-labeling (gpt-oss-120b on Cerebras)...")

    locomo_rows = label_pool(
        "locomo_per_session",
        locomo_sample,
        out_dir / "locomo_labels.json",
        health,
    )
    for row, (role, conf) in zip(locomo_rows, locomo_pred):
        row["pred_role"] = role
        row["confidence"] = conf

    lme_rows = label_pool(
        "lme_per_turn",
        lme_sample,
        out_dir / "lme_labels.json",
        health,
    )
    for row, (role, conf) in zip(lme_rows, lme_pred):
        row["pred_role"] = role
        row["confidence"] = conf

    print("\nMetrics:")
    locomo_block = metrics_block(locomo_rows, "locomo_per_session")
    lme_block = metrics_block(lme_rows, "lme_per_turn")

    pooled_rows = locomo_rows + lme_rows
    pooled_block = metrics_block(pooled_rows, "pooled")

    summary = {
        "config": {
            "n_locomo": n_locomo,
            "n_lme": n_lme,
            "oracle_model": ORACLE_MODEL,
            "oracle_temperature": ORACLE_TEMPERATURE,
            "sample_seed": SAMPLE_SEED,
            "high_conf_margin": HIGH_CONF_MARGIN,
            "gate_macro_f1": GATE_MACRO_F1,
            "gate_neither_high_conf_fraction": GATE_NEITHER_HIGH_CONF_FRACTION,
            "classifier_model": clf.model_name,
            "classifier_roles": list(clf.roles),
        },
        "per_pool": [locomo_block, lme_block],
        "pooled": pooled_block,
    }

    # AC#2 verdict on the pooled cell.
    f1 = pooled_block["in_taxonomy_macro_f1"]
    nf = pooled_block["neither_high_conf_fraction"]
    passed_f1 = f1 >= GATE_MACRO_F1
    passed_nf = nf <= GATE_NEITHER_HIGH_CONF_FRACTION
    summary["ac2_verdict"] = {
        "macro_f1": {"value": f1, "gate": GATE_MACRO_F1, "passed": passed_f1},
        "neither_high_conf_fraction": {
            "value": nf,
            "gate": GATE_NEITHER_HIGH_CONF_FRACTION,
            "passed": passed_nf,
        },
        "overall": "PASS" if (passed_f1 and passed_nf) else "FAIL",
    }

    summary_path = out_dir / "extension_test_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    rows_path = out_dir / "all_rows.json"
    rows_path.write_text(json.dumps(pooled_rows, indent=2))

    for blk in (locomo_block, lme_block, pooled_block):
        print(
            f"\n  pool={blk['pool']:<20s} n_in_taxonomy={blk['n_in_taxonomy']:>3d} "
            f"n_neither={blk['n_neither']:>3d} "
            f"n_bad_oracle={blk['n_bad_oracle']:>3d}"
        )
        print(
            f"    macro-F1 (in-taxonomy)        = {blk['in_taxonomy_macro_f1']:.3f}"
        )
        print(
            f"    neither high-conf (>{HIGH_CONF_MARGIN}) frac = "
            f"{blk['neither_high_conf_fraction']:.3f} "
            f"({blk['neither_high_conf_count']}/{blk['n_neither']})"
        )
        print(f"    per-class recall: {blk['in_taxonomy_per_class_recall']}")

    print("\nAC#2 verdict (pooled):")
    if pooled_block["n_in_taxonomy"] < 20:
        print(
            f"  WARNING: only {pooled_block['n_in_taxonomy']} in-taxonomy rows; "
            f"macro-F1 gate is not statistically meaningful below n=20."
        )
    print(
        f"  macro-F1 {f1:.3f} >= {GATE_MACRO_F1}: "
        f"{'PASS' if passed_f1 else 'FAIL'}"
    )
    print(
        f"  neither high-conf fraction {nf:.3f} <= "
        f"{GATE_NEITHER_HIGH_CONF_FRACTION}: "
        f"{'PASS' if passed_nf else 'FAIL'}"
    )
    print(f"  overall: {summary['ac2_verdict']['overall']}")

    if not health.summary_safe:
        print()
        print(health.warning())
        summary["ac2_verdict"]["overall"] = "CONTAMINATED"
        summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nresults -> {summary_path}")
    return 0 if summary["ac2_verdict"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
