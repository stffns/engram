# ruff: noqa: I001, E402
"""H10 -- interaction terms in the conditional calibration head.

H3 smoke tested the H11-chosen head on a short DEC like:
  "Fix: replaced pgbouncer session mode with transaction mode in
   db/pool.py line 42. Latency p95 180ms -> 40ms." (129 chars)

The head pushed it DOWN to P_cal=0.371 even though raw was 0.668.
Reason: w_short = -1.82 fires because len<300, and the additive
model cannot distinguish "short with numbers/file_paths" (concrete
content) from "short with no structure" (filler).

This script extends the feature vector with 5 interaction terms:
  is_short * has_numbers
  is_short * has_file_paths
  is_short * has_inline_code
  is_short * has_markdown_table
  is_ood  * is_short

If the learned head assigns POSITIVE weight to e.g. (is_short *
has_numbers), that term cancels the negative is_short alone when
the event also has numbers -- exactly what we want for concrete
short decisions.

Runs with the same C-sweep protocol as H11 on the CLEAN pool.
"""

from __future__ import annotations

import torch  # noqa: F401

import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies.types import Event, WriteContext


REPO = Path(__file__).resolve().parent.parent

SOURCES_CLEAN = [
    (REPO / "data" / "merken_labels_agree_write.jsonl", 0),
    (REPO / "data" / "merken_labels_ldjnr_capybara.jsonl", 1),
    (REPO / "data" / "merken_labels_slimorca.jsonl", 1),
]

EPS = 1e-6

CODE_FENCE = re.compile(r"```")
INLINE_CODE = re.compile(r"`[^`\n]{2,}`")
TABLE_ROW = re.compile(r"\|[^\n]*\|[^\n]*\|")
FILE_PATH = re.compile(r"\b\S+\.(?:py|js|ts|md|json|toml|yml|yaml|go|rs|sh|sql)\b")
NUMBER = re.compile(r"\d")

FEATURE_NAMES = [
    "logit_P(D)",
    "has_code_fence",
    "has_inline_code",
    "has_markdown_table",
    "has_numbers",
    "has_file_paths",
    "is_short",
    "is_long",
    "is_ood",
    "short_X_numbers",
    "short_X_file_paths",
    "short_X_inline_code",
    "short_X_markdown_table",
    "ood_X_short",
]


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def parse_pd(s: str) -> float | None:
    m = re.search(r"P\(D\)=(\d+\.\d+|\d+)", s or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def featurize(text: str, is_ood: int) -> list[int]:
    has_code_fence = 1 if CODE_FENCE.search(text) else 0
    has_inline_code = 1 if INLINE_CODE.search(text) else 0
    has_markdown_table = 1 if TABLE_ROW.search(text) else 0
    has_numbers = 1 if len(NUMBER.findall(text)) >= 3 else 0
    has_file_paths = 1 if FILE_PATH.search(text) else 0
    is_short = 1 if len(text) < 300 else 0
    is_long = 1 if len(text) >= 1000 else 0
    return [
        has_code_fence,
        has_inline_code,
        has_markdown_table,
        has_numbers,
        has_file_paths,
        is_short,
        is_long,
        is_ood,
        # Interactions:
        is_short * has_numbers,
        is_short * has_file_paths,
        is_short * has_inline_code,
        is_short * has_markdown_table,
        is_ood * is_short,
    ]


def collect(decider):
    ctx = WriteContext(project="h10")
    rows = []
    for path, is_ood in SOURCES_CLEAN:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                label = r.get("label")
                if label == "DECISION":
                    y = 1
                elif label == "NOISE":
                    y = 0
                else:
                    continue
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                p = parse_pd(r.get("shadow_reason"))
                if p is None:
                    d = decider.decide(Event(text=text), ctx)
                    p = parse_pd(d.reason)
                if p is None:
                    continue
                x = [logit(p)] + featurize(text, is_ood)
                rows.append((x, y, p, text))
    return rows


def ece(pairs, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    N = len(pairs)
    if N == 0:
        return 0.0
    e = 0.0
    for lo, hi in bins:
        in_bin = [(p, y) for p, y in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            continue
        conf = sum(p for p, _ in in_bin) / n
        acc = sum(y for _, y in in_bin) / n
        e += (n / N) * abs(conf - acc)
    return e


def main() -> int:
    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit("set MERKEN_SHADOW_NANOGPT_CKPT / _META")
    v7 = NanoGPTWriteDecider(ckpt, meta)

    rows = collect(v7)
    rng = random.Random(42)
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    split_pos = int(0.8 * len(pos))
    split_neg = int(0.8 * len(neg))
    train = pos[:split_pos] + neg[:split_neg]
    test = pos[split_pos:] + neg[split_neg:]
    rng.shuffle(train)
    rng.shuffle(test)
    print(f"train={len(train)}  test={len(test)}  features={len(FEATURE_NAMES)}")

    X_train = np.array([r[0] for r in train])
    y_train = np.array([r[1] for r in train])
    X_test = np.array([r[0] for r in test])

    print(f"\n{'C':>8}  {'ECE':>7}  {'max|w|':>8}  notable_weights")
    print("-" * 100)
    results = []
    for C in [0.1, 1.0, 10.0, 100.0, 1e6]:
        lr = LogisticRegression(C=C, fit_intercept=True, max_iter=5000).fit(X_train, y_train)
        coefs = lr.coef_[0]
        intercept = float(lr.intercept_[0])
        p_test = lr.predict_proba(X_test)[:, 1]
        pairs = [(float(p_test[i]), test[i][1]) for i in range(len(test))]
        e = ece(pairs)
        max_w_abs = float(max(abs(c) for c in coefs))
        notable = " ".join(
            f"{FEATURE_NAMES[i]}={coefs[i]:+.2f}"
            for i in range(len(coefs))
            if abs(coefs[i]) > 0.5
        )
        print(f"{C:>8g}  {e:>5.3f}  {max_w_abs:>+7.2f}  {notable}")
        results.append({
            "C": C,
            "ece": e,
            "intercept": intercept,
            "coefficients": dict(zip(FEATURE_NAMES, [float(c) for c in coefs])),
            "max_abs_weight": max_w_abs,
        })

    # Compare best to H11 baseline (ECE 0.043)
    best = min(results, key=lambda r: r["ece"])
    print(f"\nBest: C={best['C']:g}  ECE={best['ece']:.3f}")
    print(f"H11 baseline ECE: 0.043")
    delta = best["ece"] - 0.043
    print(f"delta vs H11: {delta:+.3f}")
    if delta < -0.002:
        verdict = "ACCEPTED (interactions help)"
    elif delta > 0.005:
        verdict = "REJECTED (interactions hurt)"
    else:
        verdict = "NEUTRAL (interactions don't help noticeably)"
    print(f"verdict: {verdict}")

    # Smoke test: the "Fix: replaced pgbouncer..." case
    smoke_text = (
        "Fix: replaced pgbouncer session mode with transaction mode in "
        "db/pool.py line 42. Latency p95 180ms -> 40ms."
    )
    # raw from v7
    ctx = WriteContext(project="smoke")
    smoke_d = v7.decide(Event(text=smoke_text), ctx)
    smoke_p = parse_pd(smoke_d.reason)
    if smoke_p is not None:
        smoke_x = [logit(smoke_p)] + featurize(smoke_text, is_ood=0)
        for r in results:
            if r["C"] == best["C"]:
                # reconstruct the chosen lr
                chosen_lr = LogisticRegression(C=best["C"], fit_intercept=True, max_iter=5000).fit(X_train, y_train)
                smoke_p_cal = float(chosen_lr.predict_proba(np.array([smoke_x]))[0, 1])
                print(
                    f"\nSmoke test ('Fix: replaced pgbouncer...'):\n"
                    f"  raw P(D) = {smoke_p:.3f}\n"
                    f"  H11 head P_cal would be around 0.371 (prior smoke)\n"
                    f"  H10 head P_cal = {smoke_p_cal:.3f}"
                )
                break

    out = {
        "sweep": results,
        "best": best,
        "h11_baseline_ece": 0.043,
        "delta_vs_h11": delta,
        "verdict": verdict,
    }
    out_path = REPO / "experiments" / "nanogpt" / "h10_interaction_head.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
