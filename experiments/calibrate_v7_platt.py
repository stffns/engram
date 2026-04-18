"""Post-hoc calibration of v7 via temperature scaling + Platt scaling.

Starting point (from calibration_curve_v7.py):
  ECE = 0.120, signed bias +0.090 (v7 over-confident in DEC).

Two standard post-hoc fixes:

1. Temperature scaling: learn a single scalar T >= 0 and transform
      logit  -> logit / T
      P_cal  = sigmoid(logit / T)
   T > 1 pulls all predictions toward 0.5 (softer), T < 1 sharpens.
   One parameter -- very hard to overfit.

2. Platt scaling: learn (a, b) and transform
      P_cal = sigmoid(a * logit + b)
   Two parameters -- slightly more expressive, can handle asymmetric
   miscalibration (e.g., over-confident in one class but not the
   other).

Both are fitted on a held-out split of the (P(D), oracle_label) data
so the scaling does not memorize v7's raw outputs.

We report ECE pre and post scaling on the held-out TEST split. If
post-scaling ECE < 0.05 we declare calibration "fixed" without
retraining v7.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

import torch  # noqa: F401 (Mistake #10)
from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies.types import Event, WriteContext


REPO = Path("/Users/jaysonsteffens/Desktop/Personal/Projects/engram")
LABELS_SKIP = REPO / "data" / "merken_labels_v7.jsonl"
LABELS_WRITE = REPO / "data" / "merken_labels_agree_write.jsonl"

EPS = 1e-6


def parse_pd(s: str) -> float | None:
    m = re.search(r"P\(D\)=([\d.]+)", s or "")
    return float(m.group(1)) if m else None


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def collect_pairs(decider: NanoGPTWriteDecider) -> list[tuple[float, int]]:
    """Yield (P(D), oracle_label_01) pairs from both JSONL sets.

    oracle_label_01 = 1 if Gemini said DECISION, 0 if NOISE.
    """
    ctx = WriteContext(project="calib")
    pairs: list[tuple[float, int]] = []

    def ingest(path: Path, has_shadow_reason: bool):
        if not path.exists():
            return
        for line in path.open():
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
            p = parse_pd(r.get("shadow_reason")) if has_shadow_reason else None
            if p is None:
                d = decider.decide(Event(text=text), ctx)
                p = parse_pd(d.reason)
            if p is None:
                continue
            pairs.append((p, y))

    ingest(LABELS_SKIP, has_shadow_reason=True)
    ingest(LABELS_WRITE, has_shadow_reason=False)
    return pairs


def compute_ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> tuple[float, float]:
    """Return (ece, signed_bias)."""
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    ece = 0.0
    signed = 0.0
    N = len(pairs)
    for lo, hi in bins:
        in_bin = [(p, y) for p, y in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            continue
        conf = sum(p for p, _ in in_bin) / n
        acc = sum(y for _, y in in_bin) / n
        ece += (n / N) * abs(conf - acc)
        signed += (n / N) * (conf - acc)
    return ece, signed


def temperature_scale(pairs_train, pairs_test):
    """Fit T on train, apply to test, return (T, calibrated_test_pairs)."""
    def nll(T):
        loss = 0.0
        for p, y in pairs_train:
            z = logit(p) / max(T, EPS)
            q = sigmoid(z)
            q = min(max(q, EPS), 1 - EPS)
            loss -= y * math.log(q) + (1 - y) * math.log(1 - q)
        return loss
    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T = res.x
    cal_test = [(sigmoid(logit(p) / T), y) for p, y in pairs_test]
    return T, cal_test


def platt_scale(pairs_train, pairs_test):
    """Fit (a, b) via logistic regression on train, apply to test."""
    X = np.array([[logit(p)] for p, _ in pairs_train])
    y = np.array([y for _, y in pairs_train])
    lr = LogisticRegression(C=1e6, fit_intercept=True)
    lr.fit(X, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    cal_test = [(sigmoid(a * logit(p) + b), y) for p, y in pairs_test]
    return (a, b), cal_test


def report_bin_deltas(pre, post, n_bins=10):
    """Print per-bin shift caused by calibration."""
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    print(f"\n{'bin':<12} {'n':>5} {'conf_pre':>10} {'conf_post':>11} {'acc':>7}")
    print("-" * 55)
    for lo, hi in bins:
        pre_b = [(p, y) for p, y in pre if lo <= p < (hi if hi < 1.0 else 1.01)]
        if not pre_b:
            continue
        post_b_vals = [post[i][0] for i in range(len(pre)) if lo <= pre[i][0] < (hi if hi < 1.0 else 1.01)]
        conf_pre = sum(p for p, _ in pre_b) / len(pre_b)
        conf_post = sum(post_b_vals) / len(post_b_vals)
        acc = sum(y for _, y in pre_b) / len(pre_b)
        print(
            f"[{lo:.1f}, {hi:.1f})   {len(pre_b):>5}   {conf_pre:>8.3f}   "
            f"{conf_post:>9.3f}   {acc:>5.3f}"
        )


def main() -> int:
    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit("Set MERKEN_SHADOW_NANOGPT_CKPT / MERKEN_SHADOW_NANOGPT_META")

    print("loading v7 + collecting (P(D), oracle) pairs...")
    v7 = NanoGPTWriteDecider(ckpt, meta)
    pairs = collect_pairs(v7)
    print(f"total pairs: {len(pairs)}")

    # Stratified 80/20 split to keep class balance in train/test.
    rng = random.Random(42)
    pos = [pr for pr in pairs if pr[1] == 1]
    neg = [pr for pr in pairs if pr[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    split_pos = int(0.8 * len(pos))
    split_neg = int(0.8 * len(neg))
    train = pos[:split_pos] + neg[:split_neg]
    test = pos[split_pos:] + neg[split_neg:]
    rng.shuffle(train)
    rng.shuffle(test)
    print(f"train: {len(train)}  (pos={split_pos} neg={split_neg})")
    print(f"test:  {len(test)}")

    # Baseline ECE on test
    ece_pre, signed_pre = compute_ece(test)
    print(f"\nBaseline (no calibration):")
    print(f"  ECE        = {ece_pre:.3f}")
    print(f"  signed_bias= {signed_pre:+.3f}")

    # Temperature scaling
    T, temp_test = temperature_scale(train, test)
    ece_temp, signed_temp = compute_ece(temp_test)
    print(f"\nTemperature scaling (T fit on train):")
    print(f"  T          = {T:.3f}")
    print(f"  ECE        = {ece_temp:.3f}  (delta {ece_temp - ece_pre:+.3f})")
    print(f"  signed_bias= {signed_temp:+.3f}")

    # Platt scaling
    (a, b), platt_test = platt_scale(train, test)
    ece_platt, signed_platt = compute_ece(platt_test)
    print(f"\nPlatt scaling (a, b fit on train via logistic regression):")
    print(f"  a          = {a:.3f}")
    print(f"  b          = {b:+.3f}")
    print(f"  ECE        = {ece_platt:.3f}  (delta {ece_platt - ece_pre:+.3f})")
    print(f"  signed_bias= {signed_platt:+.3f}")

    # Per-bin breakdown of the better of the two
    if ece_platt < ece_temp:
        winner = "Platt"
        winner_test = platt_test
    else:
        winner = "Temperature"
        winner_test = temp_test
    print(f"\nWinner: {winner}")
    report_bin_deltas(test, winner_test)

    # Save the fitted scaling so Memory can apply it
    out = {
        "baseline_ece": ece_pre,
        "baseline_signed_bias": signed_pre,
        "temperature": {"T": T, "ece": ece_temp, "signed_bias": signed_temp},
        "platt": {"a": a, "b": b, "ece": ece_platt, "signed_bias": signed_platt},
        "winner": winner,
        "n_train": len(train),
        "n_test": len(test),
    }
    out_path = REPO / "experiments" / "nanogpt" / "calibration_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved fitted params to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
