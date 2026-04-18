# ruff: noqa: I001, E402
"""H1 -- content-type-conditional calibration head.

A single-parameter scalar fix (Temperature) or 2-parameter scalar
fix (Platt) cannot correct opposite biases across structure types
(plain_prose over-confident, markdown_table under-confident).

A tiny head that uses (v7 logit + structure binary features) as
input and emits a calibrated P(D) should beat the scalar fixes.

Features:
  x1 = logit(v7 P(D))        (raw v7 confidence on the logit scale)
  x2 = has_code_fence        (binary)
  x3 = has_inline_code       (binary)
  x4 = has_markdown_table    (binary)
  x5 = has_numbers           (binary)
  x6 = has_file_paths        (binary)
  x7 = is_short              (binary, len < 300)
  x8 = is_long               (binary, len >= 1000)
  x9 = is_ood                (binary, 1 if from Capybara/SlimOrca;
                              0 if from Jay transcripts)

Model: logistic regression y = sigmoid(w . x + b), fit on the same
80/20 split used for the scalar Platt fit. Compared to Platt
(which only uses x1), this head can learn per-structure correction
vectors AND per-domain correction.

Decision rule: ECE improvement >= 0.03 AND no per-structure
regression above Jay's baseline ECE (aggregate 0.120).
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

# Clean label sources. NOTE: merken_labels_v7.jsonl is deliberately
# EXCLUDED by default because those 1026 shadow_skip texts were part
# of v7's training data (see nanoGPT/data/merken_bpe_v7/prepare.py
# `DECISION:transcript:v1` class). Fitting the calibration head on
# training data would measure memorization, not generalization, and
# the reported ECE gain would be inflated.
SOURCES_CLEAN = [
    (REPO / "data" / "merken_labels_agree_write.jsonl", 0),  # in-dist Jay, NOT in v7 train
    (REPO / "data" / "merken_labels_ldjnr_capybara.jsonl", 1),  # OOD public
    (REPO / "data" / "merken_labels_slimorca.jsonl", 1),
]

# Opt-in via --include-contaminated-skip to reproduce the original
# (flawed) run for historical comparison.
SOURCES_WITH_CONTAM = [
    (REPO / "data" / "merken_labels_v7.jsonl", 0),
] + SOURCES_CLEAN

EPS = 1e-6

CODE_FENCE = re.compile(r"```")
INLINE_CODE = re.compile(r"`[^`\n]{2,}`")
TABLE_ROW = re.compile(r"\|[^\n]*\|[^\n]*\|")
FILE_PATH = re.compile(r"\b\S+\.(?:py|js|ts|md|json|toml|yml|yaml|go|rs|sh|sql)\b")
NUMBER = re.compile(r"\d")


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


def featurize(text: str) -> list[int]:
    return [
        1 if CODE_FENCE.search(text) else 0,
        1 if INLINE_CODE.search(text) else 0,
        1 if TABLE_ROW.search(text) else 0,
        1 if len(NUMBER.findall(text)) >= 3 else 0,
        1 if FILE_PATH.search(text) else 0,
        1 if len(text) < 300 else 0,
        1 if len(text) >= 1000 else 0,
    ]


def collect(decider: NanoGPTWriteDecider, sources):
    """Return list of (X_row, y) plus text+tag info for diagnostics."""
    ctx = WriteContext(project="h1")
    rows = []
    for path, is_ood in sources:
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
                tags = featurize(text)
                x = [logit(p)] + tags + [is_ood]
                rows.append((x, y, p, tags, is_ood))
    return rows


def ece_signed(pairs, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    N = len(pairs)
    if N == 0:
        return 0.0, 0.0
    e = 0.0
    signed = 0.0
    for lo, hi in bins:
        in_bin = [(p, y) for p, y in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            continue
        conf = sum(p for p, _ in in_bin) / n
        acc = sum(y for _, y in in_bin) / n
        e += (n / N) * abs(conf - acc)
        signed += (n / N) * (conf - acc)
    return e, signed


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--include-contaminated-skip",
        action="store_true",
        help="Include merken_labels_v7.jsonl (v7 training data). Default"
        " excludes these so the head is fit on clean held-out events.",
    )
    args = ap.parse_args()

    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit("set MERKEN_SHADOW_NANOGPT_CKPT / _META")
    print(f"loading v7 from {ckpt}")
    v7 = NanoGPTWriteDecider(ckpt, meta)

    sources = SOURCES_WITH_CONTAM if args.include_contaminated_skip else SOURCES_CLEAN
    label = "WITH CONTAMINATED skip-set" if args.include_contaminated_skip else "CLEAN only"
    print(f"sources: {label}")
    rows = collect(v7, sources)
    print(f"total: {len(rows)} labeled rows")

    # Stratified 80/20 split on y (class-balance).
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
    print(f"train: {len(train)}  test: {len(test)}  "
          f"(DEC_train={split_pos}  DEC_test={len(pos) - split_pos})")

    # Baseline: raw v7 P(D) -> ECE
    test_raw = [(r[2], r[1]) for r in test]
    ece_pre, signed_pre = ece_signed(test_raw)
    print(f"\nBaseline raw v7 P(D) on test:")
    print(f"  ECE = {ece_pre:.3f}  signed_bias = {signed_pre:+.3f}")

    # Platt scalar on logit only
    X_train_logit = np.array([[r[0][0]] for r in train])
    y_train = np.array([r[1] for r in train])
    platt = LogisticRegression(C=1e6, fit_intercept=True).fit(X_train_logit, y_train)
    pa = float(platt.coef_[0][0])
    pb = float(platt.intercept_[0])
    test_platt = [(sigmoid(pa * r[0][0] + pb), r[1]) for r in test]
    ece_platt, signed_platt = ece_signed(test_platt)
    print(f"\nPlatt (logit only): a={pa:.3f} b={pb:+.3f}")
    print(f"  ECE = {ece_platt:.3f}  signed_bias = {signed_platt:+.3f}")

    # Conditional head: full feature vector
    X_train = np.array([r[0] for r in train])
    head = LogisticRegression(C=1e6, fit_intercept=True).fit(X_train, y_train)
    X_test = np.array([r[0] for r in test])
    p_cal = head.predict_proba(X_test)[:, 1]
    test_head = [(float(p_cal[i]), r[1]) for i, r in enumerate(test)]
    ece_head, signed_head = ece_signed(test_head)
    coefs = head.coef_[0]
    intercept = float(head.intercept_[0])
    feature_names = [
        "logit_P(D)",
        "has_code_fence",
        "has_inline_code",
        "has_markdown_table",
        "has_numbers",
        "has_file_paths",
        "is_short",
        "is_long",
        "is_ood",
    ]
    print(f"\nConditional head: intercept={intercept:+.3f}")
    for name, w in zip(feature_names, coefs):
        print(f"  w[{name:<20}] = {w:+.3f}")
    print(f"  ECE = {ece_head:.3f}  signed_bias = {signed_head:+.3f}")

    # Per-category ECE on test (head vs platt vs raw)
    print(f"\n{'category':<22} {'n':>5} {'raw_ece':>8} {'platt_ece':>10} {'head_ece':>10}")
    print("-" * 60)
    tag_names = [
        "has_code_fence",
        "has_inline_code",
        "has_markdown_table",
        "has_numbers",
        "has_file_paths",
        "is_short",
        "is_long",
    ]
    for ti, tname in enumerate(tag_names):
        idxs = [i for i, r in enumerate(test) if r[3][ti] == 1]
        if not idxs:
            continue
        raw_sub = [(test[i][2], test[i][1]) for i in idxs]
        platt_sub = [test_platt[i] for i in idxs]
        head_sub = [test_head[i] for i in idxs]
        e_raw, _ = ece_signed(raw_sub)
        e_platt, _ = ece_signed(platt_sub)
        e_head, _ = ece_signed(head_sub)
        print(f"{tname:<22} {len(idxs):>5}   {e_raw:>5.3f}     {e_platt:>5.3f}      {e_head:>5.3f}")
    # OOD slice
    idxs = [i for i, r in enumerate(test) if r[4] == 1]
    if idxs:
        raw_sub = [(test[i][2], test[i][1]) for i in idxs]
        platt_sub = [test_platt[i] for i in idxs]
        head_sub = [test_head[i] for i in idxs]
        e_raw, _ = ece_signed(raw_sub)
        e_platt, _ = ece_signed(platt_sub)
        e_head, _ = ece_signed(head_sub)
        print(f"{'is_ood':<22} {len(idxs):>5}   {e_raw:>5.3f}     {e_platt:>5.3f}      {e_head:>5.3f}")

    print(f"\n{'='*60}")
    print("Summary (test set):")
    print(f"  raw         ECE={ece_pre:.3f}  signed_bias={signed_pre:+.3f}")
    print(f"  platt       ECE={ece_platt:.3f}  signed_bias={signed_platt:+.3f}")
    print(f"  cond head   ECE={ece_head:.3f}  signed_bias={signed_head:+.3f}")
    decision_delta = ece_head - ece_platt
    verdict = "PASS" if decision_delta <= -0.03 else "FAIL"
    print(f"  head vs platt delta = {decision_delta:+.3f}  [{verdict}] "
          f"(needs <= -0.030 to accept H1)")

    out = {
        "n_train": len(train),
        "n_test": len(test),
        "raw": {"ece": ece_pre, "signed_bias": signed_pre},
        "platt": {"a": pa, "b": pb, "ece": ece_platt, "signed_bias": signed_platt},
        "head": {
            "intercept": intercept,
            "coefficients": dict(zip(feature_names, coefs.tolist())),
            "ece": ece_head,
            "signed_bias": signed_head,
        },
        "head_vs_platt_delta": decision_delta,
        "h1_verdict": verdict,
    }
    out_path = REPO / "experiments" / "nanogpt" / "h1_calibration_head.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
