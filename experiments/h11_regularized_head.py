# ruff: noqa: I001, E402
"""H11 -- L2 regularization sweep on the conditional calibration head.

H1 clean rerun produced a head with two alarmingly extreme weights:
  has_markdown_table = +8.33  (n_test=13)
  is_long            = +7.00  (n_test=15)

Those slices are small and the C=1e6 Logistic fit has essentially no
regularization, so the weights fit the exact split perfectly but
generalize poorly. Likely they'd regress on a different 80/20 seed
or on new data.

This script sweeps `C` (inverse regularization strength) across
[0.01, 0.1, 1.0, 10, 100, 1e6] and reports for each:
  - test ECE
  - largest absolute weight
  - signed bias

We accept the smallest C (strongest regularization) whose ECE stays
within 0.003 of the best fit AND whose max weight is <= 5.
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


def collect(decider):
    ctx = WriteContext(project="h11")
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
                x = [logit(p)] + featurize(text) + [is_ood]
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
    print(f"train={len(train)}  test={len(test)}")

    X_train = np.array([r[0] for r in train])
    y_train = np.array([r[1] for r in train])
    X_test = np.array([r[0] for r in test])

    print(f"\n{'C':>8}  {'ECE':>7}  {'max|w|':>8}  {'worst_feature':<22}  weights_summary")
    print("-" * 100)
    results = []
    for C in [0.01, 0.1, 1.0, 10.0, 100.0, 1e6]:
        lr = LogisticRegression(C=C, fit_intercept=True, max_iter=5000).fit(X_train, y_train)
        coefs = lr.coef_[0]
        intercept = float(lr.intercept_[0])
        p_test = lr.predict_proba(X_test)[:, 1]
        pairs = [(float(p_test[i]), test[i][1]) for i in range(len(test))]
        e = ece(pairs)
        max_w_idx = int(np.argmax(np.abs(coefs)))
        worst = FEATURE_NAMES[max_w_idx]
        max_w = float(coefs[max_w_idx])
        summary = " ".join(
            f"{n}={coefs[i]:+.2f}"
            for i, n in enumerate(FEATURE_NAMES)
            if abs(coefs[i]) > 0.5
        )
        print(f"{C:>8g}  {e:>5.3f}  {max_w:>+7.2f}  {worst:<22}  {summary}")
        results.append({
            "C": C,
            "ece": e,
            "intercept": intercept,
            "coefficients": dict(zip(FEATURE_NAMES, [float(c) for c in coefs])),
            "max_abs_weight": abs(max_w),
            "worst_feature": worst,
        })

    # Pick recommended C: smallest (strongest reg) whose ECE stays
    # within 0.003 of the best AND max |w| <= 5.
    best_ece = min(r["ece"] for r in results)
    candidates = [r for r in results if r["ece"] <= best_ece + 0.003 and r["max_abs_weight"] <= 5.0]
    candidates.sort(key=lambda r: r["C"])  # lowest C first
    if candidates:
        chosen = candidates[0]
        print(f"\nChosen: C={chosen['C']:g}  ECE={chosen['ece']:.3f}  max|w|={chosen['max_abs_weight']:.2f}")
    else:
        chosen = min(results, key=lambda r: r["ece"])
        print(f"\nNo regularized fit meets both criteria. Best-ECE: C={chosen['C']:g}")

    out = {
        "sweep": results,
        "chosen": chosen,
        "decision_rule": "min_C with ece <= best+0.003 AND max|w| <= 5.0",
    }
    out_path = REPO / "experiments" / "nanogpt" / "h11_regularized_head.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
