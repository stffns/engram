# ruff: noqa: I001, E402
"""H10 interaction-term leave-one-out ablation.

Jay's paper review (2026-04-18) pointed out that H10 shipped with
five interaction terms motivated by one smoke case and never
ablated individually. This script runs a leave-one-out sweep: refit
the head with each of the 5 interaction features removed, compare
test ECE.

Features dropped one at a time:
  - short_X_numbers
  - short_X_file_paths
  - short_X_inline_code
  - short_X_markdown_table
  - ood_X_short

All fits at C=1 on the same 80/20 split as H10. If no single
interaction is load-bearing (all leave-one-out ECEs stay within
0.005 of the full-head ECE), that's evidence the set is redundant
and fewer features would do. If one dominates (e.g. ood_X_short),
that's the real signal; the others are decoration.
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

FULL_FEATURES = [
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

INTERACTION_NAMES = [
    "short_X_numbers",
    "short_X_file_paths",
    "short_X_inline_code",
    "short_X_markdown_table",
    "ood_X_short",
]


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def parse_pd(s):
    m = re.search(r"P\(D\)=(\d+\.\d+|\d+)", s or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def featurize(text, is_ood):
    has_code_fence = 1 if CODE_FENCE.search(text) else 0
    has_inline_code = 1 if INLINE_CODE.search(text) else 0
    has_markdown_table = 1 if TABLE_ROW.search(text) else 0
    has_numbers = 1 if len(NUMBER.findall(text)) >= 3 else 0
    has_file_paths = 1 if FILE_PATH.search(text) else 0
    is_short = 1 if len(text) < 300 else 0
    is_long = 1 if len(text) >= 1000 else 0
    return {
        "has_code_fence": has_code_fence,
        "has_inline_code": has_inline_code,
        "has_markdown_table": has_markdown_table,
        "has_numbers": has_numbers,
        "has_file_paths": has_file_paths,
        "is_short": is_short,
        "is_long": is_long,
        "is_ood": is_ood,
        "short_X_numbers": is_short * has_numbers,
        "short_X_file_paths": is_short * has_file_paths,
        "short_X_inline_code": is_short * has_inline_code,
        "short_X_markdown_table": is_short * has_markdown_table,
        "ood_X_short": is_ood * is_short,
    }


def build_vector(p_raw, feats, feature_list):
    vec = []
    for name in feature_list:
        if name == "logit_P(D)":
            vec.append(logit(p_raw))
        else:
            vec.append(feats[name])
    return vec


def collect(decider):
    ctx = WriteContext(project="h10ablation")
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
                feats = featurize(text, is_ood)
                rows.append((p, feats, y))
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


def fit_and_eval(rows, feature_list, C=1.0, seed=42):
    rng = random.Random(seed)
    pos = [r for r in rows if r[2] == 1]
    neg = [r for r in rows if r[2] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    sp = int(0.8 * len(pos))
    sn = int(0.8 * len(neg))
    train = pos[:sp] + neg[:sn]
    test = pos[sp:] + neg[sn:]
    rng.shuffle(train)
    rng.shuffle(test)

    X_train = np.array([build_vector(p, f, feature_list) for (p, f, _) in train])
    y_train = np.array([y for (_, _, y) in train])
    X_test = np.array([build_vector(p, f, feature_list) for (p, f, _) in test])

    lr = LogisticRegression(C=C, fit_intercept=True, max_iter=5000).fit(X_train, y_train)
    p_test = lr.predict_proba(X_test)[:, 1]
    pairs = [(float(p_test[i]), test[i][2]) for i in range(len(test))]
    return ece(pairs), float(lr.intercept_[0]), dict(zip(feature_list, [float(c) for c in lr.coef_[0]]))


def main() -> int:
    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit("set MERKEN_SHADOW_NANOGPT_CKPT / _META")
    v7 = NanoGPTWriteDecider(ckpt, meta)
    rows = collect(v7)
    print(f"n={len(rows)}")

    full_ece, _, full_coefs = fit_and_eval(rows, FULL_FEATURES, C=1.0)
    print(f"\nFULL head (all 14 features) ECE = {full_ece:.4f}")

    print(f"\nLeave-one-out (drop each interaction, refit, measure delta):")
    print(f"{'dropped':<28} {'ECE':>7} {'delta_vs_full':>15}  dropped_feature_weight_in_full")
    results = {"full_ece": full_ece, "full_coefficients": full_coefs, "loo": {}}
    for dropped in INTERACTION_NAMES:
        features = [f for f in FULL_FEATURES if f != dropped]
        e, _, _ = fit_and_eval(rows, features, C=1.0)
        delta = e - full_ece
        w_in_full = full_coefs.get(dropped, 0.0)
        print(f"{dropped:<28} {e:>6.4f}  {delta:>+7.4f}          {w_in_full:+.3f}")
        results["loo"][dropped] = {"ece": e, "delta_vs_full": delta, "weight_in_full": w_in_full}

    # No-interactions baseline (H11-style, 9 features)
    no_inter = [f for f in FULL_FEATURES if f not in INTERACTION_NAMES]
    e, _, _ = fit_and_eval(rows, no_inter, C=1.0)
    delta = e - full_ece
    print(f"\nNO interactions (9 features, H11-style)")
    print(f"  ECE = {e:.4f}  delta_vs_full = {delta:+.4f}")
    results["no_interactions"] = {"ece": e, "delta_vs_full": delta}

    # Verdict
    largest_effect = max(results["loo"].items(), key=lambda kv: abs(kv[1]["delta_vs_full"]))
    print(f"\nLargest single effect: {largest_effect[0]} (delta {largest_effect[1]['delta_vs_full']:+.4f})")

    out_path = REPO / "experiments" / "nanogpt" / "h10_interaction_ablation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
