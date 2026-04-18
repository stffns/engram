# ruff: noqa: I001, E402
"""Ablation: does the H10 head recover v6's calibration?

Jay's paper review (2026-04-18) noted that contribution #2 claims
"uncertainty detection is training-driven, not architectural" but
the evidence (v6 vs v7 ECE gap) confounds two variables: different
training data AND different training dynamics. To support the claim
we need to measure v6 WITH the same H10 head architecture. If v6+head
still doesn't show uncertainty (high ECE / bimodal distribution),
the training-data-only explanation is stronger.

Test: run v6 on the SAME 982 clean-pool events used to fit H10, fit
an H10-style head on v6 outputs, and compare to the head fit on v7
outputs.

Expected outcomes:
  - v6+head ECE ~= v7+head ECE (0.035): training data didn't matter,
    the head alone recovers calibration for any model. Our claim #2
    is WRONG.
  - v6+head ECE >> v7+head ECE: the v6 representation lacks the
    signal the head needs. Training data matters. Claim #2 stands.
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
NANOGPT = Path(
    os.environ.get("NANOGPT_REPO")
    or (REPO.parent / "nanoGPT")
)

V6_CKPT = NANOGPT / "out-merken-bpe-v6" / "ckpt.pt"
V6_META = NANOGPT / "data" / "merken_bpe_v6" / "meta.pkl"
V7_CKPT = NANOGPT / "out-merken-bpe-v7" / "ckpt.pt"
V7_META = NANOGPT / "data" / "merken_bpe_v7" / "meta.pkl"

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
    return [
        has_code_fence, has_inline_code, has_markdown_table, has_numbers,
        has_file_paths, is_short, is_long, is_ood,
        is_short * has_numbers,
        is_short * has_file_paths,
        is_short * has_inline_code,
        is_short * has_markdown_table,
        is_ood * is_short,
    ]


def collect(decider, label):
    ctx = WriteContext(project=f"ablation_{label}")
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
                lab = r.get("label")
                if lab == "DECISION":
                    y = 1
                elif lab == "NOISE":
                    y = 0
                else:
                    continue
                text = (r.get("text") or "").strip()
                if not text:
                    continue
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


def fit_and_eval(rows, C=1.0):
    rng = random.Random(42)
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    sp = int(0.8 * len(pos))
    sn = int(0.8 * len(neg))
    train = pos[:sp] + neg[:sn]
    test = pos[sp:] + neg[sn:]
    rng.shuffle(train)
    rng.shuffle(test)

    X_train = np.array([r[0] for r in train])
    y_train = np.array([r[1] for r in train])
    X_test = np.array([r[0] for r in test])

    raw_test = [(r[2], r[1]) for r in test]
    raw_ece = ece(raw_test)

    lr = LogisticRegression(C=C, fit_intercept=True, max_iter=5000).fit(X_train, y_train)
    p_test = lr.predict_proba(X_test)[:, 1]
    head_test = [(float(p_test[i]), test[i][1]) for i in range(len(test))]
    head_ece = ece(head_test)

    # bimodality of raw
    bimodal = sum(1 for p, _ in raw_test if p < 0.1 or p >= 0.9) / len(raw_test)
    middle = sum(1 for p, _ in raw_test if 0.3 <= p < 0.7) / len(raw_test)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "raw_ece": raw_ece,
        "head_ece": head_ece,
        "head_delta": head_ece - raw_ece,
        "bimodal_frac": bimodal,
        "middle_frac": middle,
        "intercept": float(lr.intercept_[0]),
        "coefficients": dict(zip(FEATURE_NAMES, [float(c) for c in lr.coef_[0]])),
    }


def main() -> int:
    print(f"loading v6: {V6_CKPT}")
    v6 = NanoGPTWriteDecider(str(V6_CKPT), str(V6_META))
    print(f"loading v7: {V7_CKPT}")
    v7 = NanoGPTWriteDecider(str(V7_CKPT), str(V7_META))

    print("collecting v6 outputs on clean pool...")
    rows_v6 = collect(v6, "v6")
    print(f"  n={len(rows_v6)}")
    print("collecting v7 outputs on clean pool...")
    rows_v7 = collect(v7, "v7")
    print(f"  n={len(rows_v7)}")

    print("\n=== v6 + H10-style head (fit on v6 outputs) ===")
    res_v6 = fit_and_eval(rows_v6, C=1.0)
    print(f"  raw_ece     = {res_v6['raw_ece']:.3f}")
    print(f"  head_ece    = {res_v6['head_ece']:.3f}")
    print(f"  head_delta  = {res_v6['head_delta']:+.3f}")
    print(f"  bimodal_frac (<0.1 or >=0.9) = {res_v6['bimodal_frac']:.3f}")
    print(f"  middle_frac  (0.3..0.7)     = {res_v6['middle_frac']:.3f}")

    print("\n=== v7 + H10 head (fit on v7 outputs, our baseline) ===")
    res_v7 = fit_and_eval(rows_v7, C=1.0)
    print(f"  raw_ece     = {res_v7['raw_ece']:.3f}")
    print(f"  head_ece    = {res_v7['head_ece']:.3f}")
    print(f"  head_delta  = {res_v7['head_delta']:+.3f}")
    print(f"  bimodal_frac = {res_v7['bimodal_frac']:.3f}")
    print(f"  middle_frac  = {res_v7['middle_frac']:.3f}")

    print("\n=== verdict ===")
    delta = res_v6["head_ece"] - res_v7["head_ece"]
    if delta > 0.02:
        verdict = (
            "TRAINING MATTERS: v6+head ECE is materially higher than "
            "v7+head. The head cannot compensate for v6's bimodal "
            "representation. Claim #2 (uncertainty is training-driven) "
            "survives."
        )
    elif delta < -0.005:
        verdict = (
            "INVERTED: v6+head beats v7+head. Claim #2 is wrong; the "
            "head is doing the work, not v7's training."
        )
    else:
        verdict = (
            "NEAR PARITY: v6+head ~ v7+head. The head alone recovers "
            "calibration regardless of training. Claim #2 is weaker "
            "than stated; the right framing is 'calibration is a "
            "post-hoc fit that works on either base model'."
        )
    print(verdict)

    out = {
        "v6": res_v6,
        "v7": res_v7,
        "delta_v6_minus_v7_head_ece": delta,
        "verdict": verdict,
    }
    out_path = REPO / "experiments" / "nanogpt" / "v6_with_h10_head_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
