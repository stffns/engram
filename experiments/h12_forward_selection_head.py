# ruff: noqa: I001, E402
"""H12 -- forward selection for the minimal stable calibration head.

H10 shipped 5 interaction features; the leave-one-out ablation
(H10_ablation) showed only `ood_X_short` is meaningfully load-bearing
(+0.0085 ECE when dropped), and `short_X_file_paths` actively hurts
(-0.0043 when dropped). Jay asked: what if we start from just
`ood_X_short` and add interactions one at a time?

Greedy forward selection from the 9 base features + `ood_X_short`:

    step 0: 9 base + ood_X_short          (10 features, baseline)
    step 1: add each of the other 4 interactions, pick the one with
            largest ECE drop if any drop >= 0.002
    step 2: repeat with 4 remaining
    ...

Stop when no remaining interaction drops ECE by >= 0.002.
Report the winning sequence + final ECE vs:
  - H11 (9 base, no interactions)
  - H10 full (9 base + 5 interactions)
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

BASE_FEATURES = [
    "logit_P(D)",
    "has_code_fence", "has_inline_code", "has_markdown_table",
    "has_numbers", "has_file_paths",
    "is_short", "is_long", "is_ood",
]

INTERACTIONS = [
    "short_X_numbers",
    "short_X_file_paths",
    "short_X_inline_code",
    "short_X_markdown_table",
    "ood_X_short",
]

MIN_IMPROVEMENT = 0.002  # delta threshold to accept a new interaction


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


def featurize_full(text, is_ood):
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


def build_row(p, feats, feature_list):
    out = []
    for f in feature_list:
        if f == "logit_P(D)":
            out.append(logit(p))
        else:
            out.append(feats[f])
    return out


def collect(decider):
    ctx = WriteContext(project="h12")
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
                p = parse_pd(r.get("shadow_reason"))
                if p is None:
                    d = decider.decide(Event(text=text), ctx)
                    p = parse_pd(d.reason)
                if p is None:
                    continue
                rows.append((p, featurize_full(text, is_ood), y))
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

    X_train = np.array([build_row(p, f, feature_list) for (p, f, _) in train])
    y_train = np.array([y for (_, _, y) in train])
    X_test = np.array([build_row(p, f, feature_list) for (p, f, _) in test])

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

    # Baselines
    h11_ece, _, _ = fit_and_eval(rows, BASE_FEATURES, C=1.0)
    h10_ece, _, _ = fit_and_eval(rows, BASE_FEATURES + INTERACTIONS, C=1.0)
    print(f"baseline H11 (9 base) ECE     = {h11_ece:.4f}")
    print(f"baseline H10 (9 base + 5 int) = {h10_ece:.4f}")

    # Seed forward selection with ood_X_short (known best single).
    selected = ["ood_X_short"]
    current = BASE_FEATURES + selected
    cur_ece, _, _ = fit_and_eval(rows, current, C=1.0)
    print(f"\nseeded: 9 base + ood_X_short  ECE = {cur_ece:.4f}")

    history = [
        {"step": 0, "added": "ood_X_short (seed)", "features": current[:], "ece": cur_ece},
    ]

    remaining = [i for i in INTERACTIONS if i not in selected]
    step = 1
    while remaining:
        trials = {}
        for cand in remaining:
            cand_list = current + [cand]
            e, _, _ = fit_and_eval(rows, cand_list, C=1.0)
            trials[cand] = e
        best_cand = min(trials, key=lambda k: trials[k])
        best_ece = trials[best_cand]
        delta = best_ece - cur_ece
        print(f"\nstep {step}: candidates")
        for c, e in sorted(trials.items(), key=lambda kv: kv[1]):
            print(f"  + {c:<26} ECE={e:.4f}  delta={e - cur_ece:+.4f}")
        if delta <= -MIN_IMPROVEMENT:
            print(f"  ACCEPT {best_cand} (delta {delta:+.4f} <= -{MIN_IMPROVEMENT})")
            selected.append(best_cand)
            current.append(best_cand)
            cur_ece = best_ece
            remaining.remove(best_cand)
            history.append({
                "step": step,
                "added": best_cand,
                "features": current[:],
                "ece": cur_ece,
            })
            step += 1
        else:
            print(f"  STOP (best delta {delta:+.4f} does not clear threshold {MIN_IMPROVEMENT})")
            break

    print(f"\n=== Result ===")
    print(f"Final head: {len(current)} features")
    print(f"Interactions kept: {selected}")
    print(f"Final ECE = {cur_ece:.4f}")
    print(f"  vs H11 (no interactions):  delta = {cur_ece - h11_ece:+.4f}")
    print(f"  vs H10 (all 5 interactions): delta = {cur_ece - h10_ece:+.4f}")

    out = {
        "h11_ece": h11_ece,
        "h10_ece": h10_ece,
        "history": history,
        "final_features": current,
        "final_ece": cur_ece,
        "kept_interactions": selected,
    }
    out_path = REPO / "experiments" / "nanogpt" / "h12_forward_selection.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
