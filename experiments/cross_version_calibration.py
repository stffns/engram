"""Compare v6 and v7 calibration on the same 1520 oracled events.

We already know v7's calibration profile (ECE ~0.12 contaminated,
~0.05 after temperature on clean data). Question: was v6 also
miscalibrated, or did real-label training in v7 introduce the bias?

This script runs v6 and v7 back-to-back on every (text, oracle_label)
pair from the 1520 labels and produces side-by-side reliability
diagrams + ECE numbers.

Caveats:
  - v6 was trained on the v6 data (synthetic + organic + markdown-NOISE
    only). It never saw the 1026 Jay-transcript labels. So for v6,
    ALL 1520 pairs are held-out; no contamination.
  - v7 saw the 1026 skip-set texts during training. For v7, the clean
    subset is the 494 agree_write pairs (as in calibrate_v7_platt.py).

We report v6 on full 1520, v7 on full 1520 (contaminated-partially),
and v7 on clean 494 -- so the reader can see how contamination
affects the numbers.
"""

from __future__ import annotations

import torch  # noqa: F401

import json
import os
import re
from pathlib import Path

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies.types import Event, WriteContext


REPO = Path("/Users/jaysonsteffens/Desktop/Personal/Projects/engram")
NANOGPT = Path("/Users/jaysonsteffens/Desktop/Personal/Projects/nanoGPT")
LABELS_SKIP = REPO / "data" / "merken_labels_v7.jsonl"
LABELS_WRITE = REPO / "data" / "merken_labels_agree_write.jsonl"

V6_CKPT = NANOGPT / "out-merken-bpe-v6" / "ckpt.pt"
V6_META = NANOGPT / "data" / "merken_bpe_v6" / "meta.pkl"
V7_CKPT = NANOGPT / "out-merken-bpe-v7" / "ckpt.pt"
V7_META = NANOGPT / "data" / "merken_bpe_v7" / "meta.pkl"


def parse_pd(s: str) -> float | None:
    m = re.search(r"P\(D\)=([\d.]+)", s or "")
    return float(m.group(1)) if m else None


def collect(decider, labels_path: Path, has_shadow_reason: bool):
    ctx = WriteContext(project="cross")
    out = []
    if not labels_path.exists():
        return out
    for line in labels_path.open():
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
        # For v6 we must always re-infer (no pre-logged shadow_reason
        # from a v6 bootstrap exists); for v7 on skip-set we can use
        # the logged value since the bootstrap's shadow WAS v7.
        p = None
        if has_shadow_reason:
            p = parse_pd(r.get("shadow_reason"))
        if p is None:
            d = decider.decide(Event(text=text), ctx)
            p = parse_pd(d.reason)
        if p is None:
            continue
        out.append((p, y))
    return out


def ece(pairs, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total = len(pairs)
    e = 0.0
    signed = 0.0
    for lo, hi in bins:
        in_bin = [(p, y) for p, y in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            continue
        conf = sum(p for p, _ in in_bin) / n
        acc = sum(y for _, y in in_bin) / n
        e += (n / total) * abs(conf - acc)
        signed += (n / total) * (conf - acc)
    return e, signed


def report(title, pairs):
    if not pairs:
        print(f"{title}: n=0")
        return
    e, signed = ece(pairs)
    mean_pd = sum(p for p, _ in pairs) / len(pairs)
    dec_share = sum(y for _, y in pairs) / len(pairs)
    print(f"\n{title}")
    print(f"  n={len(pairs)}  mean_P(D)={mean_pd:.3f}  true_DEC_share={dec_share:.3f}")
    print(f"  ECE={e:.3f}  signed_bias={signed:+.3f}")


def bin_table(title, pairs, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total = len(pairs)
    print(f"\n{title}")
    print(f"  {'bin':<12} {'n':>5} {'conf':>7} {'acc':>7} {'gap':>7}")
    for lo, hi in bins:
        in_bin = [(p, y) for p, y in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            continue
        conf = sum(p for p, _ in in_bin) / n
        acc = sum(y for _, y in in_bin) / n
        gap = conf - acc
        print(f"  [{lo:.1f}, {hi:.1f})   {n:>5}   {conf:>5.3f}   {acc:>5.3f}  {gap:>+6.3f}")


def main() -> int:
    print(f"loading v6 from {V6_CKPT}")
    v6 = NanoGPTWriteDecider(str(V6_CKPT), str(V6_META))
    print(f"loading v7 from {V7_CKPT}")
    v7 = NanoGPTWriteDecider(str(V7_CKPT), str(V7_META))

    print("\n=== v6 on ALL 1520 pairs (fully held-out for v6) ===")
    v6_skip = collect(v6, LABELS_SKIP, has_shadow_reason=False)
    v6_write = collect(v6, LABELS_WRITE, has_shadow_reason=False)
    v6_all = v6_skip + v6_write
    report("v6 ALL", v6_all)

    print("\n=== v7 on ALL 1520 pairs (1026 contaminated, 494 clean) ===")
    v7_skip = collect(v7, LABELS_SKIP, has_shadow_reason=True)
    v7_write = collect(v7, LABELS_WRITE, has_shadow_reason=False)
    v7_all = v7_skip + v7_write
    report("v7 ALL (mixed)", v7_all)
    report("v7 clean (agree_write only, 494)", v7_write)

    bin_table("v6 reliability diagram (n=1520)", v6_all)
    bin_table("v7 reliability diagram -- full (n=1520)", v7_all)
    bin_table("v7 reliability diagram -- clean (n=494)", v7_write)

    # Save numbers for archival
    out_path = REPO / "experiments" / "nanogpt" / "cross_version_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "v6": {
            "n": len(v6_all),
            "ece": ece(v6_all)[0],
            "signed_bias": ece(v6_all)[1],
            "mean_pd": sum(p for p, _ in v6_all) / len(v6_all) if v6_all else 0,
        },
        "v7_full": {
            "n": len(v7_all),
            "ece": ece(v7_all)[0],
            "signed_bias": ece(v7_all)[1],
            "mean_pd": sum(p for p, _ in v7_all) / len(v7_all) if v7_all else 0,
        },
        "v7_clean": {
            "n": len(v7_write),
            "ece": ece(v7_write)[0],
            "signed_bias": ece(v7_write)[1],
            "mean_pd": sum(p for p, _ in v7_write) / len(v7_write) if v7_write else 0,
        },
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved numbers to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
