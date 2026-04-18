"""v7 calibration curve: does P(D) track oracle DEC-fraction?

A classifier is "calibrated" when its predicted probability matches
the empirical frequency of the positive class. If v7 predicts
P(D)=0.5 on a batch of events and the oracle actually labels 50% of
them as DECISION, v7 is perfectly calibrated on that bin.

Calibration is a stronger claim than "uncertainty recognition". A
model can have bimodal outputs (always 0 or 1) while still being
uncertain on ambiguous content -- it just happens to flip a coin
instead of producing P(D)=0.5. True calibration requires the output
probability to MEAN something about actual frequency.

This script:
  1. Joins v7's P(D) with Gemini's DECISION/NOISE label for each of
     the 1520 real labeled events (1026 shadow_skip + 494 agree_write).
  2. Bins events by v7 P(D) in 10 buckets [0.0-0.1, 0.1-0.2, ...].
  3. For each bin: computes actual DEC fraction (Gemini-labeled) and
     compares to bin midpoint (v7 predicted).
  4. Prints calibration table + ECE (expected calibration error).

ECE < 0.10: reasonably calibrated.
ECE < 0.05: well calibrated.
ECE >= 0.15: model output probabilities are unreliable.
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
LABELS_SKIP = REPO / "data" / "merken_labels_v7.jsonl"
LABELS_WRITE = REPO / "data" / "merken_labels_agree_write.jsonl"


def parse_pd(s: str) -> float | None:
    m = re.search(r"P\(D\)=([\d.]+)", s or "")
    return float(m.group(1)) if m else None


def iter_labels_with_pd(path: Path, decider: NanoGPTWriteDecider, force_reinfer: bool):
    ctx = WriteContext(project="eval")
    if not path.exists():
        return
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        label = r.get("label")
        if label not in ("DECISION", "NOISE"):
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue

        p = None
        if not force_reinfer:
            p = parse_pd(r.get("shadow_reason"))
        if p is None:
            d = decider.decide(Event(text=text), ctx)
            p = parse_pd(d.reason)
        if p is None:
            continue
        yield p, label


def main() -> int:
    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit(
            "Set MERKEN_SHADOW_NANOGPT_CKPT / MERKEN_SHADOW_NANOGPT_META"
        )
    v7 = NanoGPTWriteDecider(ckpt, meta)

    pairs: list[tuple[float, str]] = []
    pairs.extend(iter_labels_with_pd(LABELS_SKIP, v7, force_reinfer=False))
    pairs.extend(iter_labels_with_pd(LABELS_WRITE, v7, force_reinfer=True))

    print(f"Joined {len(pairs)} events with v7 P(D) + Gemini label.\n")

    # 10-bin calibration table
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    ece = 0.0
    total = len(pairs)
    print(f"{'bin':<12} {'n':>6} {'mean_P(D)':>12} {'actual_DEC':>13} {'gap':>8}")
    print("-" * 60)
    for lo, hi in bins:
        in_bin = [(p, l) for p, l in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            print(f"[{lo:.1f}, {hi:.1f})   {n:>6}   -            -           -")
            continue
        mean_pd = sum(p for p, _ in in_bin) / n
        actual_dec = sum(1 for _, l in in_bin if l == "DECISION") / n
        gap = abs(mean_pd - actual_dec)
        ece += (n / total) * gap
        stars_pred = "P" + "=" * int(mean_pd * 30)
        stars_act = "A" + "#" * int(actual_dec * 30)
        print(
            f"[{lo:.1f}, {hi:.1f})   {n:>6}   {mean_pd:>10.3f}   "
            f"{actual_dec:>11.3f}   {gap:>6.3f}"
        )
        print(f"                             pred: {stars_pred}")
        print(f"                             act:  {stars_act}")

    print()
    print(f"Expected Calibration Error (ECE): {ece:.3f}")
    if ece < 0.05:
        print("  -> WELL CALIBRATED")
    elif ece < 0.10:
        print("  -> REASONABLY CALIBRATED")
    elif ece < 0.15:
        print("  -> MILDLY MISCALIBRATED")
    else:
        print("  -> POORLY CALIBRATED (outputs are unreliable probabilities)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
