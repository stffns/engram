# ruff: noqa: I001, E402
"""Does v6 show the same cross-distribution drift as v7?

v7 produced LOW P(D) on Capybara content but Gemini labeled 69% of
those events DECISION. We attributed this to domain shift: v7 trained
on Jay transcripts, Capybara is curated IT -- v7 has no grounding
there. If the hypothesis is right, v6 (trained on an even narrower
synthetic distribution) should also show cross-distribution drift on
the same Capybara events.

This script runs v6 on:
  - The 100 Capybara LOW-P(D) events oracled for v7 (cross-dist).
  - The 494 agree_write events (Jay-transcript HIGH regime).
  - The 288 Capybara uniform-sample events from the earlier probe.

For each set it reports v6's P(D) distribution + ECE + signed bias.
Compared to v7's numbers, we see whether the domain-shift pattern
is version-specific or general.
"""

from __future__ import annotations

import torch  # noqa: F401

import json
import os
import re
from pathlib import Path

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies.types import Event, WriteContext


REPO = Path(__file__).resolve().parent.parent
NANOGPT = Path(
    os.environ.get("NANOGPT_REPO")
    or (REPO.parent / "nanoGPT")
)
V6_CKPT = NANOGPT / "out-merken-bpe-v6" / "ckpt.pt"
V6_META = NANOGPT / "data" / "merken_bpe_v6" / "meta.pkl"

DATASETS = {
    "capybara_lowpd": REPO / "data" / "merken_labels_capybara_lowpd.jsonl",
    "agree_write": REPO / "data" / "merken_labels_agree_write.jsonl",
    "capybara_uniform": REPO / "data" / "merken_labels_ldjnr_capybara.jsonl",
}


def parse_pd(s: str) -> float | None:
    m = re.search(r"P\(D\)=([\d.]+)", s or "")
    return float(m.group(1)) if m else None


def run_v6_on(decider, path: Path):
    ctx = WriteContext(project="v6probe")
    out = []
    if not path.exists():
        return out
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
        d = decider.decide(Event(text=text), ctx)
        p = parse_pd(d.reason)
        if p is None:
            continue
        out.append((p, y))
    return out


def ece_bins(pairs, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total = len(pairs)
    e = 0.0
    signed = 0.0
    rows = []
    for lo, hi in bins:
        in_bin = [(p, y) for p, y in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            continue
        conf = sum(p for p, _ in in_bin) / n
        acc = sum(y for _, y in in_bin) / n
        e += (n / total) * abs(conf - acc)
        signed += (n / total) * (conf - acc)
        rows.append((lo, hi, n, conf, acc, conf - acc))
    return e, signed, rows


def report(label, pairs):
    if not pairs:
        print(f"\n{label}: n=0")
        return
    e, signed, rows = ece_bins(pairs)
    mean_pd = sum(p for p, _ in pairs) / len(pairs)
    dec_share = sum(y for _, y in pairs) / len(pairs)
    print(f"\n=== {label} (n={len(pairs)}) ===")
    print(f"  mean_P(D)={mean_pd:.3f}  true_DEC_share={dec_share:.3f}")
    print(f"  ECE={e:.3f}  signed_bias={signed:+.3f}")
    print(f"  {'bin':<12} {'n':>5} {'conf':>7} {'acc':>7} {'gap':>7}")
    for lo, hi, n, conf, acc, gap in rows:
        print(f"  [{lo:.1f}, {hi:.1f})   {n:>5}   {conf:>5.3f}   {acc:>5.3f}  {gap:>+6.3f}")


def main() -> int:
    print(f"loading v6 from {V6_CKPT}")
    v6 = NanoGPTWriteDecider(str(V6_CKPT), str(V6_META))

    for name, path in DATASETS.items():
        pairs = run_v6_on(v6, path)
        report(name, pairs)

    # Comparative summary
    print("\n" + "=" * 60)
    print("Compare v6 vs v7 signed bias across domains:")
    print(f"  agree_write        v6 vs v7: see 'agree_write' above and the v7 clean result in CONFUSION_MATRIX.md")
    print(f"  capybara_lowpd     v6 vs v7: see 'capybara_lowpd' above and the v7 LOW probe in CONFUSION_MATRIX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())