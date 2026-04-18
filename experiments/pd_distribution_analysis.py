# ruff: noqa: I001, E402
"""Does v7 behave differently on real vs synthetic vs public data?

H1 (emergent uncertainty): v7 recognizes ambiguous content and assigns
P(D) in [0.3, 0.7] more often on real data than on synthetic.

H2 (training prior artifact): v7 has a fixed DEC bias from training
class imbalance. P(D) distribution is roughly the same across
distributions; synthetic just has more obvious noise that clears the
P(D)<0.5 threshold.

This script collects P(D) distributions from v7 on three data sources
and compares:

  - Real Jay transcripts (shadow_reason already logged in the 1520
    bootstrap labels; no re-inference needed)
  - Synthetic benchmark scenarios (events loaded from JSON, v7.decide
    called to get fresh P(D))
  - Capybara public dataset (288 oracled chunks; re-run v7 to get P(D))

For each source, tabulate the fraction of events in bins
[0.0-0.3), [0.3-0.7), [0.7-1.0]. Interpretation:
  - Balanced/bimodal -> H2 (training artifact)
  - Ambiguity bucket (0.3-0.7) larger on real+capybara than on
    synthetic -> H1 (emergent uncertainty)
"""

from __future__ import annotations

import torch  # noqa: F401

import json
import os
import re
from pathlib import Path

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies.types import Event, WriteContext


REPO = Path(__file__).resolve().parent.parent  # engram/experiments/.. -> engram
LABELS_SKIP = REPO / "data" / "merken_labels_v7.jsonl"
LABELS_WRITE = REPO / "data" / "merken_labels_agree_write.jsonl"
LABELS_CAPYBARA = REPO / "data" / "merken_labels_ldjnr_capybara.jsonl"

SCENARIOS = {
    "synthetic_knowledge_update_50t": REPO / "experiments/loop_quality/scenarios/knowledge_update_50topics.json",
    "synthetic_markdown_tables": REPO / "experiments/loop_quality/scenarios/markdown_tables_held_out.json",
}


BINS = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.01)]
BIN_LABELS = ["confident_NOI (<0.3)", "ambiguous (0.3-0.7)", "confident_DEC (>=0.7)"]


def parse_pd(shadow_reason: str) -> float | None:
    m = re.search(r"P\(D\)=(\d+\.\d+|\d+)", shadow_reason or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def bin_of(p: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= p < hi:
            return i
    return len(BINS) - 1


def histogram(pds: list[float]) -> list[int]:
    h = [0] * len(BINS)
    for p in pds:
        h[bin_of(p)] += 1
    return h


def report(name: str, pds: list[float]):
    n = len(pds)
    if n == 0:
        print(f"{name}: n=0 (skipped)")
        return
    h = histogram(pds)
    mean = sum(pds) / n
    pds_sorted = sorted(pds)
    median = pds_sorted[n // 2]
    p10 = pds_sorted[n // 10] if n >= 10 else pds_sorted[0]
    p90 = pds_sorted[int(n * 0.9)] if n >= 10 else pds_sorted[-1]
    print(f"\n{name}  (n={n})")
    print(f"  mean={mean:.3f}  median={median:.3f}  p10={p10:.3f}  p90={p90:.3f}")
    for i, cnt in enumerate(h):
        pct = cnt / n * 100
        bar = "#" * int(pct / 2)
        print(f"  {BIN_LABELS[i]:<25} {cnt:>5} ({pct:>5.1f}%) {bar}")


def collect_labels_pd(path: Path, decider: NanoGPTWriteDecider | None = None,
                      text_field: str = "text", force_reinfer: bool = False):
    """Return list of P(D) for rows in JSONL.

    Uses shadow_reason if present (bootstrap_from_transcripts format).
    Falls back to running v7.decide on text field (for formats that
    don't log shadow_reason, e.g. agree_write and capybara).
    """
    out = []
    if not path.exists():
        return out
    ctx = WriteContext(project="eval")
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        p = None
        if not force_reinfer:
            p = parse_pd(r.get("shadow_reason"))
        if p is None and decider is not None:
            text = (r.get(text_field) or "").strip()
            if text:
                d = decider.decide(Event(text=text), ctx)
                p = parse_pd(d.reason)
        if p is not None:
            out.append(p)
    return out


def collect_scenario_pd(scenario_path: Path, decider: NanoGPTWriteDecider) -> list[float]:
    if not scenario_path.exists():
        return []
    data = json.loads(scenario_path.read_text())
    ctx = WriteContext(project="eval")
    out = []
    for ev in data.get("events", []):
        text = (ev.get("text") or "").strip()
        if not text:
            continue
        d = decider.decide(Event(text=text), ctx)
        p = parse_pd(d.reason)
        if p is not None:
            out.append(p)
    return out


def collect_capybara_pd(path: Path, decider: NanoGPTWriteDecider) -> list[float]:
    if not path.exists():
        return []
    ctx = WriteContext(project="capybara")
    out = []
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        d = decider.decide(Event(text=text), ctx)
        p = parse_pd(d.reason)
        if p is not None:
            out.append(p)
    return out


def main() -> int:
    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit(
            "Set MERKEN_SHADOW_NANOGPT_CKPT / MERKEN_SHADOW_NANOGPT_META"
        )
    print(f"loading v7 from {ckpt}")
    v7 = NanoGPTWriteDecider(ckpt, meta)

    # Real skip: shadow_reason logged in the bootstrap format.
    real_skip = collect_labels_pd(LABELS_SKIP, decider=v7)
    # Real write: re-infer (agree_write format doesn't log shadow_reason).
    real_write = collect_labels_pd(LABELS_WRITE, decider=v7, force_reinfer=True)
    real_all = real_skip + real_write

    # Synthetic: fresh v7.decide calls on the scenario events.
    synth_pds: dict[str, list[float]] = {}
    for name, path in SCENARIOS.items():
        synth_pds[name] = collect_scenario_pd(path, v7)

    # Capybara: fresh v7.decide calls on the labeled chunks.
    capybara = collect_capybara_pd(LABELS_CAPYBARA, v7)

    print("\n" + "=" * 70)
    print("v7 P(D) distribution by data source")
    print("=" * 70)
    report("REAL  -- Jay shadow_skip events (v7 said SKIP)", real_skip)
    report("REAL  -- Jay agree_write events (v7 said WRITE)", real_write)
    report("REAL  -- all 1520 Jay labeled events", real_all)
    for name, pds in synth_pds.items():
        report(f"SYNTH -- {name}", pds)
    report("PUBLIC -- Capybara chunks", capybara)

    # H1 vs H2 verdict: compare ambiguous bucket share
    print("\n" + "=" * 70)
    print("H1 vs H2 verdict (ambiguous-bucket share = %% in [0.3, 0.7))")
    print("=" * 70)
    def amb_share(pds):
        if not pds:
            return 0.0
        return sum(1 for p in pds if 0.3 <= p < 0.7) / len(pds)

    print(f"  REAL (Jay, all 1520):          {amb_share(real_all)*100:>5.1f}%")
    for name, pds in synth_pds.items():
        print(f"  SYNTH {name:<30} {amb_share(pds)*100:>5.1f}%")
    print(f"  PUBLIC Capybara:               {amb_share(capybara)*100:>5.1f}%")
    print()
    print(
        "H1 holds if REAL+Capybara (diverse) ambiguity-share >> SYNTH.\n"
        "H2 holds if distributions are similar across sources."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())