"""Compute v7's full confusion matrix from the two labeled sets.

Combines:
  - data/merken_labels_v7.jsonl -- 1026 shadow_skip events, strict-
    oracle labels (produced by bootstrap_from_transcripts +
    relabel_decision_pile with gemini-2.0-flash).
  - data/merken_labels_agree_write.jsonl -- agree_write reservoir
    sample, same strict oracle (produced by oracle_agree_write_sample).

v7's decision on each labeled event is known by construction:
  - rows in merken_labels_v7.jsonl:  v7 SKIPPED  (disagreement set)
  - rows in merken_labels_agree_write.jsonl: v7 WROTE (agreement set)

From these we can compute the full 2x2 confusion matrix plus metrics
that are weighted by the real population ratio (total agree_write /
total shadow_skip observed during the bootstrap scan).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LABELS_SKIP = Path("data/merken_labels_v7.jsonl")
LABELS_WRITE = Path("data/merken_labels_agree_write.jsonl")

# Population counts from the last bootstrap_from_transcripts dry-run
# (2026-04-17, full scan of 415 transcripts). These weights convert
# per-sample rates into population-weighted metrics.
POP_WRITE = 12492
POP_SKIP = 1047


@dataclass
class Counts:
    tp: int = 0  # v7 WROTE and oracle=DEC  -> correct keep
    fp: int = 0  # v7 WROTE and oracle=NOI  -> wrong keep
    tn: int = 0  # v7 SKIPPED and oracle=NOI -> correct drop
    fn: int = 0  # v7 SKIPPED and oracle=DEC -> wrong drop


def load_labels(path: Path) -> list[dict]:
    rows = []
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows.append(r)
    return rows


def build_counts() -> tuple[Counts, dict]:
    c = Counts()
    skip_labels = load_labels(LABELS_SKIP)
    write_labels = load_labels(LABELS_WRITE)

    # Skip set: v7 said SKIP on all of these.
    for r in skip_labels:
        label = r.get("label")
        if label == "DECISION":
            c.fn += 1
        elif label == "NOISE":
            c.tn += 1
        # UNCERTAIN already dropped in extract; defensive no-op
    # Write set: v7 said WRITE on all of these.
    for r in write_labels:
        label = r.get("label")
        if label == "DECISION":
            c.tp += 1
        elif label == "NOISE":
            c.fp += 1

    return c, {
        "n_skip": len(skip_labels),
        "n_write": len(write_labels),
        "skip_dec": c.fn,
        "skip_noi": c.tn,
        "write_dec": c.tp,
        "write_noi": c.fp,
    }


def metrics_on_sample(c: Counts) -> dict[str, float]:
    total = c.tp + c.fp + c.tn + c.fn
    if total == 0:
        return {}
    return {
        "n": total,
        "accuracy": (c.tp + c.tn) / total,
        "dec_precision": c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0,
        "dec_recall": c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0,
        "noi_precision": c.tn / (c.tn + c.fn) if (c.tn + c.fn) else 0.0,
        "noi_recall": c.tn / (c.tn + c.fp) if (c.tn + c.fp) else 0.0,
    }


def metrics_population_weighted(
    write_dec_rate: float,
    write_noi_rate: float,
    skip_dec_rate: float,
    skip_noi_rate: float,
    pop_write: int = POP_WRITE,
    pop_skip: int = POP_SKIP,
) -> dict[str, float]:
    """Project sample rates onto the real population.

    The sample class ratios for agree_write vs shadow_skip differ from
    the population (we sampled uniformly inside each group). Scaling
    each group's oracle-DEC/NOI rate by the group's population size
    gives an unbiased estimate of the full confusion matrix on the
    ~13,539-event population.
    """
    # Writes in population (v7 keeps these)
    tp_pop = pop_write * write_dec_rate
    fp_pop = pop_write * write_noi_rate
    # Skips in population (v7 drops these)
    fn_pop = pop_skip * skip_dec_rate
    tn_pop = pop_skip * skip_noi_rate

    total_pop = tp_pop + fp_pop + fn_pop + tn_pop
    dec_pop = tp_pop + fn_pop
    noi_pop = fp_pop + tn_pop

    return {
        "population": int(round(total_pop)),
        "dec_pop": int(round(dec_pop)),
        "noi_pop": int(round(noi_pop)),
        "dec_share": dec_pop / total_pop,
        "accuracy": (tp_pop + tn_pop) / total_pop,
        "dec_recall": tp_pop / dec_pop if dec_pop else 0.0,
        "dec_precision": tp_pop / (tp_pop + fp_pop) if (tp_pop + fp_pop) else 0.0,
        "noi_recall": tn_pop / noi_pop if noi_pop else 0.0,
        "noi_precision": tn_pop / (tn_pop + fn_pop) if (tn_pop + fn_pop) else 0.0,
        "store_reduction": (fn_pop + tn_pop) / total_pop,
    }


def main() -> int:
    c, raw = build_counts()

    print("=== Sample counts ===")
    print(f"  skip set (v7 SKIP):   {raw['n_skip']}  -> DEC={raw['skip_dec']} NOI={raw['skip_noi']}")
    print(f"  write set (v7 WRITE): {raw['n_write']} -> DEC={raw['write_dec']} NOI={raw['write_noi']}")
    print()
    print("=== Confusion matrix (sample-level) ===")
    print(f"                 oracle_DEC   oracle_NOI")
    print(f"  v7 WRITE       {c.tp:>5} (TP)  {c.fp:>5} (FP)")
    print(f"  v7 SKIP        {c.fn:>5} (FN)  {c.tn:>5} (TN)")
    print()

    m = metrics_on_sample(c)
    print("=== Sample metrics (unweighted) ===")
    print(f"  n                {m['n']}")
    print(f"  accuracy         {m['accuracy']*100:.1f}%")
    print(f"  DEC precision    {m['dec_precision']*100:.1f}%")
    print(f"  DEC recall       {m['dec_recall']*100:.1f}%")
    print(f"  NOI precision    {m['noi_precision']*100:.1f}%")
    print(f"  NOI recall       {m['noi_recall']*100:.1f}%  (aka specificity)")
    print()

    # Per-group rates for population weighting
    write_total = raw["write_dec"] + raw["write_noi"]
    skip_total = raw["skip_dec"] + raw["skip_noi"]
    write_dec_rate = raw["write_dec"] / write_total if write_total else 0.0
    write_noi_rate = raw["write_noi"] / write_total if write_total else 0.0
    skip_dec_rate = raw["skip_dec"] / skip_total if skip_total else 0.0
    skip_noi_rate = raw["skip_noi"] / skip_total if skip_total else 0.0

    pm = metrics_population_weighted(
        write_dec_rate, write_noi_rate, skip_dec_rate, skip_noi_rate
    )
    print(f"=== Population-weighted (POP_WRITE={POP_WRITE}, POP_SKIP={POP_SKIP}) ===")
    print(f"  population size    {pm['population']:,}")
    print(f"  true DEC in pop    {pm['dec_pop']:,} ({pm['dec_share']*100:.1f}%)")
    print(f"  true NOI in pop    {pm['noi_pop']:,} ({(1-pm['dec_share'])*100:.1f}%)")
    print(f"  accuracy           {pm['accuracy']*100:.1f}%")
    print(f"  DEC recall         {pm['dec_recall']*100:.1f}%")
    print(f"  DEC precision      {pm['dec_precision']*100:.1f}%")
    print(f"  NOI recall (spec)  {pm['noi_recall']*100:.1f}%")
    print(f"  NOI precision      {pm['noi_precision']*100:.1f}%")
    print(f"  store reduction    {pm['store_reduction']*100:.1f}%")
    print()

    print("=== Reader's notes ===")
    print(
        "  DEC recall >> NOI recall: v7 is a CONSERVATIVE filter. It rarely\n"
        "  loses real decisions (~98%) but also rarely blocks noise (~27%).\n"
        "  Real-world store reduction is ~8%, not the ~86% reported against\n"
        "  synthetic scenarios (those had class imbalance the filter was\n"
        "  explicitly designed against). This is a faithful result, not a\n"
        "  regression."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
