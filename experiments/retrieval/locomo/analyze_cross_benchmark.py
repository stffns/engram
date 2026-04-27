"""Cross-benchmark analyzer: LongMemEval Phase 2 vs LoCoMo Phase 2.

Loads the 3-seed Phase 2 vstash.ask runs from both benchmarks and
emits a side-by-side breakdown:

  - Overall correct rate, mean +- stdev across seeds
  - Per-shape breakdown (LME question_type, LoCoMo category_name)
  - Aligned shapes table where the mapping is defensible
  - Verdict mix per shape (supports / partial / contradicts / neutral / error)

The script is descriptive only -- it does not score or merge the
two benchmarks into a single number, since the QA distributions and
oracle behavior differ. The output is the input to the cross-benchmark
shape transfer decision.

Usage::

    python3 experiments/retrieval/locomo/analyze_cross_benchmark.py \
      --locomo-glob 'experiments/retrieval/locomo/phase2_runs/locomo_phase2_seed*_3seed_n40_*.jsonl' \
      --lme-files \
        experiments/retrieval/longmemeval/pipeline_runs/vstash_ask_seed42_n30_vstash_ask_seed42_*.jsonl \
        experiments/retrieval/longmemeval/pipeline_runs/vstash_ask_seed43_n30_vstash_ask_seed43_*.jsonl \
        experiments/retrieval/longmemeval/pipeline_runs/vstash_ask_seed44_n30_vstash_ask_cerebras_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import Counter
from pathlib import Path

VERDICT_CORRECT = {"supports", "partial"}


# Mapping from LongMemEval question_type to a coarse semantic shape
# that has a defensible LoCoMo analog. Anything not listed here is
# left unmapped and only shows up in the per-benchmark breakdown.
LME_TO_SHAPE = {
    "temporal-reasoning": "temporal",
    "multi-session": "multi_hop",
    "single-session-user": "single_hop",
    "single-session-assistant": "single_hop",
    "single-session-preference": "single_hop",
    "knowledge-update": "open_domain",
}

LOCOMO_TO_SHAPE = {
    "temporal": "temporal",
    "multi_hop": "multi_hop",
    "single_hop": "single_hop",
    "open_domain": "open_domain",
    "adversarial": "adversarial",
}


def _seed_from_filename(name: str) -> int | None:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def _config_from_filename(name: str) -> str:
    """Extract config from a LoCoMo phase2 filename of the form
    ``locomo_phase2_<config>_seed<N>_...jsonl``. Returns 'vstash-raw'
    for older filenames that did not encode the config (those runs
    predate the --config flag and were always vstash-raw).
    """
    m = re.search(r"locomo_phase2_([a-z0-9\-]+?)_seed", name)
    if m:
        return m.group(1)
    return "vstash-raw"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _verdict(row: dict) -> str:
    if "error" in row:
        return "error"
    o = row.get("oracle") or {}
    return o.get("verdict") or "error"


def _correct_rate(rows: list[dict]) -> tuple[int, int]:
    correct = sum(1 for r in rows if _verdict(r) in VERDICT_CORRECT)
    return correct, len(rows)


def _trust_score(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    n = len(rows)
    correct = sum(1 for r in rows if _verdict(r) in VERDICT_CORRECT)
    contradicts = sum(1 for r in rows if _verdict(r) == "contradicts")
    return (correct - contradicts) / n


def _summarize_seed(rows: list[dict], shape_key) -> dict:
    by_shape: dict[str, list[dict]] = {}
    for r in rows:
        by_shape.setdefault(shape_key(r), []).append(r)
    correct, total = _correct_rate(rows)
    summary = {
        "n": total,
        "correct": correct,
        "correct_rate": correct / total if total else 0.0,
        "trust_score": _trust_score(rows),
        "verdicts": dict(Counter(_verdict(r) for r in rows)),
        "by_shape": {},
    }
    for shape, group in sorted(by_shape.items()):
        c, t = _correct_rate(group)
        summary["by_shape"][shape] = {
            "n": t,
            "correct": c,
            "correct_rate": c / t if t else 0.0,
            "verdicts": dict(Counter(_verdict(r) for r in group)),
        }
    return summary


def _shape_key_lme(row: dict) -> str:
    qt = row.get("question_type") or "unknown"
    return qt


def _shape_key_locomo(row: dict) -> str:
    return row.get("category_name") or "unknown"


def _aligned_shape(seed_summary: dict, source_to_shape: dict[str, str]) -> dict:
    out: dict[str, dict] = {}
    for shape_in, info in seed_summary["by_shape"].items():
        shape_out = source_to_shape.get(shape_in, "unmapped")
        bucket = out.setdefault(shape_out, {"n": 0, "correct": 0})
        bucket["n"] += info["n"]
        bucket["correct"] += info["correct"]
    for shape, b in out.items():
        b["correct_rate"] = b["correct"] / b["n"] if b["n"] else 0.0
    return out


def _agg_seeds(per_seed: list[dict], key: str) -> tuple[float, float]:
    """Return (mean, stdev) of `key` over per-seed summaries."""
    vals = [s[key] for s in per_seed]
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (statistics.mean(vals), statistics.stdev(vals))


def _format_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _print_block(title: str, per_seed: list[dict], source_to_shape: dict[str, str]) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    correct_mean, correct_std = _agg_seeds(per_seed, "correct_rate")
    trust_mean, trust_std = _agg_seeds(per_seed, "trust_score")
    n_seeds = len(per_seed)
    n_per = [s["n"] for s in per_seed]
    print(
        f"  seeds={n_seeds}  N/seed={n_per}  "
        f"correct_rate = {_format_pct(correct_mean)} +- {_format_pct(correct_std)}  "
        f"trust = {trust_mean*100:+.1f}% +- {trust_std*100:.1f}%"
    )

    # Per-source-shape (raw question_type / category_name)
    print()
    print(f"  by source shape:")
    print(f"    {'shape':<28} {'mean%':>7}  {'std%':>5}  per-seed (n,correct%)")
    all_shapes = set()
    for s in per_seed:
        all_shapes |= s["by_shape"].keys()
    for shape in sorted(all_shapes):
        rates = [
            s["by_shape"].get(shape, {"correct_rate": 0.0, "n": 0})["correct_rate"]
            for s in per_seed
        ]
        ns = [s["by_shape"].get(shape, {"n": 0})["n"] for s in per_seed]
        m, sd = (statistics.mean(rates), statistics.stdev(rates)) if len(rates) > 1 else (rates[0] if rates else 0.0, 0.0)
        per_seed_str = " ".join(
            f"({s['by_shape'].get(shape, {'n':0})['n']},"
            f"{_format_pct(s['by_shape'].get(shape, {'correct_rate':0.0})['correct_rate']).strip()})"
            for s in per_seed
        )
        print(f"    {shape:<28} {_format_pct(m):>7}  {sd*100:>4.1f}  {per_seed_str}")

    # Aligned shape (mapped to common space)
    aligned_per_seed = [_aligned_shape(s, source_to_shape) for s in per_seed]
    aligned_shapes = set()
    for a in aligned_per_seed:
        aligned_shapes |= a.keys()
    print()
    print(f"  by aligned shape (LME->LoCoMo space):")
    print(f"    {'shape':<14} {'mean%':>7}  {'std%':>5}  per-seed (n,correct%)")
    for shape in sorted(aligned_shapes):
        rates = [a.get(shape, {"correct_rate": 0.0})["correct_rate"] for a in aligned_per_seed]
        ns = [a.get(shape, {"n": 0})["n"] for a in aligned_per_seed]
        m, sd = (statistics.mean(rates), statistics.stdev(rates)) if len(rates) > 1 else (rates[0] if rates else 0.0, 0.0)
        per_seed_str = " ".join(
            f"({a.get(shape, {'n':0})['n']},{_format_pct(a.get(shape, {'correct_rate':0.0})['correct_rate']).strip()})"
            for a in aligned_per_seed
        )
        print(f"    {shape:<14} {_format_pct(m):>7}  {sd*100:>4.1f}  {per_seed_str}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locomo-glob", required=True,
                    help="Glob for LoCoMo phase2 jsonl. May resolve to "
                         "multiple seeds and/or configs; the script "
                         "groups by config (parsed from filename: "
                         "locomo_phase2_<config>_seed...).")
    ap.add_argument("--lme-files", nargs="+", required=True,
                    help="Explicit LME phase2 jsonl files (one per seed). "
                         "All assumed to be the vstash-raw baseline -- LME "
                         "merken configs are not yet ported.")
    args = ap.parse_args()

    # LoCoMo, grouped by config
    locomo_files = sorted(glob.glob(args.locomo_glob))
    if not locomo_files:
        raise SystemExit(f"No LoCoMo files match: {args.locomo_glob}")
    locomo_by_config: dict[str, list[dict]] = {}
    for fp in locomo_files:
        rows = _load_jsonl(Path(fp))
        # Drop empty/zero-row files (smoke artifacts that were aborted).
        if not rows:
            continue
        cfg = _config_from_filename(Path(fp).name)
        s = _summarize_seed(rows, _shape_key_locomo)
        s["seed"] = _seed_from_filename(fp)
        s["file"] = fp
        s["config"] = cfg
        locomo_by_config.setdefault(cfg, []).append(s)
        print(
            f"[locomo {cfg:<14}] seed={s['seed']} "
            f"file={Path(fp).name} n={s['n']} correct={s['correct']}"
        )

    # LME
    lme_files = []
    for pat in args.lme_files:
        lme_files.extend(sorted(glob.glob(pat)))
    if not lme_files:
        raise SystemExit(f"No LME files match: {args.lme_files}")
    lme_per_seed = []
    for fp in lme_files:
        rows = _load_jsonl(Path(fp))
        s = _summarize_seed(rows, _shape_key_lme)
        s["seed"] = _seed_from_filename(fp)
        s["file"] = fp
        lme_per_seed.append(s)
        print(f"[lme]    seed={s['seed']} file={Path(fp).name} n={s['n']} correct={s['correct']}")

    _print_block(
        "LongMemEval (Phase 2, vstash.ask, llama3.1-8b, k=8)",
        lme_per_seed, LME_TO_SHAPE,
    )
    for cfg in sorted(locomo_by_config):
        _print_block(
            f"LoCoMo [{cfg}]  (vstash.ask, llama3.1-8b, k=8)",
            locomo_by_config[cfg],
            LOCOMO_TO_SHAPE,
        )

    # Aligned shape side-by-side -- only meaningful for vstash-raw
    # since LME is not yet ported to merken configs.
    if "vstash-raw" in locomo_by_config:
        print()
        print("=" * 78)
        print("Cross-benchmark: LoCoMo[vstash-raw] vs LME[vstash-raw], mean across seeds")
        print("=" * 78)
        aligned_lme = [_aligned_shape(s, LME_TO_SHAPE) for s in lme_per_seed]
        aligned_locomo = [
            _aligned_shape(s, LOCOMO_TO_SHAPE)
            for s in locomo_by_config["vstash-raw"]
        ]
        shapes = sorted(
            set().union(
                *(a.keys() for a in aligned_lme),
                *(a.keys() for a in aligned_locomo),
            )
        )
        print(f"  {'shape':<14} {'LME mean':>10}  {'LoCoMo mean':>13}  {'delta(L-LME)':>14}")
        for shape in shapes:
            l_rates = [a.get(shape, {"correct_rate": 0.0})["correct_rate"] for a in aligned_lme]
            c_rates = [a.get(shape, {"correct_rate": 0.0})["correct_rate"] for a in aligned_locomo]
            l_n = sum(a.get(shape, {"n": 0})["n"] for a in aligned_lme)
            c_n = sum(a.get(shape, {"n": 0})["n"] for a in aligned_locomo)
            l_m = statistics.mean(l_rates) if l_rates else 0.0
            c_m = statistics.mean(c_rates) if c_rates else 0.0
            delta = c_m - l_m
            print(
                f"  {shape:<14} {l_m*100:>8.1f}% (n={l_n})  {c_m*100:>10.1f}% (n={c_n})  "
                f"{delta*100:>+12.1f}pp"
            )

    # Write-side comparison: LoCoMo configs against each other.
    if len(locomo_by_config) >= 2:
        print()
        print("=" * 78)
        print("Write-side comparison on LoCoMo (same builder, different ingest)")
        print("=" * 78)
        cfgs = sorted(locomo_by_config)
        # Per config, mean correct + per-shape mean
        print(f"  {'config':<16} {'correct':>9}  {'trust':>7}  per-shape correct%")
        for cfg in cfgs:
            per_seed = locomo_by_config[cfg]
            mean_correct, _ = _agg_seeds(per_seed, "correct_rate")
            mean_trust, _ = _agg_seeds(per_seed, "trust_score")
            shape_str_parts = []
            shape_means: dict[str, float] = {}
            for shape in ("temporal", "multi_hop", "single_hop", "open_domain", "adversarial"):
                rates = [
                    s["by_shape"].get(shape, {"correct_rate": 0.0})["correct_rate"]
                    for s in per_seed
                ]
                shape_means[shape] = statistics.mean(rates) if rates else 0.0
                shape_str_parts.append(
                    f"{shape[:4]}={shape_means[shape]*100:5.1f}%"
                )
            print(
                f"  {cfg:<16} {mean_correct*100:>8.1f}%  "
                f"{mean_trust*100:>+6.1f}%  " + "  ".join(shape_str_parts)
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
