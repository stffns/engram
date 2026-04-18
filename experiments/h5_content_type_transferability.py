# ruff: noqa: I001, E402
"""H5 -- does the content-type calibration pattern transfer across datasets?

The parent analysis (content_type_calibration.py) showed v7's ECE
varies 4x across structure tags on Jay's 1520 oracled events:
  long (>=1000c)      ECE 0.045
  has_file_paths      ECE 0.084
  has_inline_code     ECE 0.103
  has_numbers         ECE 0.104
  has_code_fence      ECE 0.107
  short (<300c)       ECE 0.168
  has_markdown_table  ECE 0.184
  plain_prose         ECE 0.191

Question: does this ordering hold on other datasets, or is it a
Jay-specific artifact?

We re-run the same tagger on two public-dataset label sets we already
have on disk (no new API calls):
  - data/merken_labels_ldjnr_capybara.jsonl (n=288)
  - data/merken_labels_slimorca.jsonl (n=200)

For each dataset we compute per-structure ECE and then compare the
per-category ECE *ranking* against Jay's ranking via Spearman
correlation. If rho >= 0.7 on both, the claim transfers.

This hypothesis is cheap: no training, no API. Just tagging +
calibration math on labels already oracled.
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
LABEL_FILES = {
    "jay_1520": [
        REPO / "data" / "merken_labels_v7.jsonl",
        REPO / "data" / "merken_labels_agree_write.jsonl",
    ],
    "capybara_288": [
        REPO / "data" / "merken_labels_ldjnr_capybara.jsonl",
    ],
    "slimorca_200": [
        REPO / "data" / "merken_labels_slimorca.jsonl",
    ],
}

CODE_FENCE = re.compile(r"```")
INLINE_CODE = re.compile(r"`[^`\n]{2,}`")
TABLE_ROW = re.compile(r"\|[^\n]*\|[^\n]*\|")
FILE_PATH = re.compile(r"\b\S+\.(?:py|js|ts|md|json|toml|yml|yaml|go|rs|sh|sql)\b")
NUMBER = re.compile(r"\d")

CATEGORIES = [
    "has_code_fence",
    "has_inline_code",
    "has_markdown_table",
    "has_numbers",
    "has_file_paths",
    "short",
    "long",
    "plain_prose",
    "all",
]


def parse_pd(s: str) -> float | None:
    m = re.search(r"P\(D\)=([\d.]+)", s or "")
    return float(m.group(1)) if m else None


def tags_for(text: str) -> list[str]:
    tags = []
    if CODE_FENCE.search(text):
        tags.append("has_code_fence")
    if INLINE_CODE.search(text):
        tags.append("has_inline_code")
    if TABLE_ROW.search(text):
        tags.append("has_markdown_table")
    if len(NUMBER.findall(text)) >= 3:
        tags.append("has_numbers")
    if FILE_PATH.search(text):
        tags.append("has_file_paths")
    if len(text) < 300:
        tags.append("short")
    elif len(text) >= 1000:
        tags.append("long")
    if not any(t for t in tags if t.startswith("has_")):
        tags.append("plain_prose")
    tags.append("all")
    return tags


def collect_tagged(decider: NanoGPTWriteDecider, paths: list[Path]):
    """Return list of (text, p_d, y, tags) from the given label files."""
    ctx = WriteContext(project="h5")
    out = []
    for path in paths:
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
                # Try to reuse any logged P(D) first.
                p = parse_pd(r.get("shadow_reason"))
                if p is None:
                    d = decider.decide(Event(text=text), ctx)
                    p = parse_pd(d.reason)
                if p is None:
                    continue
                out.append((text, p, y, tags_for(text)))
    return out


def ece_signed(pairs, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    N = len(pairs)
    if N == 0:
        return 0.0, 0.0
    e = 0.0
    signed = 0.0
    for lo, hi in bins:
        in_bin = [(p, y) for p, y in pairs if lo <= p < (hi if hi < 1.0 else 1.01)]
        n = len(in_bin)
        if n == 0:
            continue
        conf = sum(p for p, _ in in_bin) / n
        acc = sum(y for _, y in in_bin) / n
        e += (n / N) * abs(conf - acc)
        signed += (n / N) * (conf - acc)
    return e, signed


def spearman(x: list[float], y: list[float]) -> float:
    """Rank-based correlation with proper tie handling (fractional ranks).

    Ties get the average of the ranks they'd receive if broken
    arbitrarily. This is the standard Spearman behavior used by
    scipy.stats.spearmanr. Without it, our small N was letting
    tied values silently skew the correlation.
    """
    assert len(x) == len(y)
    n = len(x)
    if n < 2:
        return 0.0

    def average_ranks(vals):
        """Return ranks with ties broken by averaging."""
        indexed = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[indexed[j + 1]] == vals[indexed[i]]:
                j += 1
            # indices i..j inclusive are tied; they share average rank
            avg_rank = (i + j) / 2 + 1  # 1-indexed
            for k in range(i, j + 1):
                r[indexed[k]] = avg_rank
            i = j + 1
        return r

    rx = average_ranks(x)
    ry = average_ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def main() -> int:
    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit("set MERKEN_SHADOW_NANOGPT_CKPT / _META")
    print(f"loading v7 from {ckpt}\n")
    v7 = NanoGPTWriteDecider(ckpt, meta)

    per_dataset_ece: dict[str, dict[str, float]] = {}
    per_dataset_signed: dict[str, dict[str, float]] = {}
    per_dataset_n: dict[str, dict[str, int]] = {}

    for ds_name, paths in LABEL_FILES.items():
        rows = collect_tagged(v7, paths)
        if not rows:
            print(f"{ds_name}: NO DATA")
            continue
        print(f"=== {ds_name} (n={len(rows)}) ===")
        print(f"{'category':<22} {'n':>5} {'mean_P(D)':>10} {'DEC_share':>10} "
              f"{'ECE':>6} {'signed':>8}")
        per_dataset_ece[ds_name] = {}
        per_dataset_signed[ds_name] = {}
        per_dataset_n[ds_name] = {}
        for cat in CATEGORIES:
            selected = [(p, y) for _, p, y, tags in rows if cat in tags]
            if not selected:
                continue
            n = len(selected)
            mean_pd = sum(p for p, _ in selected) / n
            dec_share = sum(y for _, y in selected) / n
            e, s = ece_signed(selected)
            print(f"{cat:<22} {n:>5}  {mean_pd:>8.3f}   "
                  f"{dec_share:>7.3f}   {e:>5.3f}  {s:>+6.3f}")
            per_dataset_ece[ds_name][cat] = e
            per_dataset_signed[ds_name][cat] = s
            per_dataset_n[ds_name][cat] = n
        print()

    # Cross-dataset ranking comparison (exclude 'all')
    rankable = [c for c in CATEGORIES if c != "all"]
    print("=" * 72)
    print("Per-category ECE ranking (higher ECE = worse calibration):")
    print(f"{'category':<22} " + "  ".join(f"{d:>12}" for d in per_dataset_ece))
    for cat in rankable:
        cells = []
        for ds in per_dataset_ece:
            e = per_dataset_ece[ds].get(cat)
            n = per_dataset_n[ds].get(cat, 0)
            if e is None:
                cells.append("          --")
            else:
                cells.append(f"   {e:.3f} (n={n:>3})")
        print(f"{cat:<22} " + "  ".join(cells))

    print()
    print("Spearman rank correlation of ECE ordering vs Jay:")
    print("  (intersection of categories, min_n=5 per dataset to avoid "
          "rank noise from n=1 bins)")
    MIN_N = 5
    jay_cats = {
        c for c in rankable
        if per_dataset_ece["jay_1520"].get(c) is not None
        and per_dataset_n["jay_1520"].get(c, 0) >= MIN_N
    }
    spearman_out: dict[str, dict] = {}
    for ds in per_dataset_ece:
        if ds == "jay_1520":
            continue
        ds_cats = {
            c for c in rankable
            if per_dataset_ece[ds].get(c) is not None
            and per_dataset_n[ds].get(c, 0) >= MIN_N
        }
        shared = sorted(jay_cats & ds_cats)
        if len(shared) < 2:
            print(f"  jay_1520 vs {ds:<14}  rho = N/A ({len(shared)} shared bins, need >=2)")
            spearman_out[ds] = {"rho": None, "shared_categories": shared}
            continue
        jay_vals = [per_dataset_ece["jay_1520"][c] for c in shared]
        ds_vals = [per_dataset_ece[ds][c] for c in shared]
        rho = spearman(jay_vals, ds_vals)
        print(f"  jay_1520 vs {ds:<14}  rho = {rho:+.3f}  over {len(shared)} bins: {shared}")
        spearman_out[ds] = {
            "rho": rho,
            "shared_categories": shared,
            "jay_ece": jay_vals,
            "ds_ece": ds_vals,
        }

    # Save
    out = {
        "datasets": {
            ds: {
                cat: {
                    "ece": per_dataset_ece[ds].get(cat),
                    "signed_bias": per_dataset_signed[ds].get(cat),
                    "n": per_dataset_n[ds].get(cat),
                }
                for cat in CATEGORIES
                if per_dataset_ece[ds].get(cat) is not None
            }
            for ds in per_dataset_ece
        },
        "spearman_vs_jay": spearman_out,
        "min_n_per_bin": MIN_N,
    }
    out_path = REPO / "experiments" / "nanogpt" / "h5_transferability.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
