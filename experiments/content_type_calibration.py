"""Does v7's calibration depend on content type (code vs prose vs table)?

For each of the 1520 oracled events we tag the content structure,
then compute v7's P(D) distribution + ECE by category. Helps answer:

- Is v7 over-confident universally, or only on prose-heavy content?
- Does code-block presence change the calibration regime?
- Does table-format content (common in markdown) sit in a different
  regime than paragraph text?

Categories (non-exclusive, an event can belong to multiple):

  has_code_fence      contains ``` anywhere
  has_inline_code     contains `xxx` backticks
  has_markdown_table  contains two or more `|` on one line
  has_numbers         contains digits in >=3 places
  has_file_paths      contains a `.py`, `.js`, `.md`, `.json`, `/`
  short               text length < 300 chars
  long                text length >= 1000 chars

For each tag we report n, mean P(D), DEC share, ECE, signed bias.
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

CODE_FENCE = re.compile(r"```")
INLINE_CODE = re.compile(r"`[^`\n]{2,}`")
TABLE_ROW = re.compile(r"\|[^\n]*\|[^\n]*\|")
FILE_PATH = re.compile(r"\b\S+\.(?:py|js|ts|md|json|toml|yml|yaml|go|rs|sh|sql)\b")
NUMBER = re.compile(r"\d")


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


def collect_all(decider):
    ctx = WriteContext(project="content")
    out = []  # (text, P(D), y_01, tags)

    def ingest(path: Path, use_shadow_reason: bool):
        if not path.exists():
            return
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
            p = None
            if use_shadow_reason:
                p = parse_pd(r.get("shadow_reason"))
            if p is None:
                d = decider.decide(Event(text=text), ctx)
                p = parse_pd(d.reason)
            if p is None:
                continue
            out.append((text, p, y, tags_for(text)))

    ingest(LABELS_SKIP, use_shadow_reason=True)
    ingest(LABELS_WRITE, use_shadow_reason=False)
    return out


def ece_signed(pairs, n_bins=10):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    N = len(pairs)
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


def main() -> int:
    ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    if not ckpt or not meta:
        raise SystemExit("Set MERKEN_SHADOW_NANOGPT_CKPT / MERKEN_SHADOW_NANOGPT_META")

    print("loading v7 + collecting tagged pairs...")
    v7 = NanoGPTWriteDecider(ckpt, meta)
    rows = collect_all(v7)
    print(f"total: {len(rows)} events")

    categories = [
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

    print(f"\n{'category':<22} {'n':>5} {'mean_P(D)':>10} {'DEC_share':>10} "
          f"{'ECE':>6} {'signed':>8}")
    print("-" * 70)
    results = {}
    for cat in categories:
        selected = [(p, y) for _, p, y, tags in rows if cat in tags]
        if not selected:
            print(f"{cat:<22} {'0':>5}")
            continue
        mean_pd = sum(p for p, _ in selected) / len(selected)
        dec_share = sum(y for _, y in selected) / len(selected)
        e, signed = ece_signed(selected)
        print(f"{cat:<22} {len(selected):>5}  {mean_pd:>8.3f}   "
              f"{dec_share:>7.3f}   {e:>5.3f}  {signed:>+6.3f}")
        results[cat] = {
            "n": len(selected), "mean_pd": mean_pd,
            "dec_share": dec_share, "ece": e, "signed_bias": signed,
        }

    # Save
    out_path = REPO / "experiments" / "nanogpt" / "content_type_calibration.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
