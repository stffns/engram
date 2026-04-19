# ruff: noqa: I001, E402
"""Mine same-starter ambiguous pairs from the v7 training mix.

H2 thesis: v8 collapsed because dataset levers all pushed the model
toward starter shortcuts. A contrastive loss on same-starter
opposite-class pairs would force attention past the starter. Step 0
is to confirm the dataset HAS enough such pairs to train on.

Output: experiments/nanogpt/h2_starter_pairs.json
    {
      "n_total": int,
      "n_dec": int, "n_noi": int,
      "n_unique_starters": int,
      "n_ambiguous_starters": int,  # starters with both DEC and NOI
      "ambiguous_starters": [
        {
          "starter": "let me check",
          "n_dec": 3, "n_noi": 8,
          "dec_texts": [...], "noi_texts": [...]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent
SCEN = ENGRAM / "experiments" / "loop_quality" / "scenarios"
BORDERLINE = ENGRAM.parent / "nanoGPT" / "data" / "merken" / "borderline_noise.json"
ORGANIC_TRAIN = Path("/tmp/organic_train.json")
MARKDOWN_NOISE = ENGRAM / "experiments" / "data" / "markdown_noise_v6.json"
TRANSCRIPT_LABELS = ENGRAM / "data" / "merken_labels_v7.jsonl"

# Starter signature: first 4 normalized words. Strip punctuation,
# lowercase. Captures "let me check", "after replacing the cache",
# "now update the docs" - the patterns v7 fails on.
STARTER_WORDS = 3


def starter_signature(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words[:STARTER_WORDS])


def load_synthetic() -> list[tuple[str, str]]:
    examples = []
    seen = set()
    for name in [
        "knowledge_update.json",
        "knowledge_update_hard.json",
        "knowledge_update_20topics.json",
        "knowledge_update_50topics.json",
    ]:
        path = SCEN / name
        if not path.exists():
            continue
        for event in json.loads(path.read_text())["events"]:
            text = event["text"].strip()
            if text in seen:
                continue
            seen.add(text)
            label = "NOISE" if event["topic"] == "noise" else "DECISION"
            examples.append((text, label))

    if BORDERLINE.exists():
        for event in json.loads(BORDERLINE.read_text()):
            text = event["text"].strip()
            if text in seen:
                continue
            seen.add(text)
            examples.append((text, "NOISE"))
    return examples


def load_organic() -> list[tuple[str, str]]:
    if not ORGANIC_TRAIN.exists():
        return []
    return [(r["text"].strip(), "DECISION") for r in json.loads(ORGANIC_TRAIN.read_text())]


def load_markdown_noise() -> list[tuple[str, str]]:
    if not MARKDOWN_NOISE.exists():
        return []
    return [(r["text"].strip(), "NOISE") for r in json.loads(MARKDOWN_NOISE.read_text())]


def load_transcripts() -> list[tuple[str, str]]:
    if not TRANSCRIPT_LABELS.exists():
        return []
    out = []
    for line in TRANSCRIPT_LABELS.open():
        try:
            row = json.loads(line)
        except Exception:
            continue
        text = (row.get("text") or "").strip()
        label = row.get("label")
        if not text or label not in ("DECISION", "NOISE"):
            continue
        out.append((text, label))
    return out


def main() -> int:
    sources = {
        "synthetic": load_synthetic(),
        "organic": load_organic(),
        "markdown_noise": load_markdown_noise(),
        "transcripts": load_transcripts(),
    }

    seen = set()
    examples = []
    for src_name, rows in sources.items():
        kept = 0
        for text, label in rows:
            if text in seen:
                continue
            seen.add(text)
            examples.append((text, label, src_name))
            kept += 1
        print(f"{src_name:>14}: {kept} unique kept (of {len(rows)})")

    n_total = len(examples)
    n_dec = sum(1 for _, l, _ in examples if l == "DECISION")
    n_noi = n_total - n_dec
    print(f"\nTotal:  {n_total}  DEC:{n_dec}  NOI:{n_noi}")

    by_starter: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"DECISION": [], "NOISE": []}
    )
    for text, label, _src in examples:
        sig = starter_signature(text)
        if not sig:
            continue
        by_starter[sig][label].append(text)

    n_unique = len(by_starter)
    ambiguous = {
        sig: buckets
        for sig, buckets in by_starter.items()
        if buckets["DECISION"] and buckets["NOISE"]
    }
    n_amb = len(ambiguous)
    n_amb_dec = sum(len(b["DECISION"]) for b in ambiguous.values())
    n_amb_noi = sum(len(b["NOISE"]) for b in ambiguous.values())
    n_pairs_possible = sum(
        len(b["DECISION"]) * len(b["NOISE"]) for b in ambiguous.values()
    )

    print(f"\nUnique starters:     {n_unique}")
    print(f"Ambiguous starters:  {n_amb} (have both DEC and NOI)")
    print(f"  - DEC events under ambiguous starters: {n_amb_dec}")
    print(f"  - NOI events under ambiguous starters: {n_amb_noi}")
    print(f"  - Possible (DEC, NOI) cross pairs:     {n_pairs_possible}")

    top = sorted(
        ambiguous.items(),
        key=lambda kv: -(len(kv[1]["DECISION"]) + len(kv[1]["NOISE"])),
    )[:25]
    print("\nTop-25 ambiguous starters by total events:")
    for sig, buckets in top:
        print(f"  {len(buckets['DECISION']):>3} DEC + "
              f"{len(buckets['NOISE']):>3} NOI -- '{sig}'")

    out = {
        "n_total": n_total,
        "n_dec": n_dec,
        "n_noi": n_noi,
        "starter_words": STARTER_WORDS,
        "n_unique_starters": n_unique,
        "n_ambiguous_starters": n_amb,
        "n_dec_in_ambiguous": n_amb_dec,
        "n_noi_in_ambiguous": n_amb_noi,
        "n_possible_cross_pairs": n_pairs_possible,
        "ambiguous_starters": [
            {
                "starter": sig,
                "n_dec": len(buckets["DECISION"]),
                "n_noi": len(buckets["NOISE"]),
                "dec_texts": buckets["DECISION"],
                "noi_texts": buckets["NOISE"],
            }
            for sig, buckets in sorted(
                ambiguous.items(),
                key=lambda kv: -(len(kv[1]["DECISION"]) + len(kv[1]["NOISE"])),
            )
        ],
    }

    out_path = Path(__file__).parent / "h2_starter_pairs.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
