"""Convert aligned.jsonl into per-token training samples for Phase 3.

Input: scaleup_out/aligned.jsonl (385 cases, 1136 intervention spans).
Output: training_data/midloop_v0.jsonl with one sample per case:

    {
      "case_id": "...",
      "prompt": "<original CHW question>",
      "response_tokens": ["Administer", "IV", "fluids", ...],
      "intervene_labels": [1, 1, 1, 0, 0, 1, ...],  # 1 if token is inside an intervention span
      "intervention_spans": [...],  # original spans for traceability
      "metadata": {...},
    }

The label is per-token binary: 1 if the model_response token at that
position falls inside any intervention span (start <= idx < end). The
spans come from the aligner's token-level diff, so a label of 1 means
"the aligner thinks this token is part of a divergence from the
protocol-derived truth, AND it survived the semantic-equivalence
filter (cosine < threshold)".

This is the dataset shape that a sequence-tagging classifier
(NanoGPTMidloopDecider) will train on. See
`notes/midloop-training-plan.md` for the full architecture sketch.

Usage:
  python -m experiments.midloop_pilot.format_for_training
  python -m experiments.midloop_pilot.format_for_training --in custom.jsonl --out custom_training.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_IN = Path(__file__).parent / "scaleup_out" / "aligned.jsonl"
DEFAULT_OUT = Path(__file__).parent / "training_data" / "midloop_v0.jsonl"

# Word-level whitespace tokenization, MUST match
# merken/training/midloop_dataset.py:tokenize so token indices stay
# consistent across the alignment + training pipeline.
_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def labels_for_response(
    response_tokens: list[str],
    interventions: list[dict],
    *,
    mode: str = "span",
) -> list[int]:
    """Build per-token binary labels.

    Two label shapes (see notes/midloop-training-plan.md for the
    rationale):

    - mode="span" (Option A): a token at position i gets label 1
      if any intervention span (model_token_start, model_token_end)
      contains i (start <= i < end). Produces ~86% positive density
      on the Phase 1 dataset; documented but NOT recommended for
      training (positive class swamps the trivial baseline).

    - mode="boundary" (Option B, RECOMMENDED for v0): label 1 ONLY
      at the start position of each intervention span. Produces
      ~14% positive density; teaches the model "where does an
      error START" -- the actionable runtime decision.
    """
    if mode not in ("span", "boundary"):
        raise ValueError(f"unknown labeling mode: {mode!r}")
    labels = [0] * len(response_tokens)
    for span in interventions:
        start = span.get("model_token_start", 0)
        end = span.get("model_token_end", 0)
        if mode == "boundary":
            if 0 <= start < len(labels):
                labels[start] = 1
        else:  # mode == "span"
            for i in range(max(0, start), min(len(labels), end)):
                labels[i] = 1
    return labels


def convert_case(row: dict, *, labeling: str = "span") -> dict | None:
    """Convert one aligned.jsonl row to one training sample.

    Returns None if the row is malformed or has zero response tokens
    (cannot train on an empty sequence).
    """
    response = row.get("model_response", "")
    if not response:
        return None
    tokens = tokenize(response)
    if not tokens:
        return None
    interventions = row.get("interventions", []) or []
    labels = labels_for_response(tokens, interventions, mode=labeling)
    return {
        "case_id": row.get("case_id", ""),
        "prompt": row.get("prompt", ""),
        "response_tokens": tokens,
        "intervene_labels": labels,
        "labeling_mode": labeling,
        "intervention_spans": [
            {
                "model_token_start": s.get("model_token_start", 0),
                "model_token_end": s.get("model_token_end", 0),
                "truth_text": s.get("truth_text", ""),
                "model_text": s.get("model_text", ""),
                "cosine_sim": s.get("cosine_sim"),
            }
            for s in interventions
        ],
        "metadata": row.get("metadata", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", "--input", dest="input", type=Path,
                    default=DEFAULT_IN, help="aligned.jsonl path")
    ap.add_argument("--out", "--output", dest="output", type=Path,
                    default=DEFAULT_OUT, help="training-format output path")
    ap.add_argument(
        "--labeling", choices=["span", "boundary"], default="span",
        help=(
            "label shape: 'span' marks every token inside an "
            "intervention span (~86%% positive on Phase 1 dataset); "
            "'boundary' marks ONLY the first token of each span "
            "(~14%% positive, recommended for v0 training). See "
            "notes/midloop-training-plan.md for the rationale."
        ),
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_tokens_total = 0
    n_positive_total = 0
    case_label_dist: Counter[int] = Counter()

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warn: skipping malformed line {n_in}: {e}",
                      file=sys.stderr)
                continue
            sample = convert_case(row, labeling=args.labeling)
            if sample is None:
                continue
            n_out += 1
            n_tokens_total += len(sample["response_tokens"])
            n_positive_total += sum(sample["intervene_labels"])
            case_label_dist[sum(sample["intervene_labels"])] += 1
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    pos_rate = (
        n_positive_total / n_tokens_total if n_tokens_total else 0.0
    )

    print(f"\n=== format_for_training ===", file=sys.stderr)
    print(f"  input rows:      {n_in}", file=sys.stderr)
    print(f"  output samples:  {n_out}", file=sys.stderr)
    print(f"  total tokens:    {n_tokens_total}", file=sys.stderr)
    print(f"  positive tokens: {n_positive_total} ({100*pos_rate:.1f}%)",
          file=sys.stderr)
    print(f"  per-case positive distribution (top 10):", file=sys.stderr)
    for k, v in sorted(case_label_dist.items())[:10]:
        print(f"    {k} positive tokens: {v} cases", file=sys.stderr)
    print(f"  output: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
