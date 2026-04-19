"""CLI for `merken-midloop-dataset` -- Phase 1 dataset builder.

Subcommands:

  align    -- run the aligner over a JSONL of (case_id, prompt,
              truth, model_response) and emit a JSONL of intervention
              labels. Requires no large/small model -- consumes
              pre-generated cases.

Future subcommands (deferred until Phase 1 has real protocol input):
  generate-cases     -- large model -> (case, truth) pairs from protocols.
  generate-responses -- small model -> model_response per case.
  build              -- end-to-end: protocols -> cases -> responses -> aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from merken.training.midloop_dataset import Aligner, Case, default_embed_fn


def _stable_case_id(prompt: str, truth: str, model_response: str) -> str:
    """Deterministic id when the input JSONL omits case_id."""
    h = hashlib.sha1()
    for s in (prompt, truth, model_response):
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    return f"case_{h.hexdigest()[:12]}"


def _load_cases(path: Path):
    """Yield Case from a JSONL where each line has at least
    ``truth`` and ``model_response``. ``prompt``, ``case_id``, and
    ``metadata`` are optional.
    """
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"warn: skipping line {lineno}: {e}", file=sys.stderr)
                continue
            truth = row.get("truth")
            model_response = row.get("model_response")
            if truth is None or model_response is None:
                print(
                    f"warn: skipping line {lineno}: missing truth or model_response",
                    file=sys.stderr,
                )
                continue
            yield Case(
                truth=truth,
                model_response=model_response,
                prompt=row.get("prompt", ""),
                case_id=row.get("case_id") or _stable_case_id(
                    row.get("prompt", ""), truth, model_response,
                ),
                metadata=row.get("metadata", {}),
            )


def _serialize_result(result, *, drop_subseq_text: bool) -> dict:
    """Build the JSON output row for one aligned case.

    ``drop_subseq_text=True`` omits truth_text / model_text from the
    intervention list -- useful when the dataset is large and the
    full text is recoverable from (case_id, token_idx). Default
    keeps everything for inspection.
    """
    interventions = []
    for r in result.interventions:
        item = {
            "model_token_start": r.model_start,
            "model_token_end": r.model_end,
            "truth_token_start": r.truth_start,
            "truth_token_end": r.truth_end,
            "cosine_sim": r.cosine_sim,  # None for pure insert/delete
        }
        if not drop_subseq_text:
            item["truth_text"] = r.truth_text
            item["model_text"] = r.model_text
        interventions.append(item)
    return {
        "case_id": result.case.case_id,
        "prompt": result.case.prompt,
        "truth": result.case.truth,
        "model_response": result.case.model_response,
        "metadata": result.case.metadata,
        "n_divergent": len(result.divergent_regions),
        "n_dropped_semantic": len(result.semantic_drops),
        "n_interventions": len(result.interventions),
        "interventions": interventions,
    }


def cmd_align(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    if args.no_embed:
        embed_fn = None
        print(
            f"--no-embed: aligner will skip semantic filter; "
            f"all divergent regions become interventions",
            file=sys.stderr,
        )
    else:
        embed_fn = default_embed_fn(model_name=args.embed_model)

    aligner = Aligner(embed_fn=embed_fn, similarity_threshold=args.threshold)
    n_cases = 0
    n_div_total = 0
    n_drop_total = 0
    n_int_total = 0

    with out_path.open("w") as fout:
        for case in _load_cases(in_path):
            try:
                if args.no_embed:
                    result = aligner.align_only(case)
                else:
                    result = aligner.process(case)
            except Exception as e:
                print(
                    f"warn: case {case.case_id}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue
            row = _serialize_result(result, drop_subseq_text=args.compact)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_cases += 1
            n_div_total += row["n_divergent"]
            n_drop_total += row["n_dropped_semantic"]
            n_int_total += row["n_interventions"]
            if n_cases % 100 == 0:
                print(
                    f"  processed {n_cases} cases "
                    f"({n_int_total} interventions, "
                    f"{n_drop_total} semantic drops)",
                    file=sys.stderr,
                )

    print(
        f"\n=== Phase 1 dataset built ===\n"
        f"  cases:                  {n_cases}\n"
        f"  divergent regions:      {n_div_total}\n"
        f"  semantic drops:         {n_drop_total} "
        f"({100 * n_drop_total / max(n_div_total, 1):.1f}%)\n"
        f"  intervention labels:    {n_int_total}\n"
        f"  output:                 {out_path}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merken-midloop-dataset",
        description=(
            "Build the midloop training dataset from "
            "(truth, model_response) case pairs. See "
            "notes/midloop-spec.md for the design."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    aln = sub.add_parser(
        "align",
        help="align truth vs model + filter semantically equivalent divergences",
    )
    aln.add_argument(
        "--input", "--in", required=True,
        help="JSONL input. Each line needs `truth` and `model_response` "
             "(plus optional `prompt`, `case_id`, `metadata`).",
    )
    aln.add_argument(
        "--output", "--out", required=True,
        help="JSONL output, one alignment result per case.",
    )
    aln.add_argument(
        "--threshold", type=float, default=0.90,
        help="cosine-sim threshold; >= keeps as semantic drop, < keeps as "
             "intervention. Default 0.90 (Jay's spec; calibrate empirically).",
    )
    aln.add_argument(
        "--embed-model", default="BAAI/bge-small-en-v1.5",
        help="embedder model name (default: bge-small)",
    )
    aln.add_argument(
        "--no-embed", action="store_true",
        help="skip semantic filter (all divergences become interventions). "
             "Useful for fast iteration / debugging the diff alone.",
    )
    aln.add_argument(
        "--compact", action="store_true",
        help="omit truth_text / model_text from interventions to "
             "save disk; recoverable via (case_id, token_idx).",
    )
    aln.set_defaults(func=cmd_align)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
