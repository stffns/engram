"""Merge the v0 MedLocal dataset with the HF-guidelines expansion.

Produces ``training_data/midloop_v1.jsonl`` whose rows come from both
sources, with metadata.source distinguishing them. Idempotent --
safe to re-run (writes to a fresh file).

Schema is identical to v0 (same ``format_for_training`` output), so
the existing split + prepare + train pipeline works unchanged. The
only thing that grows is ``metadata.protocol_id`` diversity.

Usage:
  python -m experiments.midloop_pilot.merge_datasets
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
V0 = HERE / "training_data" / "midloop_v0.jsonl"
V1_HF = HERE / "scaleup_out_hf" / "midloop_v1_hf.jsonl"
OUT = HERE / "training_data" / "midloop_v1.jsonl"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    v0 = _load(V0)
    v1_hf = _load(V1_HF)
    print(f"v0:    {len(v0)} cases")
    print(f"v1_hf: {len(v1_hf)} cases")

    # Tag source in metadata so downstream scripts can stratify if they want.
    for r in v0:
        r.setdefault("metadata", {}).setdefault("dataset_source", "medlocal_v0")
    for r in v1_hf:
        r.setdefault("metadata", {}).setdefault("dataset_source", "hf_guidelines_v1")

    # Collision detection: two sources with the same protocol_id would mix
    # CHW / HF content under one "protocol" name, silently corrupting the
    # protocol-level train/test split (no leak but the semantics breaks).
    v0_pids = {r["metadata"]["protocol_id"] for r in v0}
    v1_pids = {r["metadata"]["protocol_id"] for r in v1_hf}
    overlap = v0_pids & v1_pids
    if overlap:
        raise SystemExit(
            f"protocol_id collision between v0 and v1_hf "
            f"({len(overlap)} ids). Samples: {sorted(overlap)[:5]}"
        )

    # case_id uniqueness: downstream eval indexes by case_id so duplicates
    # quietly over-count a test case.
    case_ids = [r["case_id"] for r in v0 + v1_hf]
    if len(set(case_ids)) != len(case_ids):
        dups = [(k, v) for k, v in Counter(case_ids).items() if v > 1]
        raise SystemExit(
            f"duplicate case_ids ({len(dups)} collisions). "
            f"Samples: {dups[:5]}"
        )

    merged = v0 + v1_hf
    print(f"merged: {len(merged)} cases")

    # protocol_id distribution in the merged set
    pids = Counter(r["metadata"]["protocol_id"] for r in merged)
    print(f"unique protocols: {len(pids)}")
    sizes = Counter(pids.values())
    print(f"cases-per-protocol distribution:")
    for count, n_protos in sorted(sizes.items()):
        print(f"  {count} cases: {n_protos} protocols")

    # source breakdown
    src = Counter(r["metadata"].get("dataset_source", "?") for r in merged)
    print(f"by dataset_source: {dict(src)}")

    # response-token sanity
    tot_tok = sum(len(r["response_tokens"]) for r in merged)
    tot_pos = sum(sum(r["intervene_labels"]) for r in merged)
    print(f"total response tokens: {tot_tok}")
    print(f"total positive labels: {tot_pos} ({tot_pos/tot_tok*100:.2f}%)")

    with OUT.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
