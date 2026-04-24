"""Diff seed=44 Mode C H18 between retrieval v1 (baseline) and v2
(pure-vec pool + sharegpt_ filter + retrieval-pool=50).

Run after the seed=44 v2 benchmark completes to produce the
per-qid delta table that goes into RESULTS.md.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

V1 = Path(
    "experiments/retrieval/longmemeval/mode_c_runs_v10/"
    "mode_c_n30_seed44_H18_seed44.jsonl"
)
V2 = Path(
    "experiments/retrieval/longmemeval/mode_c_runs_v11/"
    "mode_c_n30_seed44_H18_retrieval_v2_seed44.jsonl"
)


def _load(p: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in p.open():
        r = json.loads(line)
        if "mode_c" not in r:
            continue
        out[r["question_id"]] = r
    return out


def main() -> None:
    v1 = _load(V1)
    v2 = _load(V2)

    def _ok(v: str) -> bool:
        return v in ("supports", "partial")

    keys = sorted(set(v1) | set(v2))

    flips_ok = []
    flips_bad = []
    stayed_fail = []
    stayed_ok = []

    for k in keys:
        if k not in v2:
            continue
        v1v = v1[k]["mode_c"]["oracle"]["verdict"]
        v2v = v2[k]["mode_c"]["oracle"]["verdict"]
        qt = v2[k]["question_type"]
        tup = (k, qt, v1v, v2v)
        if not _ok(v1v) and _ok(v2v):
            flips_ok.append(tup)
        elif _ok(v1v) and not _ok(v2v):
            flips_bad.append(tup)
        elif _ok(v1v) and _ok(v2v):
            stayed_ok.append(tup)
        else:
            stayed_fail.append(tup)

    n_v1 = sum(1 for k in v1 if _ok(v1[k]["mode_c"]["oracle"]["verdict"]))
    n_v2 = sum(1 for k in v2 if _ok(v2[k]["mode_c"]["oracle"]["verdict"]))
    total = len(v2)
    print(f"v1 correct: {n_v1}/{len(v1)}")
    print(f"v2 correct: {n_v2}/{total}")
    print(f"delta:      {n_v2 - n_v1:+d}")
    print(f"flips_ok:   {len(flips_ok)}")
    print(f"flips_bad:  {len(flips_bad)}")
    print(f"stayed_fail:{len(stayed_fail)}")
    print(f"stayed_ok:  {len(stayed_ok)}")

    print("\nfail-type distribution (v1 -> v2):")
    dist_v1 = Counter(
        v1[k]["mode_c"]["oracle"]["verdict"] for k in v1
    )
    dist_v2 = Counter(
        v2[k]["mode_c"]["oracle"]["verdict"] for k in v2
    )
    for v in sorted(set(dist_v1) | set(dist_v2)):
        print(f"  {v}: {dist_v1.get(v,0)} -> {dist_v2.get(v,0)}")

    print("\nflips_ok (v1 fail -> v2 correct):")
    for k, qt, v1v, v2v in flips_ok:
        print(f"  {k}\t{qt}\t{v1v} -> {v2v}")
    print("\nflips_bad (v1 correct -> v2 fail):")
    for k, qt, v1v, v2v in flips_bad:
        print(f"  {k}\t{qt}\t{v1v} -> {v2v}")
    print("\nstill failing:")
    for k, qt, v1v, v2v in stayed_fail:
        print(f"  {k}\t{qt}\t{v1v} -> {v2v}")


if __name__ == "__main__":
    main()
