# ruff: noqa: I001, E402
"""Evaluate nanoGPT v7 vs v6 on held-out scenarios + the transcript labels.

Writes a side-by-side table. Four held-out evals are properly
held-out (v7 training didn't see them). The JSONL-labels eval uses a
seed-matched 10% split to avoid testing v7 on its own training set.

Metrics per scenario:
  - agreement: shadow decision vs ground-truth label
  - DEC_recall: of DECISION events, how many did shadow correctly write
  - NOISE_recall: of NOISE events, how many did shadow correctly skip
  - DEC_FPR: rate of DECISION-writes among NOISE ground truth
             (lower is better)
"""

from __future__ import annotations

import torch  # noqa: F401 (first, Mistake #10)

import json
import os
import random
from pathlib import Path

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies.types import Event, WriteContext


ENGRAM = Path(__file__).resolve().parent.parent
NANOGPT = Path(
    os.environ.get("NANOGPT_REPO")
    or (ENGRAM.parent / "nanoGPT")
)
SCEN = ENGRAM / "experiments" / "loop_quality" / "scenarios"
LABELS = ENGRAM / "data" / "merken_labels_v7.jsonl"

V6_CKPT = NANOGPT / "out-merken-bpe-v6" / "ckpt.pt"
V6_META = NANOGPT / "data" / "merken_bpe_v6" / "meta.pkl"
V7_CKPT = NANOGPT / "out-merken-bpe-v7" / "ckpt.pt"
V7_META = NANOGPT / "data" / "merken_bpe_v7" / "meta.pkl"
V8_CKPT = NANOGPT / "out-merken-bpe-v8" / "ckpt.pt"
V8_META = NANOGPT / "data" / "merken_bpe_v8" / "meta.pkl"
V9_CKPT = NANOGPT / "out-merken-bpe-v9" / "ckpt.pt"
V9_META = NANOGPT / "data" / "merken_bpe_v9" / "meta.pkl"


def scenario_pairs(path: Path):
    """Yield (text, is_decision) per event in a scenario JSON."""
    data = json.loads(path.read_text())
    for event in data.get("events", []):
        text = event.get("text", "").strip()
        if not text:
            continue
        topic = event.get("topic", "")
        is_dec = topic != "noise"
        yield text, is_dec


def labels_held_out_pairs():
    """Recreate the 10% val split used by prepare.py for fair eval.

    prepare.py does `random.seed(2026)` then `random.shuffle(examples)`
    and takes 90/10. The examples list has all v6 sources first, then
    transcripts. We can't trivially replicate the full ordering, so we
    use a different deterministic subsample: 20% of the 1026 labels
    picked with a distinct seed. v7 may have seen some of them in
    train; we report this score with a caveat.

    For fairness, cross-check with the 4 held-out scenarios which are
    properly out-of-train.
    """
    random.seed(12345)  # NOT 2026
    rows = [json.loads(l) for l in open(LABELS)]
    random.shuffle(rows)
    n_test = int(len(rows) * 0.2)
    sample = rows[:n_test]
    pairs = []
    for r in sample:
        text = r.get("text") or ""
        label = r.get("label")
        if label not in ("DECISION", "NOISE") or not text.strip():
            continue
        pairs.append((text, label == "DECISION"))
    return pairs


def run_eval(name: str, decider: NanoGPTWriteDecider, pairs: list[tuple[str, bool]]):
    ctx = WriteContext(project="eval")
    tp = tn = fp = fn = 0
    for text, is_dec in pairs:
        d = decider.decide(Event(text=text), ctx)
        if is_dec:
            if d.write:
                tp += 1
            else:
                fn += 1
        else:
            if d.write:
                fp += 1
            else:
                tn += 1
    total = tp + tn + fp + fn
    correct = tp + tn
    agreement = correct / total if total else 0.0
    dec_recall = tp / (tp + fn) if (tp + fn) else 0.0
    noi_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "name": name,
        "n": total,
        "correct": correct,  # integer TP+TN, safe for weighted avg
        "n_dec": tp + fn,
        "n_noi": tn + fp,
        "agreement": agreement,
        "dec_recall": dec_recall,
        "noi_recall": noi_recall,
        "fpr": fpr,
    }


def _load_if_present(ckpt: Path, meta: Path):
    if ckpt.exists() and meta.exists():
        return NanoGPTWriteDecider(str(ckpt), str(meta))
    return None


def main() -> int:
    # Optional models guarded so the script runs on machines that
    # only have a subset of checkpoints on disk.
    v6 = _load_if_present(V6_CKPT, V6_META)
    v7 = _load_if_present(V7_CKPT, V7_META)
    v8 = _load_if_present(V8_CKPT, V8_META)
    v9 = _load_if_present(V9_CKPT, V9_META)
    versions = [("v6", v6), ("v7", v7), ("v8", v8), ("v9", v9)]
    present = [(n, d) for n, d in versions if d is not None]
    if not present:
        raise SystemExit("no nanoGPT ckpts found; set NANOGPT_REPO env")

    scenarios = [
        ("markdown_tables_held_out", SCEN / "markdown_tables_held_out.json"),
        ("organic_val_held_out", SCEN / "organic_val_held_out.json"),
        ("jay_vstash_snapshot", SCEN / "jay_vstash_2026_04_09_snapshot.json"),
        ("jay_vstash_snapshot_decontam",
            SCEN / "jay_vstash_2026_04_09_snapshot_decontam.json"),
        # NOTE: knowledge_update_50t is 100% TRAINING DATA for v4-v7;
        # reported here as a training-set diagnostic, NOT a held-out.
        ("knowledge_update_50t (TRAINING)",
            SCEN / "knowledge_update_50topics.json"),
        ("analytics_project", SCEN / "analytics_project.json"),
        ("session_2026_04_09", SCEN / "session_2026_04_09.json"),
        ("bilingual_es_en_2026_04_14", SCEN / "bilingual_es_en_2026_04_14.json"),
        ("noisy_agent_stream", SCEN / "noisy_agent_stream.json"),
        # Disjoint-from-training noise-heavy held-out (H_disjoint_holdout).
        ("disjoint_noise_heavy_holdout",
            SCEN / "disjoint_noise_heavy_holdout.json"),
    ]

    col_hdr = "".join(f"{name:>8}" for name, _ in present)
    print(f"{'scenario':<36} {'n':>5} {'n_dec':>6} {'n_noi':>6} {col_hdr}")
    print("-" * (60 + 8 * len(present)))

    totals = {name: 0 for name, _ in present}
    total_n = 0
    per_scenario: dict[str, dict] = {}

    for name, path in scenarios:
        if not path.exists():
            print(f"{name:<36} MISSING {path.name}")
            continue
        pairs = list(scenario_pairs(path))
        rs = {n: run_eval(name, d, pairs) for n, d in present}
        row = f"{name:<36} {rs[present[0][0]]['n']:>5} {rs[present[0][0]]['n_dec']:>6} {rs[present[0][0]]['n_noi']:>6}"
        for n, _ in present:
            row += f"  {rs[n]['agreement']*100:>5.1f}%"
        print(row)
        per_scenario[name] = {
            "n": rs[present[0][0]]["n"],
            "n_dec": rs[present[0][0]]["n_dec"],
            "n_noi": rs[present[0][0]]["n_noi"],
            "per_version": {n: {
                "agreement": rs[n]["agreement"],
                "dec_recall": rs[n]["dec_recall"],
                "noi_recall": rs[n]["noi_recall"],
                "fpr": rs[n]["fpr"],
            } for n, _ in present},
        }
        # Weighted avg skips training-set and contaminated scenarios to
        # keep the number meaningful.
        if not ("TRAINING" in name or "jay_vstash_snapshot" == name):
            for n, _ in present:
                totals[n] += rs[n]["correct"]
            total_n += rs[present[0][0]]["n"]

    # Held-out labels subsample (contaminated for v7/v8/v9 -- diagnostic only)
    ho_pairs = labels_held_out_pairs()
    rs_ho = {n: run_eval("labels_20pct", d, ho_pairs) for n, d in present}
    row = (f"{'labels_20pct (v7/v8/v9 contam)':<36} "
           f"{rs_ho[present[0][0]]['n']:>5} {rs_ho[present[0][0]]['n_dec']:>6} "
           f"{rs_ho[present[0][0]]['n_noi']:>6}")
    for n, _ in present:
        row += f"  {rs_ho[n]['agreement']*100:>5.1f}%"
    print(row)

    print()
    weights = "  ".join(
        f"{n}={totals[n]/total_n*100:.1f}%" for n, _ in present
    ) if total_n else "(no clean scenarios)"
    print(f"weighted avg over CLEAN held-out scenarios: {weights}")

    # Save full results to JSON artifact (addresses PR #18 review C2).
    import json
    art = {
        "versions_evaluated": [n for n, _ in present],
        "scenarios": per_scenario,
        "labels_20pct_subsample": {n: {
            "agreement": rs_ho[n]["agreement"],
            "n": rs_ho[n]["n"],
        } for n, _ in present},
        "weighted_avg_clean_holdout": {
            n: totals[n] / total_n if total_n else None for n, _ in present
        },
        "notes": {
            "knowledge_update_50t": "TRAINING DATA for v4-v7; diagnostic only",
            "jay_vstash_snapshot": "CONTAMINATED (13/20 in training). Use _decontam",
            "labels_20pct_subsample": "v7/v8/v9 saw these in training",
        },
    }
    art_path = ENGRAM / "experiments" / "nanogpt" / "eval_v6_to_v9.json"
    art_path.write_text(json.dumps(art, indent=2))
    print(f"saved to {art_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())