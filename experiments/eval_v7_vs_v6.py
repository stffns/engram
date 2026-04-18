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


def main() -> int:
    v6 = NanoGPTWriteDecider(str(V6_CKPT), str(V6_META))
    v7 = NanoGPTWriteDecider(str(V7_CKPT), str(V7_META))
    v8 = NanoGPTWriteDecider(str(V8_CKPT), str(V8_META))

    scenarios = [
        ("markdown_tables_held_out", SCEN / "markdown_tables_held_out.json"),
        ("organic_val_held_out", SCEN / "organic_val_held_out.json"),
        ("jay_vstash_snapshot", SCEN / "jay_vstash_2026_04_09_snapshot.json"),
        ("knowledge_update_50t", SCEN / "knowledge_update_50topics.json"),
    ]

    print(f"{'scenario':<28} {'n':>5} {'n_dec':>6} {'n_noi':>6} "
          f"{'v6':>7} {'v7':>7} {'v8':>7}")
    print("-" * 80)

    tot = {6: 0, 7: 0, 8: 0}
    total_n = 0

    for name, path in scenarios:
        if not path.exists():
            print(f"{name:<28} MISSING {path}")
            continue
        pairs = list(scenario_pairs(path))
        r6 = run_eval(name, v6, pairs)
        r7 = run_eval(name, v7, pairs)
        r8 = run_eval(name, v8, pairs)
        print(
            f"{name:<28} {r6['n']:>5} {r6['n_dec']:>6} {r6['n_noi']:>6} "
            f"{r6['agreement']*100:>6.1f}% {r7['agreement']*100:>6.1f}% "
            f"{r8['agreement']*100:>6.1f}%"
        )
        # Track integer correct counts to avoid float rounding
        # artifacts in the weighted average.
        tot[6] += r6["correct"]
        tot[7] += r7["correct"]
        tot[8] += r8["correct"]
        total_n += r6["n"]

    # Held-out subsample of labels (20%)
    ho_pairs = labels_held_out_pairs()
    r6 = run_eval("labels_20pct", v6, ho_pairs)
    r7 = run_eval("labels_20pct", v7, ho_pairs)
    r8 = run_eval("labels_20pct", v8, ho_pairs)
    print(
        f"{'labels_20pct (v7/v8 contam)':<28} "
        f"{r6['n']:>5} {r6['n_dec']:>6} {r6['n_noi']:>6} "
        f"{r6['agreement']*100:>6.1f}% {r7['agreement']*100:>6.1f}% "
        f"{r8['agreement']*100:>6.1f}%"
    )

    print()
    print(f"weighted avg over 4 held-out scenarios: "
          f"v6={tot[6]/total_n*100:.1f}%  "
          f"v7={tot[7]/total_n*100:.1f}%  "
          f"v8={tot[8]/total_n*100:.1f}%")

    # Detail per scenario: DEC/NOI recall + FPR
    print()
    print("Per-scenario detail (v6 / v7 / v8):")
    for name, path in scenarios:
        if not path.exists():
            continue
        pairs = list(scenario_pairs(path))
        r6 = run_eval(name, v6, pairs)
        r7 = run_eval(name, v7, pairs)
        r8 = run_eval(name, v8, pairs)
        print(f"  {name}")
        print(f"    DEC_recall:   {r6['dec_recall']*100:>5.1f}%  / "
              f"{r7['dec_recall']*100:>5.1f}%  / "
              f"{r8['dec_recall']*100:>5.1f}%")
        print(f"    NOI_recall:   {r6['noi_recall']*100:>5.1f}%  / "
              f"{r7['noi_recall']*100:>5.1f}%  / "
              f"{r8['noi_recall']*100:>5.1f}%")
        print(f"    FPR (NOI->W): {r6['fpr']*100:>5.1f}%  / "
              f"{r7['fpr']*100:>5.1f}%  / "
              f"{r8['fpr']*100:>5.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())