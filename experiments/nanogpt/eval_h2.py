# ruff: noqa: I001, E402
"""H2 acceptance eval -- v7 vs v10_contrastive.

Two reported metrics (both in HYPOTHESES.md H2 decision rule):

  1. DEC false-negative recovery on the 157 DECs in
     merken_labels_v7.jsonl. v7 trained on these, so this is a
     "where does v7 still miss?" probe, not held-out. The exact
     acceptance bar: v10 catches >= 50 of v7's false negatives.

  2. markdown_tables_held_out FPR (<= 10% accept).

The script also mirrors the full scenario table from
`experiments/eval_v7_vs_v6.py` so regressions on other scenarios show
up immediately. Writes `experiments/nanogpt/h2_results.json`.
"""

from __future__ import annotations

import torch  # noqa: F401 (first, Mistake #10)

import json
import os
import random
from pathlib import Path

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies.types import Event, WriteContext


ENGRAM = Path(__file__).resolve().parent.parent.parent
NANOGPT = Path(os.environ.get("NANOGPT_REPO") or (ENGRAM.parent / "nanoGPT"))
SCEN = ENGRAM / "experiments" / "loop_quality" / "scenarios"
LABELS = ENGRAM / "data" / "merken_labels_v7.jsonl"

V7_CKPT = NANOGPT / "out-merken-bpe-v7" / "ckpt.pt"
V7_META = NANOGPT / "data" / "merken_bpe_v7" / "meta.pkl"
V10_CKPT = NANOGPT / "out-merken-bpe-v10_contrastive" / "ckpt.pt"
V10_META = NANOGPT / "data" / "merken_bpe_v10_contrastive" / "meta.pkl"


def scenario_pairs(path: Path):
    data = json.loads(path.read_text())
    for event in data.get("events", []):
        text = event.get("text", "").strip()
        if not text:
            continue
        yield text, event.get("topic", "") != "noise"


def run_eval(decider, pairs):
    ctx = WriteContext(project="eval_h2")
    tp = tn = fp = fn = 0
    fn_texts = []
    for text, is_dec in pairs:
        d = decider.decide(Event(text=text), ctx)
        if is_dec:
            if d.write:
                tp += 1
            else:
                fn += 1
                fn_texts.append(text)
        else:
            if d.write:
                fp += 1
            else:
                tn += 1
    total = tp + tn + fp + fn
    return {
        "n": total,
        "n_dec": tp + fn,
        "n_noi": tn + fp,
        "agreement": (tp + tn) / total if total else 0.0,
        "dec_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "noi_recall": tn / (tn + fp) if (tn + fp) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fn_texts": fn_texts,
    }


def labels_pairs():
    pairs = []
    for line in LABELS.open():
        row = json.loads(line)
        text = (row.get("text") or "").strip()
        label = row.get("label")
        if label not in ("DECISION", "NOISE") or not text:
            continue
        pairs.append((text, label == "DECISION"))
    return pairs


def main() -> int:
    if not V10_CKPT.exists():
        raise SystemExit(f"v10 ckpt not found at {V10_CKPT}")
    v7 = NanoGPTWriteDecider(str(V7_CKPT), str(V7_META))
    v10 = NanoGPTWriteDecider(str(V10_CKPT), str(V10_META))

    # 1) Full transcript-label set -- v7's FN recovery on DEC.
    ld_pairs = labels_pairs()
    r_v7 = run_eval(v7, ld_pairs)
    r_v10 = run_eval(v10, ld_pairs)

    v7_fn_set = set(r_v7["fn_texts"])
    v10_fn_set = set(r_v10["fn_texts"])
    recovered = v7_fn_set - v10_fn_set
    newly_lost = v10_fn_set - v7_fn_set
    print(f"\n=== Labels-157-DEC  (in-training for both v7 and v10) ===")
    print(f"v7   DEC recall: {r_v7['dec_recall']*100:.1f}% "
          f"FN={len(v7_fn_set)} / {r_v7['n_dec']}")
    print(f"v10  DEC recall: {r_v10['dec_recall']*100:.1f}% "
          f"FN={len(v10_fn_set)} / {r_v10['n_dec']}")
    print(f"  v7 -> v10 FN recovered:   {len(recovered)}  "
          f"(H2 acceptance bar: >= 50)")
    print(f"  v7 -> v10 FN newly lost:  {len(newly_lost)}")

    print(f"v7   NOI recall: {r_v7['noi_recall']*100:.1f}%")
    print(f"v10  NOI recall: {r_v10['noi_recall']*100:.1f}%")

    # 2) Full scenario suite (same as eval_v7_vs_v6).
    scenarios = [
        ("markdown_tables_held_out", SCEN / "markdown_tables_held_out.json"),
        ("organic_val_held_out", SCEN / "organic_val_held_out.json"),
        ("jay_vstash_snapshot_decontam",
         SCEN / "jay_vstash_2026_04_09_snapshot_decontam.json"),
        ("knowledge_update_50t (TRAINING)",
         SCEN / "knowledge_update_50topics.json"),
        ("analytics_project", SCEN / "analytics_project.json"),
        ("session_2026_04_09", SCEN / "session_2026_04_09.json"),
        ("bilingual_es_en_2026_04_14",
         SCEN / "bilingual_es_en_2026_04_14.json"),
        ("noisy_agent_stream", SCEN / "noisy_agent_stream.json"),
        ("disjoint_noise_heavy_holdout",
         SCEN / "disjoint_noise_heavy_holdout.json"),
    ]
    per_scen = {}
    print("\n=== Scenario suite ===")
    print(f"{'scenario':<36} {'n':>4} {'v7_agree':>9} {'v7_fpr':>8} "
          f"{'v10_agree':>10} {'v10_fpr':>8}")
    print("-" * 80)
    for name, path in scenarios:
        if not path.exists():
            continue
        pairs = list(scenario_pairs(path))
        s7 = run_eval(v7, pairs)
        s10 = run_eval(v10, pairs)
        per_scen[name] = {
            "n": s7["n"], "n_dec": s7["n_dec"], "n_noi": s7["n_noi"],
            "v7": {k: s7[k] for k in ("agreement", "dec_recall", "noi_recall", "fpr")},
            "v10": {k: s10[k] for k in ("agreement", "dec_recall", "noi_recall", "fpr")},
        }
        print(f"{name:<36} {s7['n']:>4} "
              f"{s7['agreement']*100:>8.1f}% {s7['fpr']*100:>7.1f}% "
              f"{s10['agreement']*100:>9.1f}% {s10['fpr']*100:>7.1f}%")

    # H2 decision rule evaluation.
    markdown_fpr_v10 = per_scen.get("markdown_tables_held_out", {}).get("v10", {}).get("fpr", 1.0)
    passes_bar_1 = len(recovered) >= 50
    passes_bar_2 = markdown_fpr_v10 <= 0.10
    print(f"\n=== H2 Decision ===")
    print(f"  Bar 1 (>= 50 FN recovered):         "
          f"{len(recovered)} {'PASS' if passes_bar_1 else 'FAIL'}")
    print(f"  Bar 2 (markdown FPR <= 10%):        "
          f"{markdown_fpr_v10*100:.1f}% {'PASS' if passes_bar_2 else 'FAIL'}")
    verdict = "ACCEPTED" if (passes_bar_1 and passes_bar_2) else "REJECTED"
    print(f"  Overall: {verdict}")

    out = {
        "h2_verdict": verdict,
        "labels_157_DEC": {
            "v7_dec_recall": r_v7["dec_recall"],
            "v10_dec_recall": r_v10["dec_recall"],
            "v7_noi_recall": r_v7["noi_recall"],
            "v10_noi_recall": r_v10["noi_recall"],
            "v7_fn_count": len(v7_fn_set),
            "v10_fn_count": len(v10_fn_set),
            "fn_recovered": sorted(recovered),
            "fn_newly_lost": sorted(newly_lost),
            "n_recovered": len(recovered),
            "n_newly_lost": len(newly_lost),
        },
        "scenarios": per_scen,
        "decision_rule": {
            "bar_1_recovered_fn_ge_50": passes_bar_1,
            "bar_2_markdown_fpr_le_10pct": passes_bar_2,
        },
    }
    out_path = Path(__file__).parent / "h2_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
