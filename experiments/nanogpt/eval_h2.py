# ruff: noqa: I001, E402
"""H2 / H2b acceptance eval -- v7 vs v10_contrastive vs v11_infonce.

For each contrastive variant present on disk:
  1. DEC false-negative recovery on the 157 DECs in
     merken_labels_v7.jsonl (in-training; "where does v7 still miss?"
     probe). H2 bar: >=50 recovered. H2b relaxed bar: >=30 recovered.
  2. markdown_tables_held_out FPR. H2 bar: <=10%. H2b adds OOD bars:
     organic_val and jay_vstash_decontam agreement >=85%.

Mirrors the full scenario table from `experiments/eval_v7_vs_v6.py`
so regressions on other scenarios surface immediately. Writes one
results JSON per detected contrastive variant.
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

CONTRASTIVE_VARIANTS = [
    ("v10_contrastive", "h2",  "h2_results.json"),
    ("v11_infonce",     "h2b", "h2b_results.json"),
]


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


def _decision_rule(label, recovered, scenarios_per_scen):
    """Decision rules:
       h2  -- recovered >= 50 AND markdown FPR <= 10%.
       h2b -- recovered >= 30 AND OOD agreement on organic_val
              and jay_vstash_decontam both >= 85%.
    """
    markdown_fpr = scenarios_per_scen.get("markdown_tables_held_out", {}).get("variant", {}).get("fpr", 1.0)
    organic = scenarios_per_scen.get("organic_val_held_out", {}).get("variant", {}).get("agreement", 0.0)
    vstash = scenarios_per_scen.get("jay_vstash_snapshot_decontam", {}).get("variant", {}).get("agreement", 0.0)
    if label == "h2":
        b1 = recovered >= 50
        b2 = markdown_fpr <= 0.10
        return {
            "bar_1_recovered_fn_ge_50": b1,
            "bar_2_markdown_fpr_le_10pct": b2,
            "verdict": "ACCEPTED" if (b1 and b2) else "REJECTED",
        }
    # h2b
    b1 = recovered >= 30
    b2 = organic >= 0.85
    b3 = vstash >= 0.85
    return {
        "bar_1_recovered_fn_ge_30": b1,
        "bar_2_organic_val_ge_85pct": b2,
        "bar_3_jay_vstash_decontam_ge_85pct": b3,
        "verdict": "ACCEPTED" if (b1 and b2 and b3) else "REJECTED",
    }


def main() -> int:
    v7 = NanoGPTWriteDecider(str(V7_CKPT), str(V7_META))
    ld_pairs = labels_pairs()
    r_v7 = run_eval(v7, ld_pairs)
    v7_fn_set = set(r_v7["fn_texts"])
    print(f"v7 baseline: DEC recall {r_v7['dec_recall']*100:.1f}% "
          f"({len(v7_fn_set)}/{r_v7['n_dec']} FN), "
          f"NOI recall {r_v7['noi_recall']*100:.1f}%")

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

    any_present = False
    for vname, label, out_name in CONTRASTIVE_VARIANTS:
        ckpt = NANOGPT / f"out-merken-bpe-{vname}" / "ckpt.pt"
        meta = NANOGPT / "data" / f"merken_bpe_{vname}" / "meta.pkl"
        if not ckpt.exists():
            print(f"\n[skip] {vname} ckpt not found at {ckpt}")
            continue
        any_present = True
        decider = NanoGPTWriteDecider(str(ckpt), str(meta))

        r_var = run_eval(decider, ld_pairs)
        var_fn_set = set(r_var["fn_texts"])
        recovered = v7_fn_set - var_fn_set
        newly_lost = var_fn_set - v7_fn_set
        print(f"\n=== {vname} ({label.upper()}) ===")
        print(f"labels-157  DEC recall {r_var['dec_recall']*100:.1f}% "
              f"({len(var_fn_set)}/{r_var['n_dec']} FN) | "
              f"NOI {r_var['noi_recall']*100:.1f}%")
        print(f"  recovered (v7->var FN -> ok):  {len(recovered)}")
        print(f"  newly lost (v7 ok -> var FN):  {len(newly_lost)}")

        per_scen = {}
        print(f"\n  {'scenario':<36} {'n':>4} {'v7':>7} {'fpr':>6}   "
              f"{'var':>7} {'fpr':>6}")
        for name, path in scenarios:
            if not path.exists():
                continue
            pairs = list(scenario_pairs(path))
            s7 = run_eval(v7, pairs)
            sv = run_eval(decider, pairs)
            per_scen[name] = {
                "n": s7["n"], "n_dec": s7["n_dec"], "n_noi": s7["n_noi"],
                "v7": {k: s7[k] for k in ("agreement", "dec_recall",
                                          "noi_recall", "fpr")},
                "variant": {k: sv[k] for k in ("agreement", "dec_recall",
                                               "noi_recall", "fpr")},
            }
            print(f"  {name:<36} {s7['n']:>4} "
                  f"{s7['agreement']*100:>6.1f}% {s7['fpr']*100:>5.1f}% "
                  f"  {sv['agreement']*100:>6.1f}% {sv['fpr']*100:>5.1f}%")

        rule = _decision_rule(label, len(recovered), per_scen)
        print(f"\n  Decision ({label.upper()}):")
        for k, v in rule.items():
            if k == "verdict":
                continue
            print(f"    {k:<40} {v}")
        print(f"    verdict: {rule['verdict']}")

        out = {
            "variant": vname,
            "rule_label": label,
            "verdict": rule["verdict"],
            "labels_157_DEC": {
                "v7_dec_recall": r_v7["dec_recall"],
                "variant_dec_recall": r_var["dec_recall"],
                "v7_noi_recall": r_v7["noi_recall"],
                "variant_noi_recall": r_var["noi_recall"],
                "v7_fn_count": len(v7_fn_set),
                "variant_fn_count": len(var_fn_set),
                "fn_recovered": sorted(recovered),
                "fn_newly_lost": sorted(newly_lost),
                "n_recovered": len(recovered),
                "n_newly_lost": len(newly_lost),
            },
            "scenarios": per_scen,
            "decision_rule": rule,
        }
        out_path = Path(__file__).parent / out_name
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  saved to {out_path}")

    if not any_present:
        raise SystemExit("no contrastive variants found on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
