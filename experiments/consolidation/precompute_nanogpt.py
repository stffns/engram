# ruff: noqa: I001, E402
"""Pre-compute nanoGPT write decisions offline, save to JSON.

Runs the nanoGPT classifier over every event in a scenario in a
*torch-only* process (no vstash, no fastembed) and saves each decision
as JSON. The end-to-end write_filter benchmark then reads the JSON via
`PrecomputedWriteDecider`, keeping torch out of the process that also
loads vstash/fastembed.

Why this split: in a single Python process, torch and fastembed/ONNX
corrupt each other's runtime state on macOS, even in a freshly-forked
subprocess (see notes/nanogpt-training-log.md Mistake #10). Running
inference up-front, persisting the result, and reloading it later is
the cheapest escape hatch -- inference is 1 ms per event.

Usage:
    PYTHONPATH=. python -m experiments.consolidation.precompute_nanogpt \\
        --model char --scenario knowledge_update_50topics.json \\
        --out /tmp/decisions_char.json
"""

from __future__ import annotations

# Torch MUST load before anything that pulls in vstash/fastembed; when the
# order is reversed the PyTorch runtime segfaults on first model construction
# (macOS only, see notes/nanogpt-training-log.md Mistake #10).
import torch  # noqa: F401

import argparse
import json
import time
from pathlib import Path

NANOGPT_DIR = Path(__file__).parent.parent.parent.parent / "nanoGPT"
SCENARIO_DIR = Path(__file__).parent.parent / "loop_quality" / "scenarios"

MODEL_PATHS = {
    "char": (NANOGPT_DIR / "out-merken" / "ckpt.pt",
             NANOGPT_DIR / "data" / "merken" / "meta.pkl"),
    "bpe": (NANOGPT_DIR / "out-merken-bpe" / "ckpt.pt",
            NANOGPT_DIR / "data" / "merken_bpe" / "meta.pkl"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODEL_PATHS), required=True)
    p.add_argument("--scenario", required=True,
                   help="File name under experiments/loop_quality/scenarios/")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    ckpt, meta = MODEL_PATHS[args.model]
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    from merken.classifiers.nanogpt import NanoGPTWriteDecider
    from merken.policies import Event, WriteContext

    decider = NanoGPTWriteDecider(ckpt, meta, confidence_threshold=args.threshold)
    ctx = WriteContext(project="precompute")

    scenario_path = SCENARIO_DIR / args.scenario
    scenario = json.loads(scenario_path.read_text())
    events = scenario["events"]

    decisions = {}
    t0 = time.perf_counter()
    for e in events:
        d = decider.decide(Event(text=e["text"]), ctx)
        decisions[e["id"]] = {
            "write": d.write,
            "reason": d.reason,
            "confidence": d.confidence,
            "topic": e["topic"],
        }
    elapsed = time.perf_counter() - t0

    written = sum(1 for d in decisions.values() if d["write"])
    signal = sum(1 for e in events if e["topic"] != "noise")
    true_pos = sum(1 for e in events
                   if e["topic"] != "noise" and decisions[e["id"]]["write"])
    false_pos = sum(1 for e in events
                    if e["topic"] == "noise" and decisions[e["id"]]["write"])

    meta_out = {
        "model": args.model,
        "threshold": args.threshold,
        "checkpoint": str(ckpt),
        "scenario": args.scenario,
        "n_events": len(events),
        "n_written": written,
        "n_signal": signal,
        "recall_on_signal": true_pos / signal if signal else 0.0,
        "false_positive_rate_on_noise": (
            false_pos / (len(events) - signal) if len(events) > signal else 0.0
        ),
        "inference_seconds": elapsed,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"meta": meta_out, "decisions": decisions},
                                   indent=2))
    print(json.dumps(meta_out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
