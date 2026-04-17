"""Cascade benchmark: nanoGPT v6 gate -> LLM arbiter.

Runs ``ChainedWriteDecider(NanoGPTWriteDecider(v6), LLMWriteDecider)``
over the held-out scenarios and reports recall / FPR / per-event
latency. The cascade is a production-friendly composition:

- nanoGPT v6 fires at 1 ms / event. If it says SKIP, we return
  immediately -- the LLM never runs.
- If v6 says WRITE, the LLM runs as final arbiter at 3-6 s / event.

Expected profile (from the 2026-04-17 bench):

- On scenarios with obvious noise (knowledge_update_*), v6 catches
  99%+ of the noise at 1 ms each -- cascade latency per NOISE event
  stays near 1 ms.
- On the markdown blind spot (where v6 is permissive), every event
  reaches the LLM -- cascade quality equals the LLM's (100% recall,
  16.7% FPR with Gemma 3 1B-IT) but latency stays at LLM levels.

Usage:

    PYTHONPATH=. python -m experiments.consolidation.cascade_bench \\
        --llm google/gemma-3-1b-it \\
        --scenarios markdown_tables_held_out.json \\
                    organic_val_held_out.json \\
                    jay_vstash_2026_04_09_snapshot.json
"""

from __future__ import annotations

# Torch must load before vstash/fastembed (Mistake #10).
import torch  # noqa: F401, E402

import argparse
import json
import time
from pathlib import Path

from merken.classifiers.llm import LLMWriteDecider
from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.policies import ChainedWriteDecider, Event, WriteContext

NANOGPT_DIR = Path(__file__).parent.parent.parent.parent / "nanoGPT"
SCENARIO_DIR = Path(__file__).parent.parent / "loop_quality" / "scenarios"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gate-ckpt",
        type=Path,
        default=NANOGPT_DIR / "out-merken-bpe-v6" / "ckpt.pt",
    )
    p.add_argument(
        "--gate-meta",
        type=Path,
        default=NANOGPT_DIR / "data" / "merken_bpe_v6" / "meta.pkl",
    )
    p.add_argument(
        "--llm",
        default="google/gemma-3-1b-it",
        help="HF model id or local path for the cascade arbiter",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="scenario JSON filenames under loop_quality/scenarios/",
    )
    args = p.parse_args()

    print(f"gate: {args.gate_ckpt}", flush=True)
    gate = NanoGPTWriteDecider(args.gate_ckpt, args.gate_meta)
    print(f"arbiter: {args.llm}", flush=True)
    arbiter = LLMWriteDecider(model_name=args.llm, device=args.device)
    cascade = ChainedWriteDecider(gate, arbiter)
    print(f"cascade: {cascade.name}\n", flush=True)

    ctx = WriteContext(project="cascade_bench")
    for scen_name in args.scenarios:
        path = SCENARIO_DIR / scen_name
        scen = json.loads(path.read_text())
        events = scen["events"]

        t0 = time.perf_counter()
        results = []
        gate_skips = 0
        for e in events:
            res_gate = gate.decide(Event(text=e["text"]), ctx)
            if not res_gate.write:
                gate_skips += 1
            # Run cascade (which re-runs gate internally).
            res = cascade.decide(Event(text=e["text"]), ctx)
            gt = "DECISION" if e["topic"] != "noise" else "NOISE"
            results.append((gt, res.write))
        dt = time.perf_counter() - t0

        sig = [r for r in results if r[0] == "DECISION"]
        noi = [r for r in results if r[0] == "NOISE"]
        sig_recall = sum(1 for r in sig if r[1]) / len(sig) if sig else 0.0
        fpr = sum(1 for r in noi if r[1]) / len(noi) if noi else 0.0
        avg_lat = dt / len(events) * 1000 if events else 0

        print(
            f"{scen_name}\n"
            f"  n={len(events)}  gate_skips={gate_skips}  "
            f"arbiter_runs={len(events) - gate_skips}\n"
            f"  recall={sig_recall:.1%}  FPR={fpr:.1%}\n"
            f"  avg_latency={avg_lat:.0f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
