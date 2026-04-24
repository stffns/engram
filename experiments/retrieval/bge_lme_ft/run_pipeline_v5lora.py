"""Gate 3: run an existing LongMemEval runner with the v5-lora encoder.

Monkey-patches vstash.embed + vstash.Memory to route embeddings
through the LoRA-adapted encoder, then calls the requested runner's
main(). Zero changes to the runner source.

Usage:
    python3 experiments/retrieval/bge_lme_ft/run_pipeline_v5lora.py \\
        --adapter experiments/retrieval/bge_lme_ft/adapters/v5-lora-lr1e4 \\
        --runner pipeline \\
        -- --seed 44 --n 30 --tag v5lora

Runners:
  pipeline  -> experiments.retrieval.longmemeval.pipeline_runner (default)
  mode_a    -> experiments.retrieval.longmemeval.mode_a_eval
  mode_c    -> experiments.retrieval.longmemeval.mode_c_benchmark

All args AFTER the `--` are forwarded verbatim to the runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))


def main() -> int:
    if "--" not in sys.argv:
        print("usage: run_pipeline_v5lora.py --adapter <path> -- <pipeline_runner args>",
              file=sys.stderr)
        return 2
    split = sys.argv.index("--")
    wrapper_argv = sys.argv[1:split]
    forwarded = sys.argv[split + 1:]

    adapter = None
    runner_key = "pipeline"
    i = 0
    while i < len(wrapper_argv):
        a = wrapper_argv[i]
        if a == "--adapter":
            if i + 1 >= len(wrapper_argv):
                print("--adapter needs a path argument", file=sys.stderr)
                return 2
            adapter = wrapper_argv[i + 1]
            i += 2
            continue
        if a == "--runner":
            if i + 1 >= len(wrapper_argv):
                print("--runner needs a value", file=sys.stderr)
                return 2
            runner_key = wrapper_argv[i + 1]
            i += 2
            continue
        i += 1
    if adapter is None:
        print("missing --adapter <path>", file=sys.stderr)
        return 2

    sentinel = f"v5-lora:{adapter}"
    print(f"[shim] routing all vstash embeddings through {sentinel}", flush=True)
    print(f"[shim] runner={runner_key}", flush=True)

    from experiments.retrieval.bge_lme_ft.vstash_v5lora_shim import install
    install(auto_set_model=sentinel)

    if runner_key == "pipeline":
        from experiments.retrieval.longmemeval import pipeline_runner as runner_mod
    elif runner_key == "mode_a":
        from experiments.retrieval.longmemeval import mode_a_eval as runner_mod
    elif runner_key == "mode_c":
        from experiments.retrieval.longmemeval import mode_c_benchmark as runner_mod
    else:
        print(f"unknown --runner {runner_key!r}; use 'pipeline', 'mode_a', or 'mode_c'",
              file=sys.stderr)
        return 2

    # Forward args by rewriting sys.argv for the inner parser.
    sys.argv = [sys.argv[0]] + forwarded
    return runner_mod.main()


if __name__ == "__main__":
    raise SystemExit(main())
