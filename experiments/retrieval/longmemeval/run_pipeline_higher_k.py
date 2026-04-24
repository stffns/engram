"""Wrapper that runs pipeline_runner.main() with monkey-patched
RAG_TOP_K_EPISODIC / RAG_TOP_K_BRIEFS so we can test larger k on
specific qids without editing the shared runner.

Usage:
    python3 experiments/retrieval/longmemeval/run_pipeline_higher_k.py \\
        --top-k-episodic 10 --top-k-briefs 5 \\
        -- --seed 44 --n 30 --tag higher_k_smoke \\
        --only-qids qid1,qid2,qid3

Args after `--` are forwarded verbatim to pipeline_runner.main().
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))


def main() -> int:
    if "--" not in sys.argv:
        print("usage: run_pipeline_higher_k.py --top-k-episodic N [--top-k-briefs M] -- <pipeline_runner args>",
              file=sys.stderr)
        return 2
    split = sys.argv.index("--")
    wrapper_argv = sys.argv[1:split]
    forwarded = sys.argv[split + 1:]

    top_k_epi = None
    top_k_br = None
    builder_model = None
    skip_briefs = False
    i = 0
    while i < len(wrapper_argv):
        a = wrapper_argv[i]
        if a == "--top-k-episodic":
            top_k_epi = int(wrapper_argv[i + 1]); i += 2; continue
        if a == "--top-k-briefs":
            top_k_br = int(wrapper_argv[i + 1]); i += 2; continue
        if a == "--builder-model":
            builder_model = wrapper_argv[i + 1]; i += 2; continue
        if a == "--skip-briefs":
            skip_briefs = True; i += 1; continue
        i += 1
    if (top_k_epi is None and top_k_br is None and builder_model is None
            and not skip_briefs):
        print("must pass at least one of --top-k-episodic / --top-k-briefs / --builder-model / --skip-briefs", file=sys.stderr)
        return 2

    from experiments.retrieval.longmemeval import pipeline_runner as pr
    if top_k_epi is not None:
        print(f"[wrapper] RAG_TOP_K_EPISODIC: {pr.RAG_TOP_K_EPISODIC} -> {top_k_epi}", flush=True)
        pr.RAG_TOP_K_EPISODIC = top_k_epi
    if top_k_br is not None:
        print(f"[wrapper] RAG_TOP_K_BRIEFS: {pr.RAG_TOP_K_BRIEFS} -> {top_k_br}", flush=True)
        pr.RAG_TOP_K_BRIEFS = top_k_br
    if builder_model is not None:
        # pipeline_runner imports BUILDER at top-level from cerebras_midloop,
        # so patching both locations keeps any call sites consistent.
        print(f"[wrapper] BUILDER: {pr.BUILDER} -> {builder_model}", flush=True)
        pr.BUILDER = builder_model
        import experiments.midloop_concept.medlocal.cerebras_midloop as _cm
        _cm.BUILDER = builder_model

    if skip_briefs:
        print("[wrapper] skip_briefs=True: _per_session_briefs becomes a no-op "
              "AND _retrieve skips the brief_pool search entirely "
              "(Builder sees only episodic excerpts).",
              flush=True)

        def _noop_per_session_briefs(conv, mem, collection, today):
            # Return the shape the original returns: (n_calls, n_briefs, meta)
            return 0, 0, []

        _orig_retrieve = pr._retrieve

        def _retrieve_no_briefs(mem, question, collection):
            # vstash does not accept top_k=0, so we bypass the brief-pool
            # call entirely and just reuse the original episodic retrieval.
            episodic = pr._dual_episodic(mem, question, collection, pr.RAG_TOP_K_EPISODIC)
            return [], episodic, 0

        pr._per_session_briefs = _noop_per_session_briefs
        pr._retrieve = _retrieve_no_briefs

    sys.argv = [sys.argv[0]] + forwarded
    return pr.main()


if __name__ == "__main__":
    raise SystemExit(main())
