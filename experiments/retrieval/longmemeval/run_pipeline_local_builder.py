"""Run pipeline_runner with a local MLX Builder (no Cerebras call).

Monkey-patches `cerebras_chat` to route to an MLX-hosted model when
the model name starts with the sentinel "mlx:". The rest of the
pipeline (brief synth, retrieval, oracle) is unchanged; only the
Builder step executes locally.

Usage:
    python3 experiments/retrieval/longmemeval/run_pipeline_local_builder.py \\
        --mlx-path mlx-community/Ministral-3-8B-Instruct-2512-4bit \\
        --top-k-episodic 10 \\
        --skip-briefs \\
        -- --seed 44 --n 30 --tag zeroshot_min8b

All args after `--` are forwarded to pipeline_runner.py.

Design notes:
- Reuses the same wrapper monkey-patches as run_pipeline_higher_k
  (top-k, skip-briefs, no-op briefs) so ablations compose.
- The MLX model is loaded once at startup; subsequent Builder calls
  reuse the same model / tokenizer (no per-call load overhead).
- cerebras_chat's signature is (model, messages, max_tokens,
  temperature) -> (text, wall_s, usage). The MLX path returns the
  same shape with usage={} (mlx_lm does not report token counts the
  same way Cerebras does; the key fields prompt_tokens /
  completion_tokens are set to rough estimates from the tokenizer).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))


_MLX_SENTINEL = "mlx:"


# Trust-first Builder system prompt. Encodes Jay's principle
# (2026-04-24): a memory system may doubt when it lacks data but
# must not state false facts as if true. Designed to reduce the
# `supports -> contradicts` failure class seen in Ministral 3B
# zero-shot (and worse in 8B), where confidently-wrong answers
# dominate the losses. Preferring `not found in memory` over a
# guess aligns with the four-way oracle verdict under which
# `neutral` (hedged-honestly) is a strictly better outcome than
# `contradicts` (lied).
TRUST_PIPELINE_BUILDER_SYSTEM = (
    "You are answering as a memory system that must never state "
    "false facts as if true. Use ONLY the provided context.\n\n"
    "Context shape:\n"
    "- BRIEFS (if present): LLM-generated topic summaries. Format: "
    "`## <topic>` header, `**As of:**` date, bullets.\n"
    "- EXCERPTS: raw conversational turns.\n\n"
    "Rules (in priority order):\n"
    "1. If the context EXPLICITLY states the answer, provide it "
    "and quote specific values (numbers, names, dates, titles) "
    "verbatim from the context.\n"
    "2. If the context requires combining facts present across "
    "briefs/excerpts (e.g. ordering by date, computing a total "
    "from enumerated items), do the combination and state the "
    "result with a brief justification.\n"
    "3. If the context does NOT contain the specific fact the "
    "question asks for, or if answering would require guessing, "
    "inferring from absent evidence, or stretching an analogy, "
    "respond EXACTLY: 'not enough information in memory'. Do not "
    "substitute a related or adjacent fact.\n"
    "4. Prefer admitting insufficient information over stating a "
    "fact you cannot directly support from the context.\n"
    "5. Keep answers under 150 words. Respond in plain prose. Do "
    "NOT reproduce the brief `## topic` / `**As of:**` format."
)


def _make_mlx_chat(mlx_path: str):
    """Load an MLX model + tokenizer once and return a cerebras_chat-
    compatible function that ignores the model name parameter and
    always uses the preloaded MLX model.
    """
    from mlx_lm import load, generate

    print(f"[local-builder] loading MLX model: {mlx_path}", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(mlx_path)
    print(f"[local-builder] loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    def mlx_chat(model_name, messages, max_tokens, *, temperature=0.3):
        t0 = time.perf_counter()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        # Rough token counts via the tokenizer; mlx_lm's generate does
        # not surface the precise counts Cerebras returns.
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(text or ""))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return (text or "").strip(), dt, usage

    return mlx_chat


def main() -> int:
    if "--" not in sys.argv:
        print(
            "usage: run_pipeline_local_builder.py "
            "--mlx-path <hf-or-local-path> [--top-k-episodic N] "
            "[--top-k-briefs M] [--skip-briefs] "
            "-- <pipeline_runner args>",
            file=sys.stderr,
        )
        return 2
    split = sys.argv.index("--")
    wrapper_argv = sys.argv[1:split]
    forwarded = sys.argv[split + 1:]

    mlx_path = None
    top_k_epi = None
    top_k_br = None
    skip_briefs = False
    trust_prompt = False
    i = 0
    while i < len(wrapper_argv):
        a = wrapper_argv[i]
        if a == "--mlx-path":
            mlx_path = wrapper_argv[i + 1]; i += 2; continue
        if a == "--top-k-episodic":
            top_k_epi = int(wrapper_argv[i + 1]); i += 2; continue
        if a == "--top-k-briefs":
            top_k_br = int(wrapper_argv[i + 1]); i += 2; continue
        if a == "--skip-briefs":
            skip_briefs = True; i += 1; continue
        if a == "--trust-prompt":
            trust_prompt = True; i += 1; continue
        i += 1

    if mlx_path is None:
        print("--mlx-path <path> is required", file=sys.stderr)
        return 2

    from experiments.retrieval.longmemeval import pipeline_runner as pr
    import experiments.midloop_concept.medlocal.cerebras_midloop as _cm

    if top_k_epi is not None:
        print(f"[wrapper] RAG_TOP_K_EPISODIC: {pr.RAG_TOP_K_EPISODIC} -> {top_k_epi}",
              flush=True)
        pr.RAG_TOP_K_EPISODIC = top_k_epi
    if top_k_br is not None:
        print(f"[wrapper] RAG_TOP_K_BRIEFS: {pr.RAG_TOP_K_BRIEFS} -> {top_k_br}",
              flush=True)
        pr.RAG_TOP_K_BRIEFS = top_k_br

    if skip_briefs:
        print("[wrapper] skip_briefs=True: no brief synth + retrieval skipped",
              flush=True)
        def _noop(conv, mem, collection, today):
            return 0, 0, []
        def _retrieve_no_briefs(mem, question, collection):
            episodic = pr._dual_episodic(mem, question, collection, pr.RAG_TOP_K_EPISODIC)
            return [], episodic, 0
        pr._per_session_briefs = _noop
        pr._retrieve = _retrieve_no_briefs

    # Replace cerebras_chat -- this is what the Builder call hits via
    # pipeline_runner._call_builder -> cerebras_chat. The brief synth
    # in pipeline_runner._cerebras_brief_synth ALSO calls a Cerebras
    # client directly (different code path that uses the SDK), so if
    # briefs are enabled they still hit Cerebras; to avoid that use
    # --skip-briefs.
    mlx_chat = _make_mlx_chat(mlx_path)
    _cm.cerebras_chat = mlx_chat
    pr.cerebras_chat = mlx_chat

    sentinel_name = f"{_MLX_SENTINEL}{mlx_path}"
    _cm.BUILDER = sentinel_name
    pr.BUILDER = sentinel_name
    print(f"[wrapper] BUILDER routed to MLX: {sentinel_name}", flush=True)

    if trust_prompt:
        print("[wrapper] trust_prompt=True: swapping PIPELINE_BUILDER_SYSTEM "
              "with trust-first variant (prefer 'not enough information' "
              "over confidently-wrong answers).", flush=True)
        pr.PIPELINE_BUILDER_SYSTEM = TRUST_PIPELINE_BUILDER_SYSTEM
        pr.BUILDER_SYSTEM = TRUST_PIPELINE_BUILDER_SYSTEM

    sys.argv = [sys.argv[0]] + forwarded
    return pr.main()


if __name__ == "__main__":
    raise SystemExit(main())
