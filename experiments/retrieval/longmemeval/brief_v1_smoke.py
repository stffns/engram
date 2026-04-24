"""Step 2 of notes/merken-full-pipeline-longmemeval.md:

Brief_v1 materialization smoke. Pick one seed=44 question, run
`generate_briefs` over the full haystack (AlwaysWrite substrate) with
Cerebras as synthesize_fn, inspect the briefs produced.

Gate:
  - briefs capture the multi-chunk facts that raw retrieval missed
    -> proceed to Step 3.
  - briefs vague / don't compose -> tune prompt or abandon.

Cost budget: ~$1-2. No Memory/vstash DB needed for inspection --
we bypass ingestion and call generate_briefs directly with a list
of (path, text) tuples. End-to-end consolidate() path is tested
later at Step 3.
"""

from __future__ import annotations

# ruff: noqa: E402
import torch  # noqa: F401

import json
import random
import sys
import time
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402
from merken.consolidation import generate_briefs  # noqa: E402

SEED = 44
N_SAMPLE = 30
# Index of the seed=44 question to probe. 0 = qid=099778bb (multi-session,
# leadership). See filter_recall_seed44_n5.json for the full list.
SAMPLE_INDEX = 0

CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"
MAX_COMPLETION_TOKENS = 6000
TEMPERATURE = 0.0


def _cerebras_synthesize(prompts: list[str]) -> str:
    """Match SynthesizeFn: (list[str]) -> str. Only prompts[0] is used by
    generate_briefs, but the signature accepts a list for API compatibility.
    """
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras()
    prompt = prompts[0]
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=[
            {"role": "system", "content": "You synthesize temporal briefs from event streams."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
    )
    dt = time.perf_counter() - t0
    u = resp.usage
    pt = getattr(u, "prompt_tokens", 0)
    ct = getattr(u, "completion_tokens", 0)
    text = resp.choices[0].message.content or ""
    print(
        f"  [cerebras] dt={dt:.1f}s prompt_tok={pt} completion_tok={ct} "
        f"output_chars={len(text)}",
        flush=True,
    )
    return text


def main() -> int:
    print(f"[load] longmemeval_s seed={SEED} N={N_SAMPLE}", flush=True)
    convs = load_longmemeval("longmemeval_s")
    rnd = random.Random(SEED)
    sampled = rnd.sample(convs, N_SAMPLE)
    conv = sampled[SAMPLE_INDEX]

    print(
        f"[probe] qid={conv.question_id} type={conv.question_type} "
        f"n_sessions={conv.n_sessions} n_turns={conv.n_turns}",
        flush=True,
    )
    print(f"[probe] question: {conv.question}", flush=True)
    print(f"[probe] answer: {conv.answer}", flush=True)
    print(f"[probe] answer_session_ids: {conv.answer_session_ids}", flush=True)

    # Build events list: one event per turn. path is synthetic sid:idx.
    events: list[tuple[str, str]] = []
    for sid, turns in conv.haystack_sessions.items():
        for i, t in enumerate(turns):
            events.append((f"{sid}:{i}", f"[{t.role}] {t.content}"))

    total_chars = sum(len(text) for _, text in events)
    est_prompt_tokens = total_chars // 4
    print(
        f"[prompt] n_events={len(events)} total_chars={total_chars} "
        f"est_prompt_tokens={est_prompt_tokens}",
        flush=True,
    )
    if est_prompt_tokens > 110_000:
        print(
            "[warn] prompt exceeds ~110k tokens; Qwen-235b may truncate. "
            "Consider splitting by session if briefs look incomplete.",
            flush=True,
        )

    print(f"[call] generate_briefs via {CEREBRAS_MODEL} temp={TEMPERATURE}", flush=True)
    t0 = time.perf_counter()
    briefs = generate_briefs(events, _cerebras_synthesize, today="2026-04-24")
    dt = time.perf_counter() - t0
    print(f"[done] n_briefs={len(briefs)} total_wall={dt:.1f}s", flush=True)

    # Aggregation-fact probe: for the leadership-percentage question,
    # look for "20" / "leadership" / "women" mentions in the briefs.
    needle_words = ["women", "leadership", "20", "diversity"]
    needle_brief_idx: list[int] = []
    print("", flush=True)
    print("=== briefs (head + needle probe) ===", flush=True)
    for bi, brief in enumerate(briefs):
        head = brief.split("\n")[0][:140]
        hits = [w for w in needle_words if w.lower() in brief.lower()]
        mark = f"  HITS={hits}" if hits else ""
        print(f"[{bi}] {head}{mark}", flush=True)
        if hits:
            needle_brief_idx.append(bi)

    if needle_brief_idx:
        print("", flush=True)
        print("=== briefs with needle hits (full text) ===", flush=True)
        for bi in needle_brief_idx:
            print(f"\n--- brief [{bi}] ---\n{briefs[bi]}", flush=True)
    else:
        print("\n[probe] NO brief contained any needle word.", flush=True)
        print("        The aggregation fact likely was NOT materialized.", flush=True)

    # Save
    out = {
        "qid": conv.question_id,
        "question_type": conv.question_type,
        "question": conv.question,
        "answer": conv.answer,
        "answer_session_ids": conv.answer_session_ids,
        "seed": SEED,
        "sample_index": SAMPLE_INDEX,
        "cerebras_model": CEREBRAS_MODEL,
        "temperature": TEMPERATURE,
        "n_events": len(events),
        "est_prompt_tokens": est_prompt_tokens,
        "wall_seconds": dt,
        "n_briefs": len(briefs),
        "briefs": briefs,
        "needle_words": needle_words,
        "needle_brief_indices": needle_brief_idx,
    }
    out_path = Path(__file__).parent / "brief_v1_seed44_q0_smoke.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
