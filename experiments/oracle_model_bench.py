"""Compare Gemini models on a controlled labeling test.

Hand-curated 20-event test set drawn from the current DECISION pile in
``data/merken_labels_v7.jsonl``. Each item has an expected label
reasoned from the strict prompt definition (see relabel_decision_pile).

We run N candidate models against this set and measure:

- Agreement with expected label
- Precision/recall per class
- Latency

This is not a full-scale eval -- 20 items is small. Its purpose is to
pick the best-per-dollar oracle before burning ~750 Gemini calls on
the real pile.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from google import genai

from merken.labeling import _parse_backend_response


STRICT_PROMPT = (
    "You are labeling memory events for an agent's write-filter training set.\n"
    "The filter decides which assistant messages to keep in long-term memory.\n\n"
    "Read the event below and classify it as exactly one of:\n\n"
    "  DECISION  -- a lasting commitment, conclusion, analysis result,\n"
    "               fact discovery, reference material, or explained\n"
    "               reasoning worth keeping. Must contain CONCRETE\n"
    "               content: numbers, file paths, specific findings,\n"
    "               design rationale, or resolution of an open\n"
    "               question.\n\n"
    "  NOISE     -- ephemeral/transient, including:\n"
    "               * transition sentences like 'Let me X', 'Now let me\n"
    "                 Y', 'Let me check Z', 'Now I'll do W' (regardless\n"
    "                 of what X/Y/Z/W refer to).\n"
    "               * task announcements without content: 'Now fix the\n"
    "                 tests', 'Now update Y', 'Let me also add Z'.\n"
    "               * status exclamations: 'Good', 'Perfect!', 'Great!',\n"
    "                 'Excellent!' followed by a next-step.\n"
    "               * intermediate steps that only make sense in the\n"
    "                 context of a longer chain of actions.\n\n"
    "  UNCERTAIN -- genuinely ambiguous; reasonable people disagree.\n\n"
    "Important: the mere fact that the agent is ABOUT TO take an action\n"
    "(e.g. 'Let me check the config file next') is NOT a decision. A\n"
    "decision requires the RESULT of the action, an analysis, a\n"
    "commitment, or explained rationale.\n\n"
    "Examples of NOISE (re-classify as NOISE even if they look\n"
    "purposeful):\n"
    "  - 'Let me check the server configuration next.'\n"
    "  - 'Now I need to update the imports.'\n"
    "  - 'Good. Let me look at the test file.'\n"
    "  - 'Perfect! Now let me continue with the next step.'\n"
    "  - 'Now update Documentation table to include new docs.'\n\n"
    "Examples of DECISION:\n"
    "  - 'The server timed out after 30s because the connection pool\n"
    "     was exhausted. Switching to pgbouncer in transaction mode\n"
    "     fixed it.'\n"
    "  - 'Benchmark: Cython parallel runs in 42ms vs NumPy 340ms (8x).'\n"
    "  - 'Reverted commit abc123: it broke the auth flow for SSO\n"
    "     users.'\n"
    "  - '174 tests passed. Query LRU cache wired via new CacheConfig\n"
    "     model in vstash/config.py with query_cache_size default 0.'\n\n"
    "Event:\n"
    "<<<\n"
    "{text}\n"
    ">>>\n\n"
    "Respond on a single line in this exact format (no prose, no code\n"
    "fences):\n"
    "LABEL: <DECISION|NOISE|UNCERTAIN> | CONFIDENCE: <number 0.0-1.0> "
    "| REASON: <one sentence>"
)


# Hand-curated test set: 10 NOISE, 5 DECISION, 5 borderline.
# Each item: (expected_label, text).
TEST_SET: list[tuple[str, str]] = [
    # --- clear NOISE: filler/transition/task-announcement ---
    ("NOISE", "Now let me check the ONEAPP simulation file and search for any other API test configurations:"),
    ("NOISE", "Let me use Read to determine the search method length:"),
    ("NOISE", "Let me check what's blocking real execution across all layers:"),
    ("NOISE", "Now add the history recall and recording methods. Let me find a good place:"),
    ("NOISE", "Now let me look at the reindex function to understand the vector backend sync:"),
    ("NOISE", "Perfect! Let me also check the core Config file to understand how the simulation loads configuration:"),
    ("NOISE", "Good. Let me check the test file to understand how it's structured."),
    ("NOISE", "Let me continue reading the ChatView to find the message sending logic:"),
    ("NOISE", "Now update Documentation table to include new docs:"),
    ("NOISE", "Now let me check the generate_report method signature to understand the complete flow:"),
    # --- clear DECISION: concrete findings/results/rationale ---
    ("DECISION", "174 passed, 0 failed. Here's the summary: Query LRU cache - done. Files changed: `vstash/config.py` -- new CacheConfig model with query_cache_size: int = 0; `vstash/store.py` -- LRU wrapper around search()."),
    ("DECISION", "The LocoMo benchmark showed v6 drops 7pp vs baseline. Root cause: 19 sessions between same 2 speakers = uniformly high embedding similarity. Complete linkage creates 1 mega-fact from all sessions."),
    ("DECISION", "Benchmark: Cython parallel ADC kernel runs in 42ms for M=192 K=256, vs NumPy 340ms. 8x speedup confirmed with 151 tests passing."),
    ("DECISION", "Reverted commit abc123: it broke the auth flow for SSO users because the token refresh logic did not handle the new claim format. Rolled back in develop, pinned to previous commit for hotfix."),
    ("DECISION", "Decision: switch default write_decider from HeuristicWriteDecider to ContentTypePriorDecider. Evidence: 14pp retrieval improvement on knowledge_update_50topics (42% -> 56%) with 86% store reduction. Published in v0.2.0."),
    # --- borderline: debatable, will help see how models handle grey zone ---
    ("NOISE", "Wait -- global shows `total: 3596, ok: 3596, ko: 0`. But earlier I was seeing UID not found errors. Looks like Gatling counted compile errors as un-executed instead of KO."),
    ("DECISION", "It merged to `develop` instead of `main`. Let me create a new PR targeting main:"),  # actually borderline; the fact that it merged to wrong branch is content
    ("NOISE", "Let me verify the file is syntactically valid and the imports resolve."),
    ("DECISION", "The file already exists from the mkdir -- the mkdir script created directories matching existing file names. Refactor needed to check existence before creating."),
    ("NOISE", "Commit hecho: `2469fe7`. Ahora el merge a develop:"),
]


def run_model(client, model: str, items) -> tuple[list[str], float]:
    """Return (predictions, total_seconds)."""
    preds: list[str] = []
    t0 = time.time()
    for _, text in items:
        # .replace (not .format) so `{` / `}` in raw text
        # don't trigger KeyError.
        prompt = STRICT_PROMPT.replace("{text}", text[:3000])
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt
            )
            label = _parse_backend_response(resp.text or "", model)
            preds.append(label.decision)
        except Exception as e:
            preds.append(f"ERROR:{type(e).__name__}")
    return preds, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model id to test; repeatable. Default: a curated set.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat each model N times to measure consistency.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    models = args.model or [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview",
    ]

    expected = [lab for lab, _ in TEST_SET]
    noise_idx = [i for i, lab in enumerate(expected) if lab == "NOISE"]
    decision_idx = [i for i, lab in enumerate(expected) if lab == "DECISION"]

    print(f"test set: {len(TEST_SET)} items  (expected: NOISE={len(noise_idx)}, DECISION={len(decision_idx)})")
    print()

    for model in models:
        print(f"--- {model} ---")
        all_runs: list[list[str]] = []
        for run in range(args.runs):
            preds, elapsed = run_model(client, model, TEST_SET)
            all_runs.append(preds)
            agree = sum(1 for e, p in zip(expected, preds) if e == p)
            noise_recall = sum(1 for i in noise_idx if preds[i] == "NOISE") / max(len(noise_idx), 1)
            dec_recall = sum(1 for i in decision_idx if preds[i] == "DECISION") / max(len(decision_idx), 1)
            errs = sum(1 for p in preds if p.startswith("ERROR"))
            print(
                f"  run {run+1}: agreement {agree}/{len(expected)} "
                f"({agree/len(expected)*100:.0f}%) "
                f"NOISE_recall={noise_recall:.0%} "
                f"DEC_recall={dec_recall:.0%} "
                f"errs={errs} t={elapsed:.1f}s"
            )

        if args.runs > 1:
            # self-consistency: how often did runs agree per item
            consistent = sum(
                1 for i in range(len(expected))
                if len({r[i] for r in all_runs}) == 1
            )
            print(
                f"  self-consistency: {consistent}/{len(expected)} items "
                f"identical across {args.runs} runs"
            )
        # Show disagreements with expected
        first = all_runs[0]
        disagreements = [
            (i, expected[i], first[i], TEST_SET[i][1][:90])
            for i in range(len(expected))
            if first[i] != expected[i]
        ]
        if disagreements:
            print("  disagreements (run 1):")
            for i, exp, got, txt in disagreements:
                print(f"    [{i:2d}] expected={exp} got={got}  {txt}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
