"""Re-label the DECISION pile with a stricter oracle prompt.

Quality audit of the first bootstrap pass (2026-04-17) revealed that
Gemini's default label prompt was too permissive: transition sentences
like "Let me check X" or "Now update Y" were frequently classified as
DECISION because a literal reading of "design choice" treats any
intentional action as a decision. The NOISE pile was clean (15/15
correct on a sample); the DECISION pile was estimated 25% real / 75%
Gemini-liberal on a 20-row sample.

This script re-queries Gemini on every existing DECISION (and
optionally UNCERTAIN) label with an explicit filler-pattern
specification in the prompt. If the strict pass flips a label, the
existing label row in vstash is replaced. Conservative: NOISE labels
are not touched (they were correct).

Idempotency: the new label body carries ``strict_pass: true`` so
re-runs skip already-relabeled events.

Usage::

    python3 -m experiments.relabel_decision_pile --dry-run

    python3 -m experiments.relabel_decision_pile
        # processes every DECISION label across all projects

    python3 -m experiments.relabel_decision_pile \\
        --project engram --project vex --limit-per-project 50
"""

from __future__ import annotations

import torch  # noqa: F401  -- Mistake #10

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from merken import Memory
from merken.labeling import _parse_backend_response, Label


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


class StrictGeminiBackend:
    """Same transport as GeminiLabelBackend, stricter prompt."""

    def __init__(self, *, model: str = "gemini-2.0-flash") -> None:
        from google import genai

        resolved_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not resolved_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set."
            )
        self._client = genai.Client(api_key=resolved_key)
        self.name = f"{model}+strict"
        self._model = model

    def label(self, text: str) -> Label:
        prompt = STRICT_PROMPT.format(text=text[:3000])
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt
        )
        return _parse_backend_response(resp.text or "", self.name)


def iter_decision_labels(mem: Memory, top_k: int = 1000):
    """Yield label rows whose `decision` field is DECISION."""
    for row in mem.search_labels(top_k=top_k):
        title = getattr(row, "title", "") or ""
        text = getattr(row, "text", "") or ""
        if not title.startswith("label:"):
            continue
        try:
            body = json.loads(text)
        except Exception:
            continue
        if not isinstance(body, dict):
            continue
        if body.get("decision") != "DECISION":
            continue
        if body.get("strict_pass") is True:
            continue  # idempotent
        yield title, body


def find_projects() -> list[str]:
    dbs_dir = Path.home() / ".merken"
    return sorted(p.stem for p in dbs_dir.glob("*.db"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-label the DECISION pile with a stricter oracle prompt."
    )
    parser.add_argument(
        "--project",
        action="append",
        default=None,
        help="Bootstrap this project only; repeatable. Default: all.",
    )
    parser.add_argument(
        "--limit-per-project",
        type=int,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help=(
            "Gemini model id (default: gemini-2.0-flash). Chosen after "
            "bench across 9 models with strict prompt -- see "
            "experiments/oracle_model_bench.py. 95% agreement, "
            "100% DEC_recall, ~1s/call, perfectly self-consistent."
        ),
    )
    args = parser.parse_args()

    projects = args.project if args.project else find_projects()

    backend = None
    if not args.dry_run:
        backend = StrictGeminiBackend(model=args.model)
        print(f"oracle: {backend.name}")

    grand: Counter[str] = Counter()

    for project in projects:
        db_path = Path.home() / ".merken" / f"{project}.db"
        if not db_path.exists():
            continue
        print(f"\n=== {project} ===")
        try:
            mem = Memory(project=project, db=str(db_path))
        except Exception as e:
            print(f"  open FAILED: {type(e).__name__}: {e}")
            continue

        per = Counter()
        for title, body in iter_decision_labels(mem):
            per["seen"] += 1
            text = body.get("event_text_preview") or ""
            if not text.strip():
                per["skipped_empty"] += 1
                continue

            if args.dry_run:
                per["would_relabel"] += 1
                continue

            try:
                label = backend.label(text)
            except Exception as e:
                per["errors"] += 1
                print(f"  ERROR: {type(e).__name__}: {e}")
                continue

            old_kind = body.get("decision")
            new_kind = label.decision
            per[f"new_{new_kind}"] += 1
            if old_kind != new_kind:
                per["flipped"] += 1

            body.update(
                {
                    "decision": new_kind,
                    "confidence": label.confidence,
                    "rationale": label.rationale,
                    "backend": label.backend,
                    "strict_pass": True,
                    "pre_strict_decision": old_kind,
                    "pre_strict_rationale": body.get("rationale"),
                }
            )
            # Remove label: prefix to feed remember_label(event_title=...)
            event_title = title.removeprefix("label:")
            mem.remember_label(
                event_title=event_title,
                body=json.dumps(body, ensure_ascii=False, indent=2),
            )
            per["updated"] += 1

            arrow = "==" if old_kind == new_kind else "->"
            print(
                f"  {old_kind} {arrow} {new_kind} conf={label.confidence:.2f}  "
                f"{event_title[:70]}"
            )

            if (
                args.limit_per_project is not None
                and per["updated"] >= args.limit_per_project
            ):
                break

        for k, v in per.items():
            grand[k] += v
        print(f"  stats: {dict(per)}")

    print("\n=== totals ===")
    for k, v in sorted(grand.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
