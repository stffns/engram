"""Generate synthetic markdown-NOISE training samples via Gemini.

The 2026-04-17 `markdown_tables_held_out` scenario showed that both
char v2 and BPE v4 classify EVERY table-formatted event as DECISION
(100% FPR on 6 noise tables). Root cause: the NOISE side of training
data is 100% single-paragraph flat text ("Sprint planning:
..."), so the model has no reference for table-formatted NOISE.

This script asks Gemini to generate routine-status markdown tables
across six categories (schedule / status_snapshot / toc / pricing /
log / roster) so v6 training has balanced structured NOISE. The
content deliberately uses domains absent from the 6 held-out
examples (those domains: aviation, game dev, biolab, satellite,
firmware, retail waves) -- we synthesize in adjacent-but-distinct
domains so the held-out scenario remains an honest generalization
test.

Usage:

    export GEMINI_API_KEY=...
    PYTHONPATH=. python -m experiments.consolidation.generate_markdown_noise \\
        --out data/markdown_noise_v6.json --per-category 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

CATEGORIES: dict[str, str] = {
    "schedule": (
        "A routine meeting/oncall/event schedule formatted as a markdown "
        "table. No decisions, just upcoming slots or attendance. "
        "Examples of the TYPE (do NOT reuse): daily standup agenda, "
        "oncall rotation week, retro attendance, design review roster."
    ),
    "status_snapshot": (
        "A routine status snapshot formatted as a markdown table. "
        "Numeric metrics or pass/fail columns, nothing decided. "
        "Examples of the TYPE: sprint burndown, service health, "
        "ticket triage summary, capacity report."
    ),
    "toc": (
        "A table of contents or index as a markdown table, with no "
        "substantive content besides section names and page numbers."
    ),
    "pricing": (
        "A declarative pricing / tier / configuration reference as a "
        "markdown table. Just values, no commitment to act."
    ),
    "log": (
        "A structured log digest as a markdown table (one row per event, "
        "columns for severity / source / message). Summary-of-noise, "
        "not an incident postmortem."
    ),
    "roster": (
        "A team/committee roster as a markdown table: name, role, "
        "contact. No decisions, just a directory."
    ),
}

# Mini-universes so Gemini doesn't keep producing 'engineering team'
# examples. We sample a fresh universe per prompt to spread coverage.
UNIVERSES = [
    "maritime logistics company",
    "public library system",
    "urban planning office",
    "renewable energy co-op",
    "university research lab",
    "veterinary clinic chain",
    "music festival production crew",
    "municipal water utility",
    "arts foundation grants office",
    "neighbourhood clinic network",
    "regional airline cargo ops",
    "archival digitization project",
]


_PROMPT_TEMPLATE = (
    "Generate ONE markdown document that is an example of routine NOISE "
    "(not a decision worth keeping in long-term memory). It MUST satisfy "
    "all of:\n"
    "\n"
    "- Category: {category_desc}\n"
    "- Setting: {universe}.\n"
    "- Contains at least one GitHub-flavored markdown table with 3-5 "
    "columns and 3-6 rows. A short header line is fine.\n"
    "- 400-1200 characters total.\n"
    "- Feels routine or declarative, never a decision being made. No "
    "language like 'we decided', 'migrated', 'adopted', 'chose'.\n"
    "- Do NOT repeat any of the excluded tech stacks below:\n"
    "{excluded}\n"
    "\n"
    "Return ONLY the markdown body. No code fences, no preamble."
)

# Domains used by the held-out scenario -- we MUST avoid generating
# similar content or the val set leaks into train.
EXCLUDED_STACKS = [
    "flight planner event bus (Kafka / NATS)",
    "game engine selection (Godot / Bevy)",
    "cytometry panel cutoffs",
    "retail store inventory-V2 rollout waves",
    "satellite ground station slot allocation (AsterSat, Svalbard, Kiruna)",
    "firmware 4.8 feature freeze",
]

EXCLUDED_TEXT = "\n".join(f"  - {s}" for s in EXCLUDED_STACKS)


def _generate_one(client, model: str, category: str, universe: str) -> str | None:
    prompt = _PROMPT_TEMPLATE.format(
        category_desc=CATEGORIES[category],
        universe=universe,
        excluded=EXCLUDED_TEXT,
    )
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = (resp.text or "").strip()
    except Exception as exc:
        print(f"  ! {category}/{universe}: {exc.__class__.__name__}: {exc}")
        return None

    # Strip accidental code fences.
    text = re.sub(r"^```(?:markdown)?\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)

    if len(text) < 200 or len(text) > 2000:
        print(f"  ! {category}/{universe}: length out of bounds ({len(text)} chars)")
        return None
    if "|" not in text or text.count("|") < 6:
        print(f"  ! {category}/{universe}: does not look like a table")
        return None
    return text.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSON output path (a list of {id, text, topic, category} rows)",
    )
    p.add_argument("--per-category", type=int, default=5)
    p.add_argument("--model", default="gemini-2.0-flash")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=key)

    random.seed(args.seed)
    rows = []
    for category in CATEGORIES:
        chosen = random.sample(UNIVERSES, args.per_category)
        for universe in chosen:
            print(f"- generating {category}/{universe} ...", flush=True)
            text = _generate_one(client, args.model, category, universe)
            if text is None:
                continue
            rows.append({
                "id": f"md_noise_{category}_{len(rows):02d}",
                "topic": "noise",
                "category": category,
                "universe": universe,
                "text": text,
            })
            time.sleep(0.2)  # gentle rate-limit guard

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(
        f"\nSaved {len(rows)} samples to {args.out} "
        f"(categories: {sorted({r['category'] for r in rows})})"
    )


if __name__ == "__main__":
    main()
