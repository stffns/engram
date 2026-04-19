# ruff: noqa: I001, E402
"""Generate a disjoint-topic noise-heavy held-out scenario via Gemini.

H_v9_pragmatic (HYPOTHESES.md) confirmed that knowledge_update_50t
is 100% v7 training data. Dropping it from training loses real noise
signal. The CORRECT fix is to keep KU in training but build a NEW
held-out noise-heavy scenario whose topics are DISJOINT from
knowledge_update_*.

Existing KU topics (to avoid):
    cache, database / db, auth, deploy, cache_decision,
    db_decision, auth_decision, deploy_decision, auth_tokens,
    search_index, scaling, ...

Disjoint domains this script asks Gemini to generate:
    observability (metrics, dashboards, alerting)
    mobile_ops (app stores, permissions, push notifications)
    security_response (CVEs, patches, access audits)
    ml_ops (training runs, drift, eval pipelines)
    product_rollouts (feature flags, A/B tests, kill switches)
    billing (invoicing, vendor contracts, usage-based pricing)
    accessibility (WCAG, screen readers, keyboard nav)
    i18n (localization, RTL, character encoding)

Per domain we ask for ~6 DEC + ~6 NOI = ~12 events. With 8 domains
that's ~96 events total -- decent n for a held-out noise-heavy
scenario.

Output: experiments/loop_quality/scenarios/disjoint_noise_heavy_holdout.json
in the same format as existing scenarios (list of {id, text, topic}).

Cost: ~16 Gemini calls total (one per (domain, class) pair) at
gemini-2.0-flash rates = pennies.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from google import genai


REPO = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO / "experiments" / "loop_quality" / "scenarios"


# Topics in this dict are disjoint from knowledge_update_*.json train
# scenarios. Each topic gets a DEC prompt and a NOI prompt.
DOMAINS: dict[str, dict[str, str]] = {
    "observability": {
        "dec_prompt": (
            "You are generating HELD-OUT EVALUATION data for a memory filter. Produce "
            "6 short text events (each 80-400 chars) that represent REAL "
            "engineering DECISIONS in the observability / metrics / alerting "
            "domain. Each event should be a lasting commitment, a concrete "
            "finding, or a config change with specific numbers / file paths. "
            "Do NOT use the words 'cache', 'database', 'auth', or 'deploy' "
            "anywhere. Example: 'Alert threshold for p99 API latency raised "
            "from 200ms to 350ms in prometheus/rules.yml after the holiday "
            "traffic regression analysis.'"
        ),
        "noi_prompt": (
            "You are generating HELD-OUT EVALUATION data for a memory filter. Produce "
            "6 short text events (each 80-300 chars) of observability / "
            "metrics NOISE -- transitional sentences, routine checks, task "
            "announcements without content. Do NOT use 'cache', 'database', "
            "'auth', 'deploy'. Example: 'Now let me check the Grafana "
            "dashboard for anomalies.'"
        ),
    },
    "mobile_ops": {
        "dec_prompt": (
            "Produce 6 short text events (80-400 chars) that are REAL "
            "mobile-ops DECISIONS: app store rollouts, permission changes, "
            "push-notification config, OS compatibility drops. Specific "
            "numbers / versions / filenames. Do NOT use 'cache', 'database', "
            "'auth', 'deploy'. Example: 'Dropped iOS 14 support in the 3.8 "
            "release; the install-base share fell to 0.6% per the App "
            "Store Connect report.'"
        ),
        "noi_prompt": (
            "Produce 6 short mobile-ops NOISE events (80-300 chars) -- "
            "transitional phrases, status checks, next-step announcements. "
            "Do NOT use 'cache', 'database', 'auth', 'deploy'. Example: "
            "'Let me open App Store Connect to check the review status.'"
        ),
    },
    "security_response": {
        "dec_prompt": (
            "Produce 6 REAL security-response DECISIONS (80-400 chars): CVE "
            "triage verdicts, patch timelines, access-audit findings, key "
            "rotations. Specific CVE IDs / file paths / dates. Avoid 'cache', "
            "'database', 'auth', 'deploy'. Example: 'CVE-2025-37891 (libxml2) "
            "patched in-place on prod fleet 2026-03-14 via the weekly "
            "package-refresh cron; no code changes required.'"
        ),
        "noi_prompt": (
            "Produce 6 security-response NOISE events (80-300 chars) -- "
            "procedural, no content. Avoid 'cache', 'database', 'auth', "
            "'deploy'. Example: 'Now I'll check the vulnerability scanner "
            "output for new findings.'"
        ),
    },
    "ml_ops": {
        "dec_prompt": (
            "Produce 6 REAL ML-ops DECISIONS (80-400 chars): training-run "
            "outcomes, data-drift findings, eval-pipeline config. Specific "
            "metrics / model names / loss numbers. Avoid 'cache', "
            "'database', 'auth', 'deploy'. Example: 'training run "
            "model-v42 converged at val_loss=0.189 after 12k steps on "
            "A100x4; shipping this weights file to production inference.'"
        ),
        "noi_prompt": (
            "Produce 6 ML-ops NOISE events (80-300 chars). Procedural / "
            "filler. Avoid 'cache', 'database', 'auth', 'deploy'. Example: "
            "'Let me check the training logs in wandb to see how things "
            "look.'"
        ),
    },
    "product_rollouts": {
        "dec_prompt": (
            "Produce 6 product-rollout DECISIONS (80-400 chars): feature-flag "
            "config, A/B-test verdicts, kill-switch activations. Specific "
            "metrics / percentages / file names. Avoid 'cache', 'database', "
            "'auth', 'deploy'. Example: 'Feature flag `new_onboarding_v2` "
            "ramped to 100% after the A/B arm showed +8.4% D7 retention "
            "(p=0.002, n=48k).'"
        ),
        "noi_prompt": (
            "Produce 6 product-rollout NOISE events (80-300 chars). "
            "Transitional phrases. Avoid 'cache', 'database', 'auth', "
            "'deploy'. Example: 'Now I'll look at the experiment dashboard "
            "to see current numbers.'"
        ),
    },
    "billing": {
        "dec_prompt": (
            "Produce 6 billing / finance DECISIONS (80-400 chars): invoice "
            "timing changes, vendor contract signings, pricing moves, "
            "usage-based billing thresholds. Numbers / dates / contract "
            "IDs. Avoid 'cache', 'database', 'auth', 'deploy'. Example: "
            "'Signed Cloudflare enterprise contract 2026-02-14 at "
            "$18,000/mo for 3 years -- replaces the month-to-month "
            "arrangement.'"
        ),
        "noi_prompt": (
            "Produce 6 billing / finance NOISE events (80-300 chars). "
            "Procedural. Avoid 'cache', 'database', 'auth', 'deploy'. "
            "Example: 'Let me pull up the Stripe dashboard to check "
            "invoice status.'"
        ),
    },
    "accessibility": {
        "dec_prompt": (
            "Produce 6 accessibility DECISIONS (80-400 chars): WCAG "
            "compliance gaps, screen-reader fixes, keyboard-nav changes. "
            "Specific widget names / WCAG criterion numbers. Avoid 'cache', "
            "'database', 'auth', 'deploy'. Example: 'Added aria-live=polite "
            "to the toast notification container in Toast.tsx; VoiceOver "
            "now announces new notifications per WCAG 4.1.3.'"
        ),
        "noi_prompt": (
            "Produce 6 accessibility NOISE events (80-300 chars). "
            "Transitional. Avoid 'cache', 'database', 'auth', 'deploy'. "
            "Example: 'Now let me run axe DevTools on this page.'"
        ),
    },
    "i18n": {
        "dec_prompt": (
            "Produce 6 i18n / localization DECISIONS (80-400 chars): "
            "translation-key changes, RTL-layout fixes, plural-rule "
            "corrections. Specific locales / key names / library versions. "
            "Avoid 'cache', 'database', 'auth', 'deploy'. Example: "
            "'Switched Arabic (ar-SA) number formatting from Arabic-Indic "
            "digits to Western in i18n/numbers.ts after user-research "
            "found 73% preference for Western form.'"
        ),
        "noi_prompt": (
            "Produce 6 i18n / localization NOISE events (80-300 chars). "
            "Procedural. Avoid 'cache', 'database', 'auth', 'deploy'. "
            "Example: 'Let me grep the translation files for missing "
            "keys.'"
        ),
    },
}


_OUTPUT_INSTRUCTIONS = (
    "\n\nOutput format: one event per line, NO numbering, NO bullets, "
    "NO surrounding quotes. Each line is one event text. Do not include "
    "any preamble or trailing commentary. Produce exactly 6 events."
)


def build_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY required")
    return genai.Client(api_key=api_key)


def generate_batch(client, model: str, prompt: str) -> list[str]:
    resp = client.models.generate_content(
        model=model, contents=prompt + _OUTPUT_INSTRUCTIONS
    )
    raw = resp.text or ""
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    # Strip possible leading "1. " / "* " prefixes the model might emit anyway
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln)
        ln = re.sub(r"^\"(.*)\"$", r"\1", ln)
        cleaned.append(ln)
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument(
        "--out",
        default=str(
            SCENARIOS_DIR / "disjoint_noise_heavy_holdout.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--domains",
        action="append",
        default=None,
        help="Restrict to specific domain(s); repeatable.",
    )
    args = parser.parse_args()

    domains = args.domains or list(DOMAINS.keys())
    client = None if args.dry_run else build_client()

    events = []
    for dom in domains:
        if dom not in DOMAINS:
            print(f"unknown domain {dom!r}; skipping", file=sys.stderr)
            continue
        for kind in ("dec", "noi"):
            prompt = DOMAINS[dom][f"{kind}_prompt"]
            print(f"  {dom:<20} {kind}: {'[dry-run]' if args.dry_run else 'asking Gemini...'}")
            if args.dry_run:
                continue
            try:
                texts = generate_batch(client, args.model, prompt)
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            for i, t in enumerate(texts):
                if len(t) < 50:
                    continue
                event_id = f"{dom}__{kind}__{i:02d}"
                topic = dom if kind == "dec" else "noise"
                events.append({
                    "id": event_id,
                    "text": t,
                    "topic": topic,
                })

    if args.dry_run:
        print(f"\n[dry-run] would generate events across {len(domains)} domains")
        return 0

    # Shuffle so scenario order doesn't leak domain blocks
    random.Random(42).shuffle(events)

    out_data = {
        "name": "disjoint_noise_heavy_holdout",
        "description": (
            f"Disjoint-topic noise-heavy held-out scenario, generated "
            f"2026-04-18 via Gemini 2.0 Flash. Topics span "
            f"{', '.join(sorted(domains))} -- intentionally DISJOINT "
            f"from the knowledge_update* training scenarios (no cache, "
            f"auth, db, deploy). Used to measure v7 generalization on "
            f"noise-heavy content it did NOT see in training. Replaces "
            f"the contaminated knowledge_update_50t eval -- see "
            f"experiments/nanogpt/contamination_audit.json and "
            f"H_v9_pragmatic in HYPOTHESES.md for context."
        ),
        "events": events,
        # `queries` is required by experiments/loop_quality/scenario.load_scenario;
        # this scenario has no per-query expectations (it is noise-heavy +
        # used as input to nanoGPT eval, NOT as a recall-quality benchmark).
        # Empty list keeps the runner from KeyError'ing.
        "queries": [],
    }
    Path(args.out).write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    n_dec = sum(1 for e in events if e["topic"] != "noise")
    n_noi = sum(1 for e in events if e["topic"] == "noise")
    print(f"\nSaved {len(events)} events ({n_dec} DEC, {n_noi} NOI) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
