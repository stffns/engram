# ruff: noqa: I001, E402
"""Oracle a sample of agree_write cases to close v7's confusion matrix.

The bootstrap pipeline only labeled disagreement cases (shadow said
SKIP while primary said WRITE). That gives us the TRUE_NEGATIVE and
FALSE_NEGATIVE columns but not the WRITE column. Without oracling a
sample of agree_write events we cannot compute v7's precision on
writes, overall accuracy, or the "classifier says WRITE" row of the
full 2x2 matrix.

This script:

1. Re-replays Claude Code transcripts exactly as bootstrap_from_transcripts
   did, using the same primary (HeuristicWriteDecider) and shadow
   (NanoGPTWriteDecider v7) and finding cases where BOTH say WRITE.
2. Reservoir-samples up to N agree_write events. The sample is
   uniform OVER EVENTS (Algorithm R), so projects with more events
   contribute more to the sample. If you need strict per-project
   stratification, post-filter the output by `project` field.
3. Oracles each with Gemini 2.0 Flash using the SAME strict prompt
   that relabeled the DECISION pile -- consistency matters for
   combining with the existing 1026 labels in downstream metrics.
4. Writes results to data/merken_labels_agree_write.jsonl. Does NOT
   touch vstash merken_labels collection (keep bootstrap-disagreement
   and agree-write sets separable).

Env knobs:
    MERKEN_SHADOW_NANOGPT_CKPT   path to v7 ckpt.pt
    MERKEN_SHADOW_NANOGPT_META   path to v7 meta.pkl
    GOOGLE_API_KEY / GEMINI_API_KEY
"""

from __future__ import annotations

import torch  # noqa: F401 (Mistake #10)

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from google import genai

from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.labeling import _parse_backend_response
from merken.policies.should_remember import HeuristicWriteDecider
from merken.policies.types import Event, WriteContext


TRANSCRIPTS_ROOT = Path.home() / ".claude" / "projects"


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
    "(e.g. 'Let me check the config file next') is NOT a decision.\n\n"
    "Event:\n"
    "<<<\n"
    "{text}\n"
    ">>>\n\n"
    "Respond on a single line in this exact format (no prose, no code\n"
    "fences):\n"
    "LABEL: <DECISION|NOISE|UNCERTAIN> | CONFIDENCE: <number 0.0-1.0> "
    "| REASON: <one sentence>"
)


def infer_project(path: Path) -> str:
    name = path.name
    if not name.startswith("-"):
        return name
    parts = [p for p in name.split("-") if p]
    return parts[-1] if parts else name


def extract_assistant_texts(jsonl_path: Path, min_len=50, max_len=2000):
    try:
        fh = open(jsonl_path, encoding="utf-8")
    except Exception:
        return
    with fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "assistant":
                continue
            content = (rec.get("message") or {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text") or ""
                        if len(t) >= min_len:
                            yield t[:max_len]
            elif isinstance(content, str) and len(content) >= min_len:
                yield content[:max_len]


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def scan_agree_write(shadow, limit_scan=None):
    """Yield (project, text, hash) for every agree_write candidate."""
    if not TRANSCRIPTS_ROOT.exists():
        return
    seen_hashes: set[str] = set()
    n = 0
    for proj_dir in sorted(TRANSCRIPTS_ROOT.iterdir()):
        if not proj_dir.is_dir():
            continue
        project = infer_project(proj_dir)
        primary = HeuristicWriteDecider()  # fresh per project
        ctx = WriteContext(project=project)
        for transcript in sorted(proj_dir.rglob("*.jsonl")):
            for text in extract_assistant_texts(transcript):
                h = content_hash(text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                ev = Event(text=text)
                if not primary.decide(ev, ctx).write:
                    continue
                if not shadow.decide(ev, ctx).write:
                    continue
                yield project, text, h
                n += 1
                if limit_scan and n >= limit_scan:
                    return


def reservoir_sample(iterator, k: int, seed: int):
    """Algorithm R: uniform sample k items from a stream of unknown length."""
    rng = random.Random(seed)
    reservoir: list = []
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def build_gemini_client(model: str = "gemini-2.0-flash"):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY required")
    return genai.Client(api_key=api_key), model


def label_one(client, model: str, text: str):
    # Using .replace instead of .format because raw transcript text
    # routinely contains `{` / `}` (JSON, f-strings, code). Those would
    # be interpreted as format placeholders and raise KeyError.
    prompt = STRICT_PROMPT.replace("{text}", text[:3000])
    resp = client.models.generate_content(model=model, contents=prompt)
    return _parse_backend_response(resp.text or "", model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="sample size (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="data/merken_labels_agree_write.jsonl",
        help="output JSONL path relative to repo root",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ckpt", default=os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
    )
    parser.add_argument(
        "--meta", default=os.environ.get("MERKEN_SHADOW_NANOGPT_META")
    )
    parser.add_argument(
        "--model", default="gemini-2.0-flash", help="Gemini model id"
    )
    parser.add_argument(
        "--limit-scan",
        type=int,
        default=None,
        help="Stop scanning transcripts after N agree_write hits (for testing).",
    )
    args = parser.parse_args()

    if not args.ckpt or not args.meta:
        raise SystemExit(
            "Pass --ckpt / --meta or set MERKEN_SHADOW_NANOGPT_CKPT / "
            "MERKEN_SHADOW_NANOGPT_META (used for v7 shadow scoring)."
        )

    print(f"loading shadow (v7) from {args.ckpt}")
    shadow = NanoGPTWriteDecider(args.ckpt, args.meta)

    print("scanning transcripts for agree_write cases...")
    stream = scan_agree_write(shadow, limit_scan=args.limit_scan)
    sample = reservoir_sample(stream, args.n, args.seed)
    print(f"reservoir filled with {len(sample)} agree_write cases")

    if args.dry_run:
        print("--dry-run: skipping labeling. First 5 samples:")
        for p, t, h in sample[:5]:
            print(f"  [{p}] {h}  {t[:100]!r}")
        return 0

    client, model_id = build_gemini_client(args.model)
    print(f"labeling with {model_id} using strict prompt...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing labels to make re-runs idempotent (hash-keyed).
    done_hashes: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                if r.get("hash"):
                    done_hashes.add(r["hash"])
            except Exception:
                continue

    counts = {"labeled": 0, "skipped_done": 0, "errors": 0,
              "DECISION": 0, "NOISE": 0, "UNCERTAIN": 0}

    with out_path.open("a", encoding="utf-8") as out:
        for i, (project, text, h) in enumerate(sample, 1):
            if h in done_hashes:
                counts["skipped_done"] += 1
                continue
            try:
                label = label_one(client, model_id, text)
            except Exception as e:
                counts["errors"] += 1
                print(f"  [{i}/{len(sample)}] ERROR: {type(e).__name__}: {e}")
                continue

            rec = {
                "hash": h,
                "project": project,
                "text": text,
                "label": label.decision,
                "confidence": label.confidence,
                "rationale": label.rationale,
                "backend": label.backend,
                "kind": "agree_write",
                "source": "oracle_agree_write_sample",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["labeled"] += 1
            counts[label.decision] = counts.get(label.decision, 0) + 1

            if i % 25 == 0:
                print(
                    f"  [{i}/{len(sample)}] labeled so far: "
                    f"{counts['labeled']} "
                    f"(D:{counts['DECISION']} N:{counts['NOISE']} "
                    f"U:{counts['UNCERTAIN']}) errors={counts['errors']}"
                )

    print("\n=== totals ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())