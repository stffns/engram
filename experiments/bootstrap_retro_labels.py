"""Bootstrap ``merken_labels`` retroactively from historical audit rows.

Graduation criterion #4 (>=200 oracular labels with >=95% agreement) is
gated on shadow-mode accumulation in production. Hooks are live, but
PreCompact fires rarely, so accumulation sits at zero. Meanwhile every
project in ``~/.merken/`` already has dozens of ``should_remember``
audit rows from the HeuristicWriteDecider primary.

This script replays nanoGPT v6 (BPE) offline over those historical
events, identifies where the shadow would have disagreed with the
primary, and asks an oracle (Gemini by default) to label the
disagreement. Output lands in the project's own ``merken_labels``
collection, same schema as ``merken audit --label-with gemini``.

Idempotent: re-running skips events already labeled (keyed on
``event_title``).

Why this is honest:

- It does not invent new shadow data: every label covers a real
  historical event that primary wrote or skipped.
- It uses the same label schema the live hook path uses, so bootstrap
  labels and organic labels pile up together.
- It runs the same v6 checkpoint that shadow mode runs, so
  "disagreement" means the exact same thing.

Usage::

    # dry-run first: see how many disagreements exist, no API calls
    python3 -m experiments.bootstrap_retro_labels --dry-run

    # label first 20 disagreements in engram only (cheap sanity check)
    python3 -m experiments.bootstrap_retro_labels --project engram --limit-per-project 20

    # full run with a safety cap per project
    python3 -m experiments.bootstrap_retro_labels --limit-per-project 50

Env knobs:

    GOOGLE_API_KEY / GEMINI_API_KEY   required for --backend gemini
    MERKEN_LABEL_LLM_MODEL            model name for --backend local
    MERKEN_LABEL_LLM_DEVICE           cpu | cuda | mps
"""

# torch FIRST: fastembed + torch load-order segfault on macOS.
# See notes/nanogpt-training-log.md Mistake #10.
import torch  # noqa: F401

import argparse
import json
import os
from pathlib import Path

from merken import Memory
from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.labeling import (
    GeminiLabelBackend,
    LocalLLMLabelBackend,
    _parse_audit_body,
)
from merken.policies.types import Event, WriteContext


MERKEN_DBS_DIR = Path.home() / ".merken"

# No hardcoded fallback: users must pass --ckpt/--meta or set the same
# env vars the live shadow-mode hook reads.
DEFAULT_CKPT = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
DEFAULT_META = os.environ.get("MERKEN_SHADOW_NANOGPT_META")


def find_project_dbs(wanted: list[str] | None) -> list[Path]:
    dbs = sorted(MERKEN_DBS_DIR.glob("*.db"))
    if wanted:
        names = set(wanted)
        dbs = [p for p in dbs if p.stem in names]
    return dbs


def iter_remember_audits(mem: Memory, top_k: int):
    # fts_only=False: FTS tokenizer splits "should_remember" into low-IDF
    # pieces and drops most audit rows. Hybrid search recovers the full
    # set via the vector layer. We re-filter on the parsed `decision`
    # field below so false-positive rows (should_recall / should_forget)
    # are dropped.
    rows = mem.audit(query="should_remember", top_k=top_k, fts_only=False)
    for r in rows:
        text = r.text or ""
        if "decision: should_remember" not in text:
            continue
        fields = _parse_audit_body(text)
        if fields.get("decision") != "should_remember":
            continue
        yield fields


def already_labeled(mem: Memory, top_k: int = 1000) -> set[str]:
    seen: set[str] = set()
    for row in mem.search_labels(top_k=top_k):
        title = getattr(row, "title", "") or ""
        if title.startswith("label:"):
            seen.add(title.removeprefix("label:"))
    return seen


def bootstrap_project(
    db_path: Path,
    decider: NanoGPTWriteDecider,
    backend,
    *,
    limit: int | None,
    include_skipped: bool,
    dry_run: bool,
    audit_top_k: int,
) -> dict[str, int]:
    project = db_path.stem
    mem = Memory(project=project, db=str(db_path))
    ctx = WriteContext(project=project)

    seen_labels = already_labeled(mem)
    counts = {
        "scanned": 0,
        "agree": 0,
        "disagree": 0,
        "labeled": 0,
        "skipped_no_text": 0,
        "skipped_primary_skipped": 0,
        "skipped_already_labeled": 0,
        "errors": 0,
    }

    for fields in iter_remember_audits(mem, top_k=audit_top_k):
        counts["scanned"] += 1
        title = (fields.get("event_title") or "").strip()
        text = (fields.get("event_text_preview") or "").strip()

        if not title or not text:
            counts["skipped_no_text"] += 1
            continue

        primary_wrote = (fields.get("write") or "").strip().lower() == "true"
        if not primary_wrote and not include_skipped:
            counts["skipped_primary_skipped"] += 1
            continue

        if title in seen_labels:
            counts["skipped_already_labeled"] += 1
            continue

        dec = decider.decide(Event(text=text), ctx)
        shadow_wrote = dec.write

        if shadow_wrote == primary_wrote:
            counts["agree"] += 1
            continue

        counts["disagree"] += 1
        print(
            f"  disagree: primary={primary_wrote} shadow={shadow_wrote} "
            f"[{dec.reason}] {title[:80]}"
        )

        if dry_run:
            continue

        try:
            label = backend.label(text)
        except Exception as e:
            counts["errors"] += 1
            print(f"    ERROR labeling: {type(e).__name__}: {e}")
            continue

        body = json.dumps(
            {
                "decision": label.decision,
                "confidence": label.confidence,
                "rationale": label.rationale,
                "backend": label.backend,
                "shadow_marker": "shadow_disagree",
                "primary_wrote": primary_wrote,
                "shadow_wrote": shadow_wrote,
                "shadow_reason": dec.reason,
                "event_text_preview": text[:500],
                "source": "bootstrap_retro_labels",
            },
            ensure_ascii=False,
            indent=2,
        )
        mem.remember_label(event_title=title, body=body)
        seen_labels.add(title)
        counts["labeled"] += 1
        print(
            f"    -> label={label.decision} conf={label.confidence:.2f} "
            f"backend={label.backend}"
        )

        if limit is not None and counts["labeled"] >= limit:
            break

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap merken_labels from historical audit rows."
    )
    parser.add_argument(
        "--project",
        action="append",
        default=None,
        help="Bootstrap this project only; repeatable. Default: all ~/.merken/*.db.",
    )
    parser.add_argument(
        "--limit-per-project",
        type=int,
        default=None,
        help="Max new labels per project (default: unlimited).",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "local"],
        default="gemini",
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Also run shadow over audit rows where primary skipped "
        "(write=False). Default: only primary-wrote rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print disagreements, do not call the oracle, do not write labels.",
    )
    parser.add_argument(
        "--audit-top-k",
        type=int,
        default=1000,
        help="Per-project audit pagination cap (vstash hard max: 1000).",
    )
    args = parser.parse_args()

    projects = find_project_dbs(args.project)
    if not projects:
        print("no projects matched; nothing to do.")
        return 1

    if not args.ckpt or not args.meta:
        print(
            "nanoGPT checkpoint/meta path missing. Pass --ckpt / --meta "
            "or set MERKEN_SHADOW_NANOGPT_CKPT / "
            "MERKEN_SHADOW_NANOGPT_META (same vars the hook uses)."
        )
        return 2

    print(f"loading nanoGPT v6 from {args.ckpt}")
    decider = NanoGPTWriteDecider(args.ckpt, args.meta)

    backend = None
    if not args.dry_run:
        if args.backend == "gemini":
            backend = GeminiLabelBackend()
        else:
            backend = LocalLLMLabelBackend(
                model_name=os.environ.get(
                    "MERKEN_LABEL_LLM_MODEL", "google/gemma-3-1b-it"
                ),
                device=os.environ.get("MERKEN_LABEL_LLM_DEVICE", "cpu"),
            )
        print(f"oracle backend: {backend.name}")

    grand: dict[str, int] = {
        "scanned": 0,
        "agree": 0,
        "disagree": 0,
        "labeled": 0,
        "skipped_no_text": 0,
        "skipped_primary_skipped": 0,
        "skipped_already_labeled": 0,
        "errors": 0,
    }
    for db in projects:
        print(f"\n=== {db.stem} ({db}) ===")
        try:
            stats = bootstrap_project(
                db,
                decider,
                backend,
                limit=args.limit_per_project,
                include_skipped=args.include_skipped,
                dry_run=args.dry_run,
                audit_top_k=args.audit_top_k,
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        for k, v in stats.items():
            grand[k] = grand.get(k, 0) + v
        print(f"  stats: {stats}")

    print("\n=== totals ===")
    for k in (
        "scanned",
        "agree",
        "disagree",
        "labeled",
        "skipped_no_text",
        "skipped_primary_skipped",
        "skipped_already_labeled",
        "errors",
    ):
        print(f"  {k}: {grand[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
