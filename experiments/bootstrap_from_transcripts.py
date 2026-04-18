# ruff: noqa: I001, E402
"""Bootstrap ``merken_labels`` from Claude Code transcripts (offline replay).

The live PreCompact hook was silently broken for months: it ran
``json.load`` on JSONL transcripts, swallowed the exception via
``2>/dev/null``, and produced zero writes. That's why shadow mode
accumulated zero disagreements despite being "active" on every session.

This script replays the same extraction logic the hook uses (now
fixed), but offline and across every transcript in
``~/.claude/projects/``. Per assistant-text block, it runs the primary
(``HeuristicWriteDecider``) and shadow (``NanoGPTWriteDecider``)
together, catches cases where they disagree, and labels those with an
oracle backend (Gemini by default).

Key differences vs. live mode:

- Primary is instantiated without ``set_hydrate_fn`` hookup. We
  intentionally skip hydration against existing vstash content so the
  "novelty" check degrades to "unseen in this run". The trade-off is
  that some candidate events may be duplicates of already-ingested
  content. That's fine for bootstrap -- the oracle labels the raw
  text; downstream dedup happens via the content-hash in the label
  title.
- Labels land in the same per-project ``merken_labels`` collection the
  live pipeline writes to. ``source: bootstrap_from_transcripts`` in
  the label body distinguishes them from organic labels.

Usage::

    # dry-run: see candidates and disagreement volume per project
    python3 -m experiments.bootstrap_from_transcripts --dry-run

    # real labels, engram project, first 30 disagreements
    python3 -m experiments.bootstrap_from_transcripts \
        --project engram --limit-per-project 30

Env knobs:

    GOOGLE_API_KEY / GEMINI_API_KEY   required for --backend gemini
    MERKEN_LABEL_LLM_MODEL            model name for --backend local
    MERKEN_LABEL_LLM_DEVICE           cpu | cuda | mps
"""

# torch FIRST: fastembed + torch load-order segfault on macOS.
import torch  # noqa: F401

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from merken import Memory
from merken.classifiers.nanogpt import NanoGPTWriteDecider
from merken.labeling import (
    GeminiLabelBackend,
    LocalLLMLabelBackend,
)
from merken.policies.should_remember import HeuristicWriteDecider
from merken.policies.types import Event, WriteContext


TRANSCRIPTS_ROOT = Path.home() / ".claude" / "projects"
MERKEN_DBS_DIR = Path.home() / ".merken"

# No hardcoded fallback: users must pass --ckpt/--meta or set the env
# vars that the live shadow-mode hook already uses
# (MERKEN_SHADOW_NANOGPT_CKPT / MERKEN_SHADOW_NANOGPT_META).
DEFAULT_CKPT = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
DEFAULT_META = os.environ.get("MERKEN_SHADOW_NANOGPT_META")


def infer_project_from_dir(transcript_dir: Path) -> str:
    name = transcript_dir.name
    if not name.startswith("-"):
        return name
    parts = [p for p in name.split("-") if p]
    return parts[-1] if parts else name


def extract_assistant_texts(
    jsonl_path: Path, *, min_len: int = 50, max_len: int = 2000
):
    """Yield text blocks from assistant messages. Mirrors the fixed hook."""
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


def label_title_for(text: str) -> str:
    """Stable label title tying a label to one candidate event.

    First line (truncated) plus a content hash so re-runs are
    idempotent and same-text events across projects are distinguishable
    at a glance.
    """
    first = text.strip().split("\n", 1)[0][:80].strip()
    first = re.sub(r"[^\w\-\s.,:;=()\[\]%/]", "_", first) or "event"
    return f"{first} [h={content_hash(text)}]"


def load_project_memory(
    project: str, *, memories_cache: dict[str, Memory]
) -> Memory:
    mem = memories_cache.get(project)
    if mem is not None:
        return mem
    db_path = MERKEN_DBS_DIR / f"{project}.db"
    mem = Memory(project=project, db=str(db_path))
    memories_cache[project] = mem
    return mem


def load_seen_labels(mem: Memory) -> set[str]:
    seen: set[str] = set()
    for row in mem.search_labels(top_k=1000):
        title = getattr(row, "title", "") or ""
        if title.startswith("label:"):
            seen.add(title.removeprefix("label:"))
    return seen


def process_project(
    project_dir: Path,
    *,
    shadow: NanoGPTWriteDecider,
    backend,
    memories_cache: dict[str, Memory],
    seen_labels_cache: dict[str, set[str]],
    limit: int | None,
    dry_run: bool,
) -> dict[str, int]:
    project = infer_project_from_dir(project_dir)
    mem = load_project_memory(project, memories_cache=memories_cache)
    if project not in seen_labels_cache:
        seen_labels_cache[project] = load_seen_labels(mem)
    seen = seen_labels_cache[project]
    ctx = WriteContext(project=project)
    # Fresh primary per project: HeuristicWriteDecider accumulates a
    # `_seen` set for dedup. Sharing that across projects would let
    # event-text similarity in project A shadow-skip a real novel event
    # in project B, masking real disagreements.
    primary = HeuristicWriteDecider()

    # Per-project content-hash dedup: one label per distinct text, even
    # if it recurs across many transcripts (a common pattern for agent
    # boilerplate or repeated status summaries).
    local_hashes: set[str] = set()

    counts = {
        "files": 0,
        "candidates": 0,
        "dup_in_run": 0,
        "agree_write": 0,
        "agree_skip": 0,
        "disagree_shadow_skip": 0,
        "disagree_shadow_write": 0,
        "already_labeled": 0,
        "labeled": 0,
        "errors": 0,
    }
    labeled_this_project = 0

    # rglob: subagent transcripts live in `<session_id>/subagents/`.
    for transcript in sorted(project_dir.rglob("*.jsonl")):
        counts["files"] += 1
        for text in extract_assistant_texts(transcript):
            counts["candidates"] += 1

            h = content_hash(text)
            if h in local_hashes:
                counts["dup_in_run"] += 1
                continue
            local_hashes.add(h)

            title = label_title_for(text)
            if title in seen:
                counts["already_labeled"] += 1
                continue

            event = Event(text=text)
            pdec = primary.decide(event, ctx)
            sdec = shadow.decide(event, ctx)

            if pdec.write and sdec.write:
                counts["agree_write"] += 1
                continue
            if not pdec.write and not sdec.write:
                counts["agree_skip"] += 1
                continue

            if pdec.write and not sdec.write:
                counts["disagree_shadow_skip"] += 1
                kind = "shadow_skip"
            else:
                counts["disagree_shadow_write"] += 1
                kind = "shadow_write"

            print(
                f"  [{project}/{transcript.name}] {kind} "
                f"primary={pdec.write} shadow={sdec.write} "
                f"[{sdec.reason}] {title[:70]}"
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
                    "kind": kind,
                    "primary_wrote": pdec.write,
                    "shadow_wrote": sdec.write,
                    "primary_reason": pdec.reason,
                    "shadow_reason": sdec.reason,
                    # Store the full candidate text, not a 500-char
                    # preview. The oracle saw up to 2000 chars and we
                    # want the training extract to include the same
                    # context the oracle ruled on. Field name stays
                    # `event_text_preview` for schema compat with the
                    # live labeling path.
                    "event_text_preview": text,
                    "transcript_file": transcript.name,
                    "source": "bootstrap_from_transcripts",
                },
                ensure_ascii=False,
                indent=2,
            )
            mem.remember_label(event_title=title, body=body)
            seen.add(title)
            counts["labeled"] += 1
            labeled_this_project += 1
            print(
                f"    -> label={label.decision} conf={label.confidence:.2f} "
                f"backend={label.backend}"
            )

            if limit is not None and labeled_this_project >= limit:
                return counts
        if limit is not None and labeled_this_project >= limit:
            return counts

    return counts


def discover_project_dirs(wanted: list[str] | None) -> list[Path]:
    """Find ``~/.claude/projects/*`` subdirs, optionally filtered.

    With ``wanted``, a dir matches if its inferred merken project name
    is in the filter. So ``--project engram`` picks up both the main
    dir and any worktree-style variants ending in ``-engram``.
    """
    out: list[Path] = []
    if not TRANSCRIPTS_ROOT.exists():
        print(
            f"transcripts root missing: {TRANSCRIPTS_ROOT}. "
            f"Claude Code has not written any transcripts yet."
        )
        return out
    for sub in sorted(TRANSCRIPTS_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        project = infer_project_from_dir(sub)
        if wanted is not None and project not in wanted:
            continue
        out.append(sub)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap merken_labels from Claude Code transcripts."
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
        help="Max new labels per project (default: unlimited).",
    )
    parser.add_argument(
        "--backend", choices=["gemini", "local"], default="gemini"
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print disagreements without calling the oracle.",
    )
    args = parser.parse_args()

    project_dirs = discover_project_dirs(args.project)
    if not project_dirs:
        print("no project transcript dirs matched; nothing to do.")
        return 1

    if not args.ckpt or not args.meta:
        print(
            "nanoGPT checkpoint/meta path missing. Pass --ckpt / --meta "
            "or set MERKEN_SHADOW_NANOGPT_CKPT / "
            "MERKEN_SHADOW_NANOGPT_META (same vars the hook uses)."
        )
        return 2

    print(f"loading nanoGPT v6 from {args.ckpt}")
    shadow = NanoGPTWriteDecider(args.ckpt, args.meta)

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

    memories_cache: dict[str, Memory] = {}
    seen_labels_cache: dict[str, set[str]] = {}
    grand: dict[str, int] = {
        "files": 0,
        "candidates": 0,
        "dup_in_run": 0,
        "agree_write": 0,
        "agree_skip": 0,
        "disagree_shadow_skip": 0,
        "disagree_shadow_write": 0,
        "already_labeled": 0,
        "labeled": 0,
        "errors": 0,
    }

    for pdir in project_dirs:
        project = infer_project_from_dir(pdir)
        print(f"\n=== {project} ({pdir.name}) ===")
        try:
            stats = process_project(
                pdir,
                shadow=shadow,
                backend=backend,
                memories_cache=memories_cache,
                seen_labels_cache=seen_labels_cache,
                limit=args.limit_per_project,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        for k, v in stats.items():
            grand[k] = grand.get(k, 0) + v
        print(f"  stats: {stats}")

    print("\n=== totals ===")
    for k in (
        "files",
        "candidates",
        "dup_in_run",
        "agree_write",
        "agree_skip",
        "disagree_shadow_skip",
        "disagree_shadow_write",
        "already_labeled",
        "labeled",
        "errors",
    ):
        print(f"  {k}: {grand[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())