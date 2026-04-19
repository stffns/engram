"""Backfill episodic events from Claude Code transcripts that the
PreCompact hook never ingested.

Context: from 2026-04-18 (after the hook fix that started SHADOW
mode + brief_v1) until 2026-04-19 (today), the PreCompact hook only
fired ONCE in engram (and similar in other projects). Every session
between then and now produced transcripts but no merken episodic
events. The data exists in ~/.claude/projects/*/<session>.jsonl --
this script replays what the hook should have done.

Behavior mirrors `~/.claude/hooks/merken-save.sh`:
  - Only `type:assistant` records.
  - Only `text` blocks (not tool_use / tool_result).
  - Skip blocks shorter than 50 chars (matches hook).
  - Truncate each block to 2000 chars (matches hook).
  - Tag with `source:backfill,session:<session_id>` so backfilled
    rows are distinguishable from live ingests.

Skips subagent transcripts (subagents/) -- those are conversational
noise, not user-facing content. Top-level session transcripts only.

Dedup happens for free at the merken layer (HeuristicWriteDecider's
`_seen` set + hydrate_fn against existing episodic).

Usage:
  python -m experiments.probes.backfill_episodic --project engram
  python -m experiments.probes.backfill_episodic --project engram --dry-run
  python -m experiments.probes.backfill_episodic --all       # ALL projects
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def parse_transcript(path: Path) -> list[tuple[str, str, str]]:
    """Yield (project, session_id, text) per assistant text block.

    project is taken from the per-record `cwd` field (basename),
    not from the directory name -- the directory uses path-encoded
    `/`->`-` substitution and is ambiguous when project basenames
    contain hyphens.
    """
    out = []
    try:
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") != "assistant":
                    continue
                cwd = rec.get("cwd") or ""
                project = Path(cwd).name if cwd else path.parent.name
                session = rec.get("sessionId") or rec.get("session_id") or "unknown"
                content = (rec.get("message") or {}).get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = (block.get("text") or "").strip()
                            if len(t) >= 50:
                                out.append((project, session, t[:2000]))
                elif isinstance(content, str) and len(content.strip()) >= 50:
                    out.append((project, session, content.strip()[:2000]))
    except Exception:
        pass
    return out


def project_dirs() -> list[Path]:
    if not CLAUDE_PROJECTS.exists():
        return []
    return [p for p in CLAUDE_PROJECTS.iterdir() if p.is_dir()]


def discover_texts(project_filter: str | None) -> dict[str, list[tuple[str, str]]]:
    """Return {project: [(session_id, text), ...]} from top-level transcripts only."""
    by_project: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n_files = 0
    for proj_dir in project_dirs():
        # only top-level *.jsonl, not subagent jsonl files under subdirs
        for transcript in sorted(proj_dir.glob("*.jsonl")):
            n_files += 1
            for project, session, text in parse_transcript(transcript):
                if project_filter and project != project_filter:
                    continue
                by_project[project].append((session, text))
    print(f"scanned {n_files} top-level transcript files across "
          f"{len(project_dirs())} project dirs", file=sys.stderr)
    return by_project


def setup_env_for_project(project: str, v7_ckpt: Path, v7_meta: Path) -> None:
    """Mirror merken-save.sh env: engram -> PRIMARY, others -> SHADOW."""
    if project == "engram":
        os.environ.setdefault("MERKEN_PRIMARY", "nanogpt")
        os.environ.setdefault("MERKEN_PRIMARY_NANOGPT_CKPT", str(v7_ckpt))
        os.environ.setdefault("MERKEN_PRIMARY_NANOGPT_META", str(v7_meta))
        os.environ.setdefault("MERKEN_PRIMARY_NANOGPT_CALIBRATOR", "default")
    else:
        os.environ.setdefault("MERKEN_SHADOW", "nanogpt")
        os.environ.setdefault("MERKEN_SHADOW_NANOGPT_CKPT", str(v7_ckpt))
        os.environ.setdefault("MERKEN_SHADOW_NANOGPT_META", str(v7_meta))
        os.environ.setdefault("MERKEN_SHADOW_NANOGPT_CALIBRATOR", "default")


def ingest_project(project: str, items: list[tuple[str, str]], dry_run: bool) -> dict:
    """Ingest texts into merken's per-project DB."""
    seen_text = set()
    unique = []
    for session, text in items:
        if text in seen_text:
            continue
        seen_text.add(text)
        unique.append((session, text))
    print(f"  unique texts (in-batch dedup): {len(unique)} (of {len(items)})",
          file=sys.stderr)
    if dry_run:
        return {"would_ingest": len(unique), "ingested": 0, "skipped": 0}

    # Lazy-import merken so MERKEN_PRIMARY env vars are honored.
    from merken import Memory

    mem = Memory(project=project)
    ingested = 0
    skipped = 0
    for session, text in unique:
        try:
            r = mem.remember(text, tags=f"source:backfill,session:{session}")
            if r.written:
                ingested += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"    ingest error: {type(e).__name__}: {e}", file=sys.stderr)
    return {"would_ingest": len(unique), "ingested": ingested, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="Backfill only this project (default: engram)")
    ap.add_argument("--all", action="store_true", help="Backfill all projects")
    ap.add_argument("--dry-run", action="store_true", help="Don't write, just count")
    ap.add_argument("--consolidate", action="store_true",
                    help="Run brief_v1 consolidate after ingest (per project)")
    args = ap.parse_args()

    if not args.project and not args.all:
        args.project = "engram"  # safe default

    v7_ckpt = Path.home() / "Desktop/Personal/Projects/nanoGPT/out-merken-bpe-v7/ckpt.pt"
    v7_meta = Path.home() / "Desktop/Personal/Projects/nanoGPT/data/merken_bpe_v7/meta.pkl"
    if not v7_ckpt.exists() or not v7_meta.exists():
        print("WARN: v7 ckpt/meta not found; continuing without classifier",
              file=sys.stderr)
        v7_ckpt = v7_meta = None

    pf = None if args.all else args.project
    by_project = discover_texts(pf)
    if not by_project:
        print(f"No texts found for project={pf!r}; nothing to do.", file=sys.stderr)
        return 0

    print(f"Projects with text: {len(by_project)}", file=sys.stderr)
    for p, items in sorted(by_project.items(), key=lambda kv: -len(kv[1])):
        print(f"  {p:<60} {len(items)} texts")

    summary = {}
    for project, items in by_project.items():
        print(f"\n=== Backfilling {project} ===", file=sys.stderr)
        if v7_ckpt:
            setup_env_for_project(project, v7_ckpt, v7_meta)
        result = ingest_project(project, items, args.dry_run)
        summary[project] = result
        print(f"  result: {result}", file=sys.stderr)

        if args.consolidate and not args.dry_run:
            print(f"  consolidating briefs...", file=sys.stderr)
            import subprocess
            r = subprocess.run(
                ["merken", "--project", project, "consolidate", "--method", "brief_v1"],
                capture_output=True, text=True, timeout=120,
            )
            tail = (r.stdout + r.stderr).strip().splitlines()
            for line in tail[-5:]:
                print(f"    {line}", file=sys.stderr)

    out_path = Path(__file__).parent / "backfill_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
