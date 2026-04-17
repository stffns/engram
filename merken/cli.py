"""merken CLI — the SDK seen from outside.

Every command maps 1:1 to a ``Memory`` method. No business logic
lives here; if the CLI needs behavior that isn't in ``Memory``, the
method goes in ``Memory`` first and the CLI wraps it.

Defaults, by design:

- **Project** comes from ``--project``, else ``$ENGRAM_PROJECT``,
  else ``"default"``. Same priority order as git for its config.
- **DB** lives at ``~/.merken/<project>.db`` unless ``--db`` is
  passed. This is deliberately **not** your main vstash store — the
  CLI is test-friendly and non-destructive by default. Point at
  your real vstash with ``--db ~/.vstash/memory.db`` when you
  actually want to attach merken to live memory.
- **Output** is human-readable. Pass ``--json`` anywhere in the
  invocation to get structured output you can pipe.

Commands (v1):

    remember       write an event to memory
    recall         query memory through should_recall
    consolidate    episodic → semantic facts
    forget         tombstone episodic events
    audit          query the decision audit log
    tombstones     query the forgotten-events collection
    status         project summary
    stats          pass-through to vstash.Memory.stats

Not in v1 (deferred):

    mcp-serve      MCP server — separate slice
    hooks          Claude Code hooks installer — separate slice
    init           bootstrap — defaults should just work
    migrate        needs vstash upstream changes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from merken import (
    ForgetConsolidated,
    Memory,
    NeverForget,
)

_DEFAULT_PROJECT = "default"


def default_db_path(project: str) -> Path:
    """``~/.merken/<project>.db`` — isolated from any other vstash store."""
    return Path.home() / ".merken" / f"{project}.db"


def _resolve_db(args: argparse.Namespace) -> Path:
    db: Path = args.db or default_db_path(args.project)
    db.parent.mkdir(parents=True, exist_ok=True)
    return db


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _hit_payload(hit: Any) -> dict[str, Any]:
    return {
        "title": getattr(hit, "title", None),
        "path": getattr(hit, "path", None),
        "text": getattr(hit, "text", None),
        "score": getattr(hit, "score", None),
        "chunk": getattr(hit, "chunk", None),
    }


# --------------------------------------------------------------------- commands


def cmd_remember(args: argparse.Namespace) -> int:
    if args.stdin:
        text = sys.stdin.read()
    elif args.text:
        text = args.text
    else:
        print(
            "error: provide TEXT as a positional argument or pass --stdin",
            file=sys.stderr,
        )
        return 2

    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        result = mem.remember(
            text,
            layer=args.layer,
            title=args.title,
            tags=args.tags,
        )

    if args.json:
        payload = {
            "written": result.written,
            "decision": {
                "write": result.decision.write,
                "reason": result.decision.reason,
                "policy": result.decision.policy,
                "confidence": result.decision.confidence,
            },
        }
        print(_json_dump(payload))
    else:
        status = "wrote" if result.written else "skipped"
        mark = "✓" if result.written else "⊘"
        print(
            f"{mark} {status}  reason={result.decision.reason}  "
            f"policy={result.decision.policy}"
        )

    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        hits = mem.recall(args.query, top_k=args.top_k, layer=args.layer)

    if args.json:
        print(_json_dump([_hit_payload(h) for h in hits]))
    else:
        if not hits:
            print("(no hits)")
        else:
            for i, h in enumerate(hits, 1):
                title = (h.title or "")[:70]
                snippet = (h.text or "").replace("\n", " ")[:100]
                print(f"{i}. {title}")
                print(f"   {snippet}")
    return 0


def cmd_recall_briefs(args: argparse.Namespace) -> int:
    """Dual-channel recall: briefs + episodic hits.

    Wrapper over ``Memory.recall_with_briefs``. The hook consumer
    (``~/.claude/hooks/merken-session-start.sh``) prepends the briefs
    to the Claude context so they don't compete with episodic docs in
    the same retrieval pool.
    """
    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        episodic, briefs = mem.recall_with_briefs(
            args.query,
            top_k=args.top_k,
            brief_k=args.brief_k,
            max_brief_tokens=args.max_brief_tokens,
        )

    if args.json:
        print(_json_dump({
            "briefs": briefs,
            "episodic": [_hit_payload(h) for h in episodic],
        }))
    else:
        if briefs:
            print("=== briefs ===")
            for i, b in enumerate(briefs, 1):
                preview = b[:300].replace("\n", " ")
                print(f"{i}. {preview}")
            print()
        if not episodic:
            print("(no episodic hits)")
        else:
            print("=== episodic ===")
            for i, h in enumerate(episodic, 1):
                title = (h.title or "")[:70]
                snippet = (h.text or "").replace("\n", " ")[:100]
                print(f"{i}. {title}")
                print(f"   {snippet}")
    return 0


def _gemini_synthesize_fn():
    """Build a SynthesizeFn backed by Gemini Flash.

    Used by ``--method brief_v1`` to materialize temporal briefs from
    episodic events. Rich enough to pick schemas (DECISION / ENTITY /
    EVENT / FREE), cheap enough to run on PreCompact hooks.
    """
    from google import genai

    key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise SystemExit(
            "brief_v1 needs GEMINI_API_KEY or GOOGLE_API_KEY set."
        )
    client = genai.Client(api_key=key)
    model = os.environ.get("MERKEN_BRIEF_MODEL", "gemini-2.0-flash")

    def synth(prompts: list[str]) -> str:
        prompt = "\n\n".join(prompts) if isinstance(prompts, list) else prompts
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()

    return synth


def cmd_consolidate(args: argparse.Namespace) -> int:
    kwargs: dict[str, Any] = {
        "method": args.method,
        "embedding_threshold": args.threshold,
        "min_cluster": args.min_cluster,
        "force": args.force,
    }
    if args.method == "brief_v1":
        kwargs["synthesize_fn"] = _gemini_synthesize_fn()

    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        result = mem.consolidate(**kwargs)

    if args.json:
        payload = {
            "events_examined": result.events_examined,
            "facts_written": result.facts_written,
            "skipped": result.skipped,
            "reason": result.reason,
            "decider": result.decider,
            "method": result.method,
            "facts": [
                {
                    "text": f.text,
                    "cluster_size": f.cluster_size,
                    "method": f.method,
                    "derived_from": f.derived_from,
                }
                for f in result.facts
            ],
        }
        print(_json_dump(payload))
    else:
        if result.skipped:
            print(
                f"⊘ skipped  reason={result.reason}  "
                f"({result.events_examined} events examined)"
            )
        else:
            print(
                f"✓ {result.facts_written} fact(s) from "
                f"{result.events_examined} events  "
                f"method={result.method}  decider={result.decider}"
            )
            for i, fact in enumerate(result.facts, 1):
                preview = fact.text[:90].replace("\n", " ")
                print(f"  [{i}] size={fact.cluster_size}  {preview}")
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    if args.decider == "consolidated":
        decider = ForgetConsolidated(min_facts=args.min_facts)
    else:
        decider = NeverForget()

    with Memory(
        project=args.project,
        db=_resolve_db(args),
        forget_decider=decider,
    ) as mem:
        result = mem.forget(force=args.force)

    if args.json:
        payload = {
            "tombstoned": result.tombstoned,
            "skipped": result.skipped,
            "events_examined": result.events_examined,
            "decider": result.decider,
        }
        print(_json_dump(payload))
    else:
        print(
            f"✓ tombstoned={len(result.tombstoned)}  "
            f"skipped={len(result.skipped)}  "
            f"events={result.events_examined}  "
            f"decider={result.decider}"
        )
        if args.verbose and result.tombstoned:
            print("  tombstoned:")
            for p in result.tombstoned:
                print(f"    - {p}")
    return 0


_SHADOW_FLAG_TO_MARKER = {
    "shadow_disagree": "shadow_disagree",
    "shadow_agree": "shadow_agree",
    "shadow_error": "shadow_error",
}


def _build_label_backend(name: str):
    """Construct a ``LabelBackend`` from a CLI string identifier."""
    if name == "gemini":
        from merken.labeling import GeminiLabelBackend
        return GeminiLabelBackend()
    raise SystemExit(f"unknown --label-with backend: {name!r}")


def _run_labeling(args: argparse.Namespace) -> int:
    from merken.labeling import label_disagreements

    backend = _build_label_backend(args.label_with)
    print(f"oracle: {backend.name}")

    counts = {"labeled": 0, "skipped": 0, "error": 0}
    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        for status, event_title, label, detail in label_disagreements(
            mem, backend, limit=args.limit
        ):
            counts[status] = counts.get(status, 0) + 1
            if status == "labeled" and label is not None:
                print(
                    f"✓ {event_title[:60]:<60}  "
                    f"{label.decision:<9}  conf={label.confidence:.2f}  "
                    f"{label.rationale[:80]}"
                )
            elif status == "error":
                print(f"✗ {event_title[:60]:<60}  {detail[:80]}")
            # "skipped" stays silent to keep the output compact.

    print()
    print(
        f"labeled={counts['labeled']}  "
        f"skipped={counts['skipped']}  "
        f"error={counts['error']}"
    )
    return 0 if counts["error"] == 0 else 1


def _parse_audit_row(text: str) -> dict[str, str]:
    """Parse the ``key: value`` block emitted by ``format_audit_row``."""
    fields: dict[str, str] = {}
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _shadow_tag_from_reason(reason: str) -> tuple[str, str, str] | None:
    """Extract (marker, shadow_label, shadow_conf) from a reason string.

    Reason format (ShadowWriteDecider):
    ``{primary}|shadow_{agree|disagree|error}:<policy>=<label>:<conf>``.
    Returns None if no shadow tag is present.
    """
    if "|shadow_" not in reason:
        return None
    _, _, tag = reason.partition("|shadow_")
    marker, _, rest = tag.partition(":")
    marker = "shadow_" + marker
    if marker == "shadow_error":
        return (marker, "", rest)
    _, _, label_conf = rest.partition("=")
    label, _, conf = label_conf.partition(":")
    return (marker, label, conf)


def cmd_audit(args: argparse.Namespace) -> int:
    # Shadow filters are mutually exclusive with --query; they override.
    shadow_filter = None
    for flag, marker in _SHADOW_FLAG_TO_MARKER.items():
        if getattr(args, flag, False):
            shadow_filter = marker
            break

    # --label-with implies --shadow-disagree (only disagreements can
    # become training labels) and runs the labeling loop instead of
    # just printing.
    if getattr(args, "label_with", None):
        return _run_labeling(args)

    query = shadow_filter or args.query or "should_"

    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        rows = mem.audit(query=query, top_k=args.top_k)

    # FTS on underscored tokens can over-match adjacent tokens; drop
    # rows that don't actually carry the requested marker.
    if shadow_filter:
        rows = [r for r in rows if shadow_filter in (r.text or "")]

    if args.json:
        print(_json_dump([_hit_payload(r) for r in rows]))
        return 0

    if not rows:
        label = shadow_filter if shadow_filter else "audit rows"
        print(f"(no {label} found)")
        return 0

    for r in rows:
        if shadow_filter:
            fields = _parse_audit_row(r.text or "")
            reason = fields.get("reason", "")
            tag = _shadow_tag_from_reason(reason)
            title = fields.get("event_title") or "(no title)"
            preview = fields.get("event_text_preview", "")[:140]
            timestamp = fields.get("timestamp", "")
            if tag is None:
                print(f"• {title[:80]}  {timestamp}")
                print(f"    {preview}")
                continue
            marker, label, conf = tag
            if marker == "shadow_error":
                verdict = f"shadow_error:{label or conf}"
            else:
                verdict = f"{marker}:{label} ({conf})"
            print(f"• {title[:60]}  [{verdict}]  {timestamp}")
            print(f"    {preview}")
        else:
            print(f"• {(r.title or '(no title)')[:80]}")
            for line in (r.text or "").split("\n")[:5]:
                stripped = line.strip()
                if stripped:
                    print(f"    {stripped[:100]}")
    return 0


def cmd_tombstones(args: argparse.Namespace) -> int:
    query = args.query or "tombstone"
    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        rows = mem.tombstones(query=query, top_k=args.top_k)

    if args.json:
        print(_json_dump([_hit_payload(r) for r in rows]))
    else:
        if not rows:
            print("(no tombstones)")
        else:
            for r in rows:
                print(f"• {(r.title or '(no title)')[:80]}")
                # Tombstone body has "---\n" separating metadata from
                # the preserved text. Show a preview of the text.
                text = r.text or ""
                if "---\n" in text:
                    _, preserved = text.split("---\n", 1)
                    preview = preserved.strip().replace("\n", " ")[:120]
                    print(f"    {preview}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    db = _resolve_db(args)
    with Memory(project=args.project, db=db) as mem:
        docs = mem._vstash.list(collection=mem.collection)
        layers = Counter(d.layer or "(none)" for d in docs)

    if args.json:
        payload = {
            "project": args.project,
            "db": str(db),
            "collection": "default",
            "total_events": sum(layers.values()),
            "layers": dict(layers),
        }
        print(_json_dump(payload))
    else:
        print(f"project:     {args.project}")
        print(f"db:          {db}")
        print("collection:  default")
        print(f"total:       {sum(layers.values())}")
        if layers:
            for layer, n in sorted(layers.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {layer:20}  {n}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with Memory(project=args.project, db=_resolve_db(args)) as mem:
        stats = mem._vstash.stats()

    if args.json:
        payload = {}
        for field in (
            "documents",
            "chunks",
            "collections",
            "db_size_mb",
            "db_path",
        ):
            val = getattr(stats, field, None)
            if val is not None:
                payload[field] = val
        print(_json_dump(payload))
    else:
        print(stats)
    return 0


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="merken",
        description="merken — agent-loop memory, CLI surface over the Python SDK.",
    )
    p.add_argument(
        "--project",
        default=os.environ.get("ENGRAM_PROJECT", _DEFAULT_PROJECT),
        help=(
            "project name (env ENGRAM_PROJECT, default 'default'). "
            "Drives the DB filename when --db is not given."
        ),
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "path to the vstash DB. Default is ~/.merken/<project>.db, "
            "intentionally NOT your main ~/.vstash/memory.db. Pass "
            "~/.vstash/memory.db explicitly to attach merken to your "
            "real store."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of human-readable output",
    )

    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    rem = sub.add_parser(
        "remember",
        help="write an event to memory",
        description=(
            "Pass text as a positional argument or via --stdin. "
            "The should_remember decider may skip the write "
            "(empty text, exact duplicate, etc.); the exit code is "
            "still 0 because the decider's 'no' is a valid outcome."
        ),
    )
    rem.add_argument("text", nargs="?", default=None, help="event text")
    rem.add_argument("--stdin", action="store_true", help="read text from stdin")
    rem.add_argument("--title", default=None, help="optional doc title")
    rem.add_argument("--layer", default="episodic", help="layer tag (default: episodic)")
    rem.add_argument("--tags", default=None, help="comma-separated tags")
    rem.set_defaults(func=cmd_remember)

    rec = sub.add_parser("recall", help="query memory through should_recall")
    rec.add_argument("query", help="natural language query")
    rec.add_argument("--top-k", type=int, default=5)
    rec.add_argument(
        "--layer",
        default=None,
        help="restrict to one layer (skips the decider)",
    )
    rec.set_defaults(func=cmd_recall)

    recb = sub.add_parser(
        "recall-briefs",
        help="dual-channel recall: brief_v1 briefs + episodic hits",
        description=(
            "Same as ``recall`` but also returns the top brief_k briefs "
            "from the semantic layer (filtered to method:brief_v1). "
            "Output is structured so the SessionStart hook can prepend "
            "briefs above episodic in the Claude context."
        ),
    )
    recb.add_argument("query", help="natural language query")
    recb.add_argument("--top-k", type=int, default=5)
    recb.add_argument("--brief-k", type=int, default=3)
    recb.add_argument("--max-brief-tokens", type=int, default=8000)
    recb.set_defaults(func=cmd_recall_briefs)

    cons = sub.add_parser(
        "consolidate",
        help="episodic → semantic facts",
        description=(
            "Cluster episodic events and write one semantic fact per "
            "cluster of size >= --min-cluster. Default method is "
            "embedding_v1 with complete linkage at threshold 0.70 — "
            "calibrated on the loop_quality scenarios."
        ),
    )
    cons.add_argument(
        "--method",
        default="embedding_v1",
        choices=["embedding_v1", "jaccard_v1", "recall_v1", "brief_v1"],
    )
    cons.add_argument("--threshold", type=float, default=0.70)
    cons.add_argument("--min-cluster", type=int, default=2)
    cons.add_argument(
        "--force",
        action="store_true",
        help="bypass the should_consolidate decider",
    )
    cons.set_defaults(func=cmd_consolidate)

    forg = sub.add_parser(
        "forget",
        help="tombstone episodic events (reversible via tombstones)",
        description=(
            "Default decider is NeverForget — the command is a no-op "
            "unless you pass --decider consolidated (tombstone events "
            "already in a fact) or --force (tombstone everything). "
            "Tombstones preserve the full text in the merken_tombstones "
            "collection; nothing is permanently destroyed."
        ),
    )
    forg.add_argument(
        "--decider",
        default="never",
        choices=["never", "consolidated"],
        help="forget policy (default: never)",
    )
    forg.add_argument(
        "--min-facts",
        type=int,
        default=1,
        help="ForgetConsolidated's min_facts (ignored for other deciders)",
    )
    forg.add_argument(
        "--force",
        action="store_true",
        help="tombstone every episodic event regardless of decider",
    )
    forg.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="list tombstoned event paths",
    )
    forg.set_defaults(func=cmd_forget)

    aud = sub.add_parser("audit", help="query the decision audit log")
    aud.add_argument(
        "query",
        nargs="?",
        default=None,
        help="free-text query; default matches every should_* decision",
    )
    aud.add_argument("--top-k", type=int, default=20)
    aud.add_argument(
        "--shadow-disagree",
        action="store_true",
        help="surface events where the shadow decider disagreed with the primary",
    )
    aud.add_argument(
        "--shadow-agree",
        action="store_true",
        help="surface events where the shadow decider agreed with the primary",
    )
    aud.add_argument(
        "--shadow-error",
        action="store_true",
        help="surface events where the shadow decider raised (primary unaffected)",
    )
    aud.add_argument(
        "--label-with",
        choices=("gemini",),
        default=None,
        help=(
            "run oracular labeling on unlabeled shadow_disagree rows with the "
            "chosen backend. Labels are stored in the merken_labels "
            "collection; re-running is safe."
        ),
    )
    aud.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max new labels per --label-with run (default: all)",
    )
    aud.set_defaults(func=cmd_audit)

    tomb = sub.add_parser("tombstones", help="query forgotten events")
    tomb.add_argument(
        "query",
        nargs="?",
        default=None,
        help="free-text query; default matches every tombstone row",
    )
    tomb.add_argument("--top-k", type=int, default=20)
    tomb.set_defaults(func=cmd_tombstones)

    st = sub.add_parser("status", help="project summary (db, collection, layer counts)")
    st.set_defaults(func=cmd_status)

    stats_p = sub.add_parser(
        "stats",
        help="pass-through to vstash.Memory.stats",
    )
    stats_p.set_defaults(func=cmd_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
