"""Real-content smoke test — engram digests the user's own vstash.

This is NOT a scenario-runner benchmark. It has no ground-truth
topic labels because the content was not curated for measurement.
It is a stomach test: does engram's loop behave sensibly on
content the user produced organically, outside the two fixture
scenarios?

What it does:
  1. Opens the user's real vstash (default location, default
     collection).
  2. Pulls the N most recent documents, reads their text.
  3. Writes each one as an episodic event into a *separate*
     throwaway engram Memory so we never touch the user's real DB.
  4. Runs ``consolidate()`` with the current engram defaults.
  5. Reports how many facts came out, shows a summary of each
     cluster with its source document titles so a human can judge
     coherence.
  6. Runs four natural queries through the full recall path and
     shows the top hits.

The output is a qualitative report, not a metric. Pass rate is
what a human says after reading the clusters.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import vstash

from engram import Memory


def _pull_real_docs(n: int) -> list[tuple[str, str, str]]:
    """Pull (path, title, full_text) tuples from the user's vstash.

    Reassembles text by joining chunks. Sorted by added_at desc so
    the newest content wins when ``n`` caps the list.
    """
    src = vstash.Memory()
    docs = src.list(collection="default")
    docs.sort(key=lambda d: d.added_at or "", reverse=True)

    out: list[tuple[str, str, str]] = []
    for doc in docs[:n]:
        chunks = src.get_document_chunks(doc.path, collection="default")
        text = " ".join(chunks).strip()
        if text:
            out.append((doc.path, doc.title or "(untitled)", text))
    return out


def _run_smoke(n_docs: int, queries: list[str]) -> int:
    print(f"=== engram real-vstash smoke test — n_docs={n_docs} ===\n")

    docs = _pull_real_docs(n_docs)
    print(f"pulled {len(docs)} docs from vstash. titles:")
    for _, title, text in docs:
        preview = text[:70].replace("\n", " ")
        print(f"  • {title[:60]:60}  ({len(text)} chars)")
    print()

    with tempfile.TemporaryDirectory(prefix="engram_smoke_real_") as td:
        db = Path(td) / "smoke.db"
        with Memory(project="smoke_real_vstash", db=db) as mem:
            # --- ingest -----------------------------------------------
            # Build path→title map from the ACTUAL vstash paths that
            # engram assigns, not from prefix-matching on the title.
            path_to_title: dict[str, str] = {}
            ingested = 0
            for _, title, text in docs:
                result = mem.remember(text, title=title[:80])
                if result.written and result.ingest is not None:
                    path_to_title[result.ingest.source] = title
                    ingested += 1
            print(f"ingested {ingested} / {len(docs)} (the rest were "
                  f"skipped by should_remember: duplicates or too "
                  f"short)\n")

            # --- consolidate ------------------------------------------
            print("running consolidate(embedding_v1, threshold=0.65, "
                  "complete-link)...")
            result = mem.consolidate()
            print(f"  events_examined: {result.events_examined}")
            print(f"  facts_written:   {result.facts_written}")
            print(f"  decider:         {result.decider}")
            print()

            # --- inspect clusters -------------------------------------
            if result.facts:
                print("=== facts produced (judge coherence by reading) ===")
                for i, fact in enumerate(result.facts, 1):
                    print(f"\n[Fact {i}] cluster_size={fact.cluster_size} "
                          f"method={fact.method}")
                    print(f"  anchor: {fact.text[:160]}")
                    print(f"  derived from {len(fact.derived_from)} event(s):")
                    for path in fact.derived_from:
                        title = path_to_title.get(path, f"(not in map: {path})")
                        print(f"    - {title[:70]}")
                # Also show which docs did NOT end up in any fact
                clustered_paths: set[str] = set()
                for fact in result.facts:
                    clustered_paths.update(fact.derived_from)
                singletons = [
                    (p, t) for p, t in path_to_title.items()
                    if p not in clustered_paths
                ]
                if singletons:
                    print(f"\n=== singletons (not in any cluster) ===")
                    for _, title in singletons:
                        print(f"  - {title[:70]}")

            # --- query through full recall path ----------------------
            print("\n=== recall tests (decider-routed) ===")
            for q in queries:
                print(f"\nQ: {q}")
                hits = mem.recall(q, top_k=3)
                if not hits:
                    print("  (no hits)")
                    continue
                for h in hits:
                    snippet = (h.text or "").replace("\n", " ")[:100]
                    print(f"  • {snippet}")

    print()
    print("=== done. judge the clusters and recalls by reading. ===")
    return 0


def _slugify(title: str) -> str:
    """Approximates vstash's auto-slug from a title for path matching."""
    return title.lower().replace(" ", "-")[:30]


def _best_title_for_path(path: str, docs: list[tuple[str, str, str]]) -> str:
    """Walk the docs list and return the first title whose text() path
    prefix would plausibly match the fact's derived-from path.
    Engram writes with title=doc.title[:80]; vstash auto-slugs and
    adds a timestamp suffix, so exact match is unlikely. We fall
    back to the common-prefix heuristic."""
    tail = path.replace("text://", "")
    best_title = "(unknown)"
    best_score = 0
    for _, title, _ in docs:
        slug = _slugify(title)
        # Count matching prefix chars
        score = 0
        for a, b in zip(slug, tail, strict=False):
            if a == b:
                score += 1
            else:
                break
        if score > best_score:
            best_score = score
            best_title = title
    return best_title


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="engram smoke test against real vstash content",
    )
    p.add_argument("--n-docs", type=int, default=20,
                   help="how many recent vstash docs to pull")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    queries = [
        "what did the MedLocal benchmark achieve?",
        "what vstash bugs or improvements were found?",
        "what are the engram architecture decisions?",
        "what happened in the Kafka merchant pipeline meeting?",
    ]
    return _run_smoke(args.n_docs, queries)


if __name__ == "__main__":
    raise SystemExit(main())
