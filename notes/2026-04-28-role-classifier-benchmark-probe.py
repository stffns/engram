"""Probe the V2 SCR RoleClassifier on the actual ingest unit of each
benchmark.

Runs the V2 SCR RoleClassifier over the *exact* text shape the merken
pipeline ingests in each benchmark, per the granularity lessons we
already learned:

- **LoCoMo**: ingest per-session. Each session is the [date_time]
  header followed by ``"\\n".join("{speaker}: {text}")`` for every
  turn. This is what runner_rerank.py / runner_phase2.py / runner.py
  all do. Per the 2026-04-25 finding, per-turn fragments below the
  grain that contains the answer.
- **LongMemEval**: ingest per-turn. LME turns are ~967 chars
  individually (document-sized), so per-session blobs would
  destructively chunk. Per-turn is the natural unit for LME.

We classify each ingestable directly (not via Memory.remember) so the
probe can run in seconds. The Memory.remember integration is already
covered by tests/test_memory_role_classifier_integration.py; this
probe asks the orthogonal question: what role tags would we attach
if we turned the classifier on for the actual benchmark pipelines?

Run: ``python -m notes.2026-04-28-role-classifier-benchmark-probe``
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from merken.role_classifier import RoleClassifier

LOCOMO_PATH = Path("experiments/retrieval/locomo/data/locomo10.json")
LME_PATH = Path("experiments/retrieval/longmemeval/.cache/longmemeval_oracle.json")

SAMPLE_SIZE = 100
SEED = 42
LOW_CONF_THRESHOLD = 0.05


# ----------------------------------------------------------------- LoCoMo

def _build_locomo_per_session() -> list[tuple[str, str]]:
    """Mirror runner_rerank._ingest_per_session: one ingestable per
    session, formatted as ``[date_time]\\n{speaker}: {text}\\n...``.
    Returns (text, title) pairs.
    """
    with LOCOMO_PATH.open() as f:
        raw = json.load(f)
    items: list[tuple[str, str]] = []
    for entry in raw:
        sample_id = entry["sample_id"]
        conv = entry["conversation"]
        i = 1
        while f"session_{i}" in conv:
            dt = conv.get(f"session_{i}_date_time", "")
            turns = conv[f"session_{i}"]
            lines = [f"{t['speaker']}: {t['text']}" for t in turns]
            text = f"[{dt}]\n" + "\n".join(lines)
            items.append((text, f"{sample_id}::session_{i}"))
            i += 1
    return items


# ----------------------------------------------------------------- LongMemEval

def _build_lme_per_turn() -> list[tuple[str, str]]:
    """LME-oracle subset, one ingestable per turn. Format mirrors
    longmemeval/runner._format_turn ('{role}: {content}').

    LME ``haystack_sessions`` is a list parallel to ``haystack_session_ids``
    (not a dict). Each session is a list of turn dicts with ``role`` and
    ``content``.
    """
    if not LME_PATH.exists():
        return []
    with LME_PATH.open() as f:
        raw = json.load(f)
    items: list[tuple[str, str]] = []
    for q in raw:
        qid = q.get("question_id") or q.get("id") or "q?"
        sessions = q.get("haystack_sessions") or []
        sids = q.get("haystack_session_ids") or [str(i) for i in range(len(sessions))]
        for sid, turns in zip(sids, sessions):
            if not isinstance(turns, list):
                continue
            for i, t in enumerate(turns):
                role = t.get("role", "?")
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                text = f"{role}: {content}"
                items.append((text, f"{qid}::{sid}::{i}"))
    return items


# ----------------------------------------------------------------- analysis

def _summarise(label: str, items: list[tuple[str, str]], clf: RoleClassifier) -> None:
    print(f"\n{'=' * 72}")
    print(f"== {label}")
    print(f"== population: {len(items)} ingestables")
    print("=" * 72)

    if not items:
        print("  (no items -- skipping)")
        return

    rng = random.Random(SEED)
    sample = rng.sample(items, k=min(SAMPLE_SIZE, len(items)))
    texts = [t for t, _ in sample]
    titles = [tt for _, tt in sample]

    char_lens = [len(t) for t in texts]
    print(
        f"  sampled n={len(sample)}, "
        f"text chars min/median/max = "
        f"{min(char_lens)}/{sorted(char_lens)[len(char_lens)//2]}/{max(char_lens)}"
    )

    results = clf.classify_batch(texts)

    counts = Counter(r.role for r in results)
    by_role: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_role[r.role].append(r.confidence)

    print("\n  Role distribution + per-role confidence:")
    for role in clf.roles:
        n = counts.get(role, 0)
        pct = 100 * n / len(results)
        if n > 0:
            confs = by_role[role]
            print(
                f"    {role:25s} {n:>4d}  ({pct:>4.1f}%)   "
                f"conf min/mean/max = {min(confs):.3f}/"
                f"{sum(confs)/len(confs):.3f}/{max(confs):.3f}"
            )
        else:
            print(f"    {role:25s} {n:>4d}  ({pct:>4.1f}%)")

    n_below = sum(1 for r in results if r.confidence < LOW_CONF_THRESHOLD)
    print(
        f"\n  Below uncertain threshold (< {LOW_CONF_THRESHOLD}): "
        f"{n_below}/{len(results)} ({100*n_below/len(results):.1f}%)"
    )

    print("\n  Top-confidence example per predicted role:")
    for role in clf.roles:
        ex = [
            (t, tt, r) for (t, tt), r in zip(zip(texts, titles), results)
            if r.role == role
        ]
        if not ex:
            continue
        ex.sort(key=lambda triple: -triple[2].confidence)
        text, title, r = ex[0]
        snippet = text[:240].replace("\n", " | ")
        print(f"    [{role:25s} c={r.confidence:.3f}] [{title}]")
        print(f"        {snippet}")


def main() -> None:
    print("Loading RoleClassifier.default()...")
    clf = RoleClassifier.default()
    print(f"  embedder: {clf.model_name}")
    print(f"  taxonomy: {clf.roles}")

    print("\nBuilding ingest streams...")
    locomo_items = _build_locomo_per_session()
    print(f"  LoCoMo per-session ingestables: {len(locomo_items)}")
    lme_items = _build_lme_per_turn()
    print(f"  LME per-turn ingestables:        {len(lme_items)}")

    _summarise(
        "LoCoMo per-session (the actual merken ingest unit)",
        locomo_items,
        clf,
    )
    _summarise(
        "LongMemEval (oracle) per-turn (the actual merken ingest unit)",
        lme_items,
        clf,
    )


if __name__ == "__main__":
    main()
