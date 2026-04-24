"""Step 0 of the v5-ft bge contrastive fine-tune (notes/bge-small-lme-ft-plan.md).

Emit contrastive triples for LongMemEval-domain fine-tuning of
bge-small-en-v1.5. Each triple is:

    {"qid": str,
     "question": str,
     "positive": str,           # turn content (a "yes" label from gpt-oss-120b)
     "hard_negatives": [str]}   # turns in the SAME haystack, not relevant to q

Positives come from `experiments/nanogpt/longmemeval_positive_labels.jsonl`
(1128 "yes" labels across 414 holdout questions, produced by
label_longmemeval_positives.py -- decontaminated by construction against
seeds 42/43/44 N=30).

Hard negatives come from:
  1. "no"-labeled turns in the SAME answer_session (hardest -- same
     session, same topic, oracle said they don't answer the question).
  2. All turns in distractor_session_ids (not in answer_session_ids --
     topically adjacent haystack noise).

We sample `--n-hard` hard negatives per positive, preferring same-session
"no" turns first (higher signal), then falling back to distractor-session
turns. Selection is deterministic with `--seed`.

Also writes a markdown preview to
`experiments/retrieval/bge_lme_ft/train_preview.md` for manual inspection
per the plan's Step 0 requirement ("Inspect 20 triples manually for
quality").

Safety check: the script asserts that no qid from seeds 42/43/44 N=30
appears in the emitted triples -- guards invariant #3 of the v5-ft plan.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import Conversation, load_longmemeval  # noqa: E402

HELD_OUT_SEEDS = [42, 43, 44]
HELD_OUT_N = 30


def _test_qids(convs: list[Conversation]) -> set[str]:
    """Exactly reproduce the held-out test set that must never enter training.

    Mirrors `collect_positives()` in label_longmemeval_positives.py so the
    two stay in sync.
    """
    used: set[str] = set()
    for s in HELD_OUT_SEEDS:
        rnd = random.Random(s)
        for c in rnd.sample(convs, HELD_OUT_N):
            used.add(c.question_id)
    return used


def _persist_and_verify_test_qids(path: Path, qids: set[str]) -> None:
    """Pin the held-out qid set to disk.

    First call writes the set (committed to the repo). Every subsequent run
    re-derives and compares: if the live `load_longmemeval` order drifts
    for any reason (dataset update, cache variance), the mismatch aborts
    the run instead of silently leaking test qids into training.
    """
    sorted_qids = sorted(qids)
    if path.exists():
        on_disk = sorted(json.loads(path.read_text()))
        if on_disk != sorted_qids:
            only_disk = set(on_disk) - set(sorted_qids)
            only_live = set(sorted_qids) - set(on_disk)
            raise RuntimeError(
                f"Held-out qid drift detected!\n"
                f"  on-disk: {len(on_disk)} qids at {path}\n"
                f"  live:    {len(sorted_qids)} qids from current load_longmemeval\n"
                f"  only on disk: {sorted(only_disk)[:5]}{'...' if len(only_disk) > 5 else ''}\n"
                f"  only live:    {sorted(only_live)[:5]}{'...' if len(only_live) > 5 else ''}\n"
                f"Resolve before training: either the dataset changed or "
                f"the held-out computation drifted."
            )
        print(f"[holdout] verified {len(on_disk)} qids match {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted_qids, indent=2))
    print(f"[holdout] pinned {len(sorted_qids)} qids to {path}", flush=True)


def _load_positives(path: Path) -> list[dict]:
    """Load positive labels, keeping only rows where label == 'yes'."""
    out = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") == "yes":
                out.append(r)
    return out


def _build_neg_pool(
    conv: Conversation,
    pos_labels_for_qid: list[dict],
) -> tuple[list[str], list[str]]:
    """Return (same_session_negatives, distractor_session_negatives) for one qid.

    same_session_negatives: turn contents from answer_session_ids whose
    (qid, sid, idx) is NOT a "yes" positive. These are mostly "no"-labeled
    (oracle said the turn doesn't answer the q) plus any unlabeled chit-chat
    in the answer session. Hardest negatives for contrastive training.

    distractor_session_negatives: every turn content from sessions NOT in
    answer_session_ids -- full haystack noise. Easier but larger pool.
    """
    ans_ids = set(conv.answer_session_ids)
    pos_keys = {(p["sid"], p["idx"]) for p in pos_labels_for_qid}

    same_session: list[str] = []
    distractors: list[str] = []
    for sid, turns in conv.haystack_sessions.items():
        if sid in ans_ids:
            for i, t in enumerate(turns):
                if (sid, i) in pos_keys:
                    continue
                if not t.content.strip():
                    continue
                same_session.append(t.content)
        else:
            for t in turns:
                if t.content.strip():
                    distractors.append(t.content)
    return same_session, distractors


def build_triples(
    convs: list[Conversation],
    positives: list[dict],
    n_hard: int,
    seed: int,
    test_qids: set[str],
) -> list[dict]:
    conv_by_qid: dict[str, Conversation] = {c.question_id: c for c in convs}
    pos_by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in positives:
        pos_by_qid[p["qid"]].append(p)

    triples: list[dict] = []
    rng = random.Random(seed)
    n_skipped_missing_conv = 0
    n_skipped_few_negs = 0

    for qid, pos_list in pos_by_qid.items():
        assert qid not in test_qids, f"{qid} is in held-out test set -- leak!"
        conv = conv_by_qid.get(qid)
        if conv is None:
            n_skipped_missing_conv += len(pos_list)
            continue
        same_neg, dist_neg = _build_neg_pool(conv, pos_list)

        for p in pos_list:
            # pop() after shuffle() is a deterministic random sample.
            picked: list[str] = []
            ss_pool = same_neg.copy()
            rng.shuffle(ss_pool)
            dist_pool = dist_neg.copy()
            rng.shuffle(dist_pool)

            # Prefer same-session negatives first -- oracle labeled them
            # "no" but they share the session's topic, so they're the
            # hardest examples for contrastive training.
            while len(picked) < n_hard and ss_pool:
                picked.append(ss_pool.pop())
            while len(picked) < n_hard and dist_pool:
                picked.append(dist_pool.pop())

            if len(picked) < n_hard:
                n_skipped_few_negs += 1
                continue

            triples.append({
                "qid": qid,
                "question": p["question"],
                "positive": p["content"],
                "hard_negatives": picked,
                "positive_session": p["sid"],
            })

    if n_skipped_missing_conv:
        print(f"[skip] {n_skipped_missing_conv} positives had no matching conv",
              flush=True)
    if n_skipped_few_negs:
        print(f"[skip] {n_skipped_few_negs} positives had < {n_hard} hard negs",
              flush=True)
    return triples


def write_preview(triples: list[dict], path: Path, n: int = 20, seed: int = 0) -> None:
    rng = random.Random(seed)
    sample = rng.sample(triples, min(n, len(triples)))
    lines = [
        f"# Contrastive triples preview ({len(sample)} of {len(triples)})",
        "",
        "Manual inspection per notes/bge-small-lme-ft-plan.md Step 0.",
        "Look for: (a) positive actually answers the question, (b) hard",
        "negatives are topically close but don't answer it.",
        "",
    ]
    for i, t in enumerate(sample, 1):
        lines.append(f"## {i}. qid={t['qid']}")
        lines.append("")
        lines.append(f"**Q**: {t['question']}")
        lines.append("")
        lines.append(f"**Positive**: {t['positive'][:400]}")
        lines.append("")
        for j, n in enumerate(t["hard_negatives"], 1):
            lines.append(f"**Neg {j}**: {n[:300]}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=ENGRAM / "experiments" / "nanogpt" / "longmemeval_positive_labels.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ENGRAM / "data" / "bge_lme_ft" / "train.jsonl",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=ENGRAM / "experiments" / "retrieval" / "bge_lme_ft" / "train_preview.md",
    )
    parser.add_argument("--n-hard", type=int, default=4,
                        help="Hard negatives per positive (default 4).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-qids-path",
        type=Path,
        default=ENGRAM / "experiments" / "retrieval" / "bge_lme_ft" / "test_qids.json",
        help="Committed pin of the 86 held-out qids. First run writes it; "
             "subsequent runs verify match and abort on drift.",
    )
    args = parser.parse_args()

    print(f"[load] labels from {args.labels}", flush=True)
    positives = _load_positives(args.labels)
    print(f"[load] {len(positives)} 'yes'-labeled positive turns", flush=True)

    print("[load] LongMemEval (longmemeval_s)", flush=True)
    convs = load_longmemeval("longmemeval_s", download=True)
    print(f"[load] {len(convs)} total questions", flush=True)

    test_qids = _test_qids(convs)
    print(f"[holdout] {len(test_qids)} qids reserved (seeds {HELD_OUT_SEEDS} N={HELD_OUT_N})",
          flush=True)
    _persist_and_verify_test_qids(args.test_qids_path, test_qids)

    pos_qids = {p["qid"] for p in positives}
    leak = pos_qids & test_qids
    assert not leak, f"label file contains {len(leak)} test qids -- abort"

    triples = build_triples(convs, positives, args.n_hard, args.seed, test_qids)
    print(f"[build] {len(triples)} triples emitted "
          f"({args.n_hard} hard-neg each)", flush=True)

    # Coverage sanity
    covered_qids = {t["qid"] for t in triples}
    print(f"[coverage] {len(covered_qids)} / {len(pos_qids)} positive qids "
          f"carried into triples", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[write] {args.out}", flush=True)

    write_preview(triples, args.preview, n=20, seed=args.seed)
    print(f"[write] preview -> {args.preview}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
