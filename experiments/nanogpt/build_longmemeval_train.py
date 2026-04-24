"""Build nanoGPT v8 training corpus from LongMemEval labels.

Inputs:
- `experiments/nanogpt/longmemeval_positive_labels.jsonl` -- the
  gpt-oss-120b-refined per-turn labels of turns in answer sessions.
  Each row has label in {"yes", "no", "parse_failure", "error"}.
- LongMemEval dataset (via dataset.load_longmemeval).

Outputs:
- `data/merken_bpe_v8_longmemeval/{train.jsonl, val.jsonl}` with
  rows `{text, label}` where label in {"DECISION", "NOISE"}.
- Question-level 80/20 split -- no question's turns in both.

Class assignment:
- "yes" labels (confirmed needle-carriers in answer sessions) -> DECISION
- "no" labels (chit-chat inside answer sessions) -> NOISE
- Turns in distractor sessions (not labeled by gpt-oss) -> NOISE
- "parse_failure" / "error" -> skipped (not used in training)

Rationale for the "no" -> NOISE assignment: Jay's third-way idea.
The classifier sees not just "answer-topic vs distractor-topic" but
"carries specific fact vs same-topic but generic". This is the finer
distinction the filter needs for top-1 precision on RAG.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402


HELD_OUT_SEEDS = [42, 43, 44]
HELD_OUT_N = 30

_SPECIAL_TOKENS = re.compile(r"<\|[a-z_]+\|>")


def _format_turn(role: str, content: str) -> str:
    text = f"{role}: {content}"
    return _SPECIAL_TOKENS.sub("", text).strip()


def collect_holdout() -> list:
    convs = load_longmemeval("longmemeval_s")
    used = set()
    for s in HELD_OUT_SEEDS:
        rnd = random.Random(s)
        for c in rnd.sample(convs, HELD_OUT_N):
            used.add(c.question_id)
    return [c for c in convs if c.question_id not in used]


def load_positive_labels(path: Path) -> dict:
    """Return {(qid, sid, idx): label} for rows with usable labels.

    parse_failure / error rows are dropped (treated as if unlabeled,
    meaning those specific turns are skipped rather than assigned a
    noisy label).
    """
    out: dict = {}
    for line in path.open():
        r = json.loads(line)
        label = r.get("label")
        if label not in ("yes", "no"):
            continue
        out[(r["qid"], r["sid"], r["idx"])] = label
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("experiments/nanogpt/longmemeval_positive_labels.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/merken_bpe_v8_longmemeval"),
    )
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--include-unlabeled-positives",
        action="store_true",
        help=(
            "If set, answer-session turns not found in the labels "
            "JSONL are treated as DECISION (fallback to session-level). "
            "Default: skip them so only gpt-oss-verified turns land."
        ),
    )
    parser.add_argument(
        "--oversample-positive", type=int, default=10,
        help=(
            "Duplicate each DECISION example N times in train (default 10). "
            "Val is never oversampled (keeps honest recall signal). "
            "Needed because raw DEC:NOI ratio is ~1:177 after gpt-oss "
            "refinement; vanilla training collapses to 'always NOISE'. "
            "Set to 1 to disable. Ignored when --balanced-ratio is set."
        ),
    )
    parser.add_argument(
        "--balanced-ratio", type=int, default=None,
        help=(
            "If set, downsample train NOISE to N * DECISION count. "
            "Val is NEVER rebalanced. Overrides --oversample-positive "
            "(forces it to 1). Matches v7's shape (1:5.5 DEC:NOI with "
            "single-instance examples) more faithfully than 10x oversample "
            "over 162k imbalanced negatives. 2026-04-24 v8a first training "
            "collapsed to 'always NOISE' with 1:17.7 ratio + 10x oversample; "
            "this mode is the conservative retry path."
        ),
    )
    args = parser.parse_args()

    holdout = collect_holdout()
    print(f"[pool] {len(holdout)} holdout questions", flush=True)

    pos_labels = load_positive_labels(args.labels)
    n_yes = sum(1 for v in pos_labels.values() if v == "yes")
    n_no = sum(1 for v in pos_labels.values() if v == "no")
    print(f"[labels] loaded {len(pos_labels)} rows  yes={n_yes}  no={n_no}", flush=True)

    # Question-level split BEFORE collecting turns to avoid leakage.
    rnd = random.Random(args.seed)
    rnd.shuffle(holdout)
    n_val = int(len(holdout) * args.val_frac)
    val_qs = {c.question_id for c in holdout[:n_val]}
    train_qs = {c.question_id for c in holdout[n_val:]}
    print(f"[split] train={len(train_qs)} val={len(val_qs)}", flush=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    skipped_unlabeled = 0

    for c in holdout:
        target = train_rows if c.question_id in train_qs else val_rows
        ans_set = set(c.answer_session_ids)
        for sid, turns in c.haystack_sessions.items():
            is_answer_session = sid in ans_set
            for i, t in enumerate(turns):
                text = _format_turn(t.role, t.content)
                if not text:
                    continue
                if is_answer_session:
                    key = (c.question_id, sid, i)
                    lbl = pos_labels.get(key)
                    if lbl is None:
                        if args.include_unlabeled_positives:
                            label = "DECISION"
                        else:
                            skipped_unlabeled += 1
                            continue
                    elif lbl == "yes":
                        label = "DECISION"
                    else:  # "no"
                        label = "NOISE"
                else:
                    label = "NOISE"
                target.append({"text": text, "label": label})

    def _stats(rows, name):
        d = sum(1 for r in rows if r["label"] == "DECISION")
        n = sum(1 for r in rows if r["label"] == "NOISE")
        print(f"  {name}: total={len(rows)}  DECISION={d}  NOISE={n}  "
              f"dec_frac={d/max(1,len(rows))*100:.1f}%", flush=True)

    print("[stats before rebalancing]", flush=True)
    _stats(train_rows, "train")
    _stats(val_rows, "val  ")
    print(f"  skipped (unlabeled answer-session turns): {skipped_unlabeled}", flush=True)

    # Rebalancing priority: --balanced-ratio wins over --oversample-positive.
    if args.balanced_ratio is not None:
        dec_rows = [r for r in train_rows if r["label"] == "DECISION"]
        noi_rows = [r for r in train_rows if r["label"] == "NOISE"]
        target_noi = min(len(noi_rows), len(dec_rows) * args.balanced_ratio)
        rnd_bal = random.Random(args.seed + 1)
        rnd_bal.shuffle(noi_rows)
        noi_rows = noi_rows[:target_noi]
        rnd_bal.shuffle(dec_rows)  # shuffle for interleave
        train_rows = dec_rows + noi_rows
        rnd_bal.shuffle(train_rows)
        print(f"\n[balancing] DEC:NOI target ratio 1:{args.balanced_ratio} -> "
              f"DEC={len(dec_rows)} NOI={len(noi_rows)} (downsampled from "
              f"{len(noi_rows) if target_noi == len(noi_rows) else target_noi})",
              flush=True)
        print("[stats after balancing]", flush=True)
        _stats(train_rows, "train")
    elif args.oversample_positive > 1:
        dec_rows = [r for r in train_rows if r["label"] == "DECISION"]
        extra = dec_rows * (args.oversample_positive - 1)
        train_rows = train_rows + extra
        print(f"\n[oversampling] DECISION rows duplicated "
              f"{args.oversample_positive}x (extra={len(extra)})", flush=True)
        print("[stats after oversampling]", flush=True)
        _stats(train_rows, "train")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    with train_path.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with val_path.open("w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[out] train -> {train_path}", flush=True)
    print(f"[out] val   -> {val_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
