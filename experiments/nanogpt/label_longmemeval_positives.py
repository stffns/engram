"""Phase 2b labeler: refine LongMemEval positive turns via gpt-oss-120b.

Session-level positive labels (turn in answer_session_ids) are ~60-70%
noisy (chit-chat inside answer session without the actual needle fact).
This labeler asks gpt-oss-120b a targeted yes/no question per positive
turn to get clean turn-level labels for nanoGPT v8 training.

Negatives (turns in distractor sessions) are NOT relabeled -- the
Step 0 inspection showed they are clean.

Output JSONL schema per row:
  {"qid": str, "sid": str, "idx": int, "role": str,
   "content": str, "question": str,
   "label": "yes" | "no", "reason": str,
   "prompt_tokens": int, "completion_tokens": int, "wall_s": float}

Idempotent: if the output file exists, resumes from the last labeled
(qid, sid, idx) so a killed run can be restarted without double spend.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGRAM))
sys.path.insert(0, str(ENGRAM / "experiments" / "retrieval" / "longmemeval"))

from dataset import load_longmemeval  # noqa: E402


HELD_OUT_SEEDS = [42, 43, 44]
HELD_OUT_N = 30

LABELER_MODEL = "gpt-oss-120b"
# 200 was truncating ~4% of calls (2/50 calibration hit the cap with
# empty content -- model was still generating). 400 gives headroom
# at no extra cost for non-truncated calls (completion_tokens counts
# actually-emitted tokens, not the cap).
MAX_COMPLETION_TOKENS = 400
TEMPERATURE = 0.0
N_WORKERS = 8

PROMPT_TEMPLATE = (
    "You are labeling conversational turns for a memory-filter training "
    "dataset.\n\n"
    "QUESTION (to be answered later using turns like this one):\n"
    "{question}\n\n"
    "TURN (in a session where the question's topic was discussed):\n"
    "[{role}] {content}\n\n"
    "Does this TURN contain a specific fact that directly helps answer "
    "the QUESTION?\n"
    "- \"yes\": the turn explicitly states the answer, or contains a "
    "fact that is a direct input to computing the answer (a name, "
    "number, date, event, or entity named in the question).\n"
    "- \"no\": the turn is in the same session but is generic "
    "assistant intro, chit-chat, topic transition, or discusses a "
    "different fact that is adjacent but not used to answer.\n\n"
    "Respond with JSON only, no prose, no markdown:\n"
    "{{\"verdict\": \"yes\" | \"no\", "
    "\"reason\": \"<one short sentence>\"}}"
)


def _cerebras_client():
    from cerebras.cloud.sdk import Cerebras
    return Cerebras()


def _call(client, prompt: str) -> tuple[str, int, int, float]:
    from cerebras.cloud.sdk import APIStatusError

    backoffs = [2, 4, 8, 16]
    t0 = time.perf_counter()
    for attempt in range(len(backoffs) + 1):
        try:
            resp = client.chat.completions.create(
                model=LABELER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=TEMPERATURE,
            )
            break
        except APIStatusError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = isinstance(status, int) and (status >= 500 or status == 429)
            if not transient or attempt == len(backoffs):
                raise
            time.sleep(backoffs[attempt])
    dt = time.perf_counter() - t0
    content = resp.choices[0].message.content or ""
    u = resp.usage
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    return content, pt, ct, dt


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> tuple[str, str]:
    text = raw.strip()
    # Strip markdown fences if any
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()
    m = _JSON_RE.search(text)
    if not m:
        return "parse_failure", raw[:200]
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "parse_failure", raw[:200]
    verdict = obj.get("verdict", "").strip().lower()
    reason = obj.get("reason", "").strip()
    if verdict not in ("yes", "no"):
        return "parse_failure", f"bad_verdict={verdict!r} raw={raw[:200]}"
    return verdict, reason


def collect_positives() -> list[dict]:
    convs = load_longmemeval("longmemeval_s")
    used = set()
    for s in HELD_OUT_SEEDS:
        rnd = random.Random(s)
        for c in rnd.sample(convs, HELD_OUT_N):
            used.add(c.question_id)
    holdout = [c for c in convs if c.question_id not in used]

    positives: list[dict] = []
    for c in holdout:
        ans_set = set(c.answer_session_ids)
        for sid, turns in c.haystack_sessions.items():
            if sid not in ans_set:
                continue
            for i, t in enumerate(turns):
                positives.append({
                    "qid": c.question_id,
                    "sid": sid,
                    "idx": i,
                    "role": t.role,
                    "content": t.content,
                    "question": c.question,
                })
    return positives


def _row_key(r: dict) -> tuple[str, str, int]:
    return (r["qid"], r["sid"], r["idx"])


def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add((r.get("qid"), r.get("sid"), r.get("idx")))
    return done


def label_one(client, row: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        question=row["question"][:1500],
        role=row["role"],
        content=row["content"][:3000],
    )
    raw, pt, ct, dt = _call(client, prompt)
    verdict, reason = _parse(raw)
    return {
        "qid": row["qid"],
        "sid": row["sid"],
        "idx": row["idx"],
        "role": row["role"],
        "content": row["content"],
        "question": row["question"],
        "label": verdict,
        "reason": reason,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "wall_s": dt,
        "raw": raw[:500] if verdict == "parse_failure" else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/nanogpt/longmemeval_positive_labels.jsonl"),
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N positive turns (for calibration).")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    positives = collect_positives()
    print(f"[pool] total positive turns: {len(positives)}", flush=True)

    done = _load_done(args.out)
    print(f"[resume] {len(done)} rows already labeled; will skip", flush=True)

    todo = [r for r in positives if _row_key(r) not in done]
    if args.n is not None:
        todo = todo[: args.n]
    print(f"[todo] {len(todo)} turns to label this run", flush=True)

    if not todo:
        print("[done] nothing to do", flush=True)
        return 0

    # Cost estimate
    avg_prompt_tokens = 200  # conservative
    avg_completion_tokens = 50
    # gpt-oss-120b rough rates: input $0.25/M, output $1.00/M
    est_cost = len(todo) * (avg_prompt_tokens * 0.25 / 1e6 + avg_completion_tokens * 1.00 / 1e6)
    print(f"[est] ~${est_cost:.3f} at gpt-oss-120b rates", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = _cerebras_client()

    t_all = time.perf_counter()
    n_done = 0
    n_yes = 0
    n_no = 0
    n_parse_fail = 0
    total_pt = 0
    total_ct = 0

    with args.out.open("a") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures_map = {ex.submit(label_one, client, r): r for r in todo}
            for fut in futures_map:
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        **futures_map[fut],
                        "label": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "wall_s": 0.0,
                    }
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()
                n_done += 1
                if row["label"] == "yes":
                    n_yes += 1
                elif row["label"] == "no":
                    n_no += 1
                elif row["label"] == "parse_failure":
                    n_parse_fail += 1
                total_pt += row.get("prompt_tokens", 0)
                total_ct += row.get("completion_tokens", 0)

                if n_done % 100 == 0 or n_done == len(todo):
                    elapsed = time.perf_counter() - t_all
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - n_done) / rate if rate > 0 else float("inf")
                    print(
                        f"[{n_done}/{len(todo)}] yes={n_yes} no={n_no} "
                        f"fail={n_parse_fail}  "
                        f"rate={rate:.1f}/s  eta={eta/60:.1f}min  "
                        f"tok={total_pt+total_ct}",
                        flush=True,
                    )

    total_wall = time.perf_counter() - t_all
    actual_cost = total_pt * 0.25 / 1e6 + total_ct * 1.00 / 1e6
    print("", flush=True)
    print(f"=== done ===", flush=True)
    print(f"labeled: {n_done}  yes={n_yes}  no={n_no}  parse_fail={n_parse_fail}", flush=True)
    print(f"yes fraction: {n_yes / max(1, n_yes + n_no)*100:.1f}%", flush=True)
    print(f"tokens: prompt={total_pt} completion={total_ct}", flush=True)
    print(f"actual cost: ~${actual_cost:.3f}", flush=True)
    print(f"wall: {total_wall:.1f}s", flush=True)
    print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
