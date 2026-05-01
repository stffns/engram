"""Phase 2 distillation -- Step 3 zero-shot smoke (LM Studio).

Step 1+2 confirmed the teacher (gpt-oss-120b on Cerebras) populates
reasoning at 100% across all 11 shapes. Step 3 measures the floor:
how does a *zero-shot* local student handle the same prompts on
the same retrieval, before any SFT?

The student is local by design (the program's whole point is
$1/Q on Cerebras -> $0/Q on local hardware). This script targets
LM Studio's OpenAI-compatible endpoint at
``http://localhost:1234/v1`` and supports any reasoning model
loaded there (the LM Studio convention surfaces the trace as
``message.reasoning_content``, not Cerebras's ``message.reasoning``).

Sample design (deterministic seed=42, mirrors step2): 2 calls per
shape x 11 shapes = 22 calls per model. Smaller than step2's 5 per
shape because Step 3 is a smoke -- we want directional signal on
"can the student even produce a coherent answer + trace?", not
statistical power. That comes in Step 4 (smoke distillation) and
Step 6 (canonical eval gate).

No oracle grading. Each row carries the question, GT, content, and
reasoning_content for hand inspection. The teacher's answers from
step2's rows.jsonl can be diffed offline if direct comparison is
needed.

Usage::

    python -m experiments.phase2_distillation.step3_zero_shot_smoke \\
        --model qwen/qwen3.5-9b

Outputs go to ``experiments/phase2_distillation/runs/step3_<model_slug>_<ts>/``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import vstash
from vstash.chat import _build_messages  # probe-only: see CLAUDE.md

from experiments.retrieval.locomo.runner import load_locomo
from experiments.retrieval.longmemeval.dataset import load_longmemeval
from experiments.phase2_distillation.step2_stratified_capture import (
    LME_TYPES,
    LOCOMO_CATEGORIES,
    LOCOMO_CONV_ID,
    _stratified_locomo_picks,
    _stratified_lme_picks,
    _ingest_locomo_per_session,
    _ingest_lme_per_turn,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCOMO_DATA = REPO_ROOT / "experiments" / "retrieval" / "locomo" / "data" / "locomo10.json"
LME_CACHE = REPO_ROOT / "experiments" / "retrieval" / "longmemeval" / ".cache"

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
TOP_K = 8
VEC_WEIGHT = 0.5
FTS_WEIGHT = 0.5
MAX_TOKENS = 4096
TEMPERATURE = 0.0  # deterministic for smoke

PER_SHAPE = 2  # smaller than step2's 5 -- smoke, not power

SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(name: str) -> str:
    return SLUG_RE.sub("_", name).strip("_")


def _take_per_shape(picks: list, key: str, n: int, valid_keys: list) -> list:
    """Take the first ``n`` per shape from a step2 stratified picks
    list. Empirically verified to match step2's first-n per shape
    (see step3 commit message). Note: this is NOT equivalent to
    re-running ``_stratified_*_picks`` with ``n=N`` -- step2 uses
    ``rng.sample(pool, K)`` whose draw differs by K. This helper
    instead slices step2's K=5 draw to take the first n per shape,
    which preserves teacher-vs-student question alignment.
    """
    by_shape: dict[str, list] = {}
    for p in picks:
        shape = getattr(p, key)
        by_shape.setdefault(shape, []).append(p)
    out = []
    for s in valid_keys:
        out.extend(by_shape.get(s, [])[:n])
    return out


def _call_lm_studio(
    model: str,
    messages: list[dict],
    temperature: float = TEMPERATURE,
    repetition_penalty: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> dict:
    """Call LM Studio's OpenAI-compatible endpoint. Returns
    {content, reasoning_content, reasoning_present, wall_s, usage}.

    LM Studio surfaces reasoning models' hidden trace as
    ``message.reasoning_content`` (note: different from Cerebras's
    ``message.reasoning``). Sentinel-guard so non-reasoning models
    degrade to ``reasoning_present=False``.

    Optional sampling tweaks (default off, opt-in via the runner
    CLI flags). Useful when an undertrained SFT student loops
    on greedy decoding -- pass ``--temperature 0.2`` and
    ``--repetition-penalty 1.1`` to break the loop.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }
    if repetition_penalty is not None:
        # LM Studio accepts repetition_penalty as a llama.cpp-style
        # extra param; ignored gracefully on backends that don't
        # support it.
        payload["repetition_penalty"] = repetition_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        LM_STUDIO_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=600) as resp:
            raw_body = resp.read().decode("utf-8")
        data = json.loads(raw_body)
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        return {
            "content": "",
            "reasoning_content": None,
            "reasoning_present": False,
            "wall_s": time.perf_counter() - t0,
            "usage": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    wall = time.perf_counter() - t0
    # Defensive: an API that returns 200 with empty `choices` would
    # IndexError on [0]. Treat as a no-content failure rather than
    # crashing the whole smoke loop.
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    msg = (choice.get("message") if isinstance(choice, dict) else None) or {}
    sentinel = object()
    raw = msg.get("reasoning_content", sentinel)
    reasoning_present = raw is not sentinel
    reasoning_content = raw if reasoning_present else None
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning_content": reasoning_content,
        "reasoning_present": reasoning_present,
        "wall_s": wall,
        "usage": data.get("usage", {}),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
    }


def _run(
    model: str,
    out_dir: Path,
    temperature: float,
    repetition_penalty: float | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
) -> list[dict]:
    rows: list[dict] = []
    rows_path = out_dir / "rows.jsonl"
    fh = rows_path.open("w", encoding="utf-8")

    def _record(row: dict) -> None:
        rows.append(row)
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()

    # ----- LoCoMo ---------------------------------------------- #
    locomo_convs = load_locomo(LOCOMO_DATA)
    locomo_conv = next(c for c in locomo_convs if c.sample_id == LOCOMO_CONV_ID)
    full = _stratified_locomo_picks(locomo_conv)
    locomo_picks = _take_per_shape(full, "category_name", PER_SHAPE, LOCOMO_CATEGORIES)
    print(
        f"[locomo] conv={locomo_conv.sample_id} picks={len(locomo_picks)} "
        f"({len(LOCOMO_CATEGORIES)}cat x {PER_SHAPE})",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="step3_locomo_") as tmpdir:
        db = Path(tmpdir) / "step3.db"
        mem = vstash.Memory(project="step3_locomo", db=db, collection="default")
        n_ing = _ingest_locomo_per_session(mem, locomo_conv)
        print(f"[locomo] ingested {n_ing} sessions", flush=True)

        for i, qa in enumerate(locomo_picks):
            print(
                f"[locomo {i+1:2d}/{len(locomo_picks)}] cat={qa.category_name:12s} "
                f"q={qa.question[:70]!r}",
                flush=True,
            )
            chunks = mem.search(
                qa.question,
                top_k=TOP_K,
                vec_weight=VEC_WEIGHT,
                fts_weight=FTS_WEIGHT,
            )
            messages = _build_messages(qa.question, chunks, history=None)
            res = _call_lm_studio(
                model, messages,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            _record({
                "source": "locomo",
                "model": model,
                "sample_id": locomo_conv.sample_id,
                "category": qa.category,
                "category_name": qa.category_name,
                "question": qa.question,
                "ground_truth": qa.answer,
                "n_chunks_retrieved": len(chunks),
                **res,
            })
        mem.close()

    # ----- LME ------------------------------------------------- #
    lme_convs = load_longmemeval(subset="longmemeval_s", cache_dir=LME_CACHE, download=False)
    full_lme = _stratified_lme_picks(lme_convs)
    lme_picks = _take_per_shape(full_lme, "question_type", PER_SHAPE, LME_TYPES)
    print(
        f"[lme] picks={len(lme_picks)} ({len(LME_TYPES)} types x {PER_SHAPE})",
        flush=True,
    )

    for i, conv in enumerate(lme_picks):
        try:
            with tempfile.TemporaryDirectory(prefix="step3_lme_") as tmpdir:
                db = Path(tmpdir) / "step3.db"
                mem = vstash.Memory(project="step3_lme", db=db, collection="default")
                n_written, n_skipped = _ingest_lme_per_turn(mem, conv)
                print(
                    f"[lme {i+1:2d}/{len(lme_picks)}] qid={conv.question_id} "
                    f"type={conv.question_type:30s} ingested {n_written} "
                    f"turns ({n_skipped} skipped)",
                    flush=True,
                )
                chunks = mem.search(
                    conv.question,
                    top_k=TOP_K,
                    vec_weight=VEC_WEIGHT,
                    fts_weight=FTS_WEIGHT,
                )
                messages = _build_messages(conv.question, chunks, history=None)
                res = _call_lm_studio(
                model, messages,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
                _record({
                    "source": "lme",
                    "model": model,
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": conv.answer,
                    "n_chunks_retrieved": len(chunks),
                    "n_turns_ingested": n_written,
                    "n_turns_skipped": n_skipped,
                    **res,
                })
                mem.close()
        except Exception as exc:  # noqa: BLE001 -- per-conv fail-soft
            print(
                f"[lme {i+1:2d}/{len(lme_picks)}] qid={conv.question_id} "
                f"FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _record({
                "source": "lme",
                "model": model,
                "question_id": conv.question_id,
                "question_type": conv.question_type,
                "question": conv.question,
                "ground_truth": conv.answer,
                "error": f"{type(exc).__name__}: {exc}",
            })

    fh.close()
    return rows


def _summarize(rows: list[dict]) -> dict:
    def _per_shape(rs: list[dict], key: str, prefix: str) -> dict:
        out: dict = {}
        groups: dict[str, list] = {}
        for r in rs:
            shape = r.get(key)
            if shape:
                groups.setdefault(f"{prefix}:{shape}", []).append(r)
        for shape, group in groups.items():
            n = len(group)
            n_with_text = sum(1 for r in group if (r.get("reasoning_content") or "").strip())
            r_lens = [len(r.get("reasoning_content") or "") for r in group]
            c_lens = [len(r.get("content") or "") for r in group]
            walls = [r.get("wall_s") or 0 for r in group]
            n_errors = sum(1 for r in group if r.get("error"))
            # Surface "reasoned but never reached an answer" as a
            # separate signal -- a model that exhausts max_tokens on
            # reasoning_content is technically reasoning at 100% but
            # ships zero answers, which is not what the smoke is
            # checking for.
            n_finish_length = sum(1 for r in group if r.get("finish_reason") == "length")
            n_empty_content = sum(1 for r in group if not (r.get("content") or "").strip() and not r.get("error"))
            out[shape] = {
                "n": n,
                "n_reasoning_nonempty": n_with_text,
                "frac_reasoning_nonempty": n_with_text / n if n else 0.0,
                "reasoning_chars_median": int(statistics.median(r_lens)) if r_lens else 0,
                "content_chars_median": int(statistics.median(c_lens)) if c_lens else 0,
                "wall_s_median": round(statistics.median(walls), 1) if walls else 0.0,
                "n_finish_length": n_finish_length,
                "n_empty_content": n_empty_content,
                "n_errors": n_errors,
            }
        return out

    locomo = [r for r in rows if r.get("source") == "locomo"]
    lme = [r for r in rows if r.get("source") == "lme"]
    n = len(rows)
    n_with_text = sum(1 for r in rows if (r.get("reasoning_content") or "").strip())
    return {
        "n_calls": n,
        "n_reasoning_nonempty": n_with_text,
        "frac_reasoning_nonempty": n_with_text / n if n else 0.0,
        "by_locomo_category": _per_shape(locomo, "category_name", "locomo"),
        "by_lme_question_type": _per_shape(lme, "question_type", "lme"),
        "totals_by_source": {"locomo": len(locomo), "lme": len(lme)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="LM Studio model id (see /v1/models)")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE,
                    help=f"Sampling temperature. Default {TEMPERATURE} "
                         "(greedy). Bump to 0.2 for SFT'd students that "
                         "loop on greedy decoding.")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="llama.cpp-style repetition penalty. None = "
                         "off (default). 1.1 is a typical value to "
                         "break greedy repetition loops without "
                         "harming legitimate reasoning structure.")
    ap.add_argument("--frequency-penalty", type=float, default=None,
                    help="OpenAI-style frequency penalty. Mutually "
                         "compatible with repetition_penalty.")
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--tag", default="",
                    help="Suffix appended to the run dir name (e.g. "
                         "'temp0.2-reppen1.1') so multiple sweeps on "
                         "the same model don't collide.")
    args = ap.parse_args()

    stamp = _now_stamp()
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = REPO_ROOT / "experiments" / "phase2_distillation" / "runs" / f"step3_{_slug(args.model)}{suffix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step3] writing to {out_dir}", flush=True)
    print(
        f"[step3] config: model={args.model} top_k={TOP_K} "
        f"max_tokens={MAX_TOKENS} temp={args.temperature} "
        f"repetition_penalty={args.repetition_penalty} "
        f"frequency_penalty={args.frequency_penalty} "
        f"per_shape={PER_SHAPE}",
        flush=True,
    )
    print(
        f"[step3] N total = {len(LOCOMO_CATEGORIES) * PER_SHAPE} locomo + "
        f"{len(LME_TYPES) * PER_SHAPE} lme = "
        f"{(len(LOCOMO_CATEGORIES) + len(LME_TYPES)) * PER_SHAPE}",
        flush=True,
    )

    t0 = time.perf_counter()
    rows = _run(
        args.model, out_dir,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )
    wall = time.perf_counter() - t0
    print(f"[step3] {len(rows)} rows in {wall/60:.1f} min", flush=True)

    summary = _summarize(rows)
    summary["wall_minutes"] = wall / 60
    summary["model"] = args.model
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
