"""Mode A concept smoke test -- Cerebras + vstash.

Proves the midloop pattern end-to-end without training anything.
The Builder is a small Cerebras model (``llama3.1-8b``) simulating
a CHW-edge assistant that hallucinates or under-specifies clinical
claims. The midloop components are LLM-as-judge calls to a bigger
Cerebras model (``qwen-3-235b-a22b-instruct-2507``):

  1. ClaimDetector (LLM call): does the draft contain a verifiable
     medical claim? If yes, what queries should we run against
     memory?
  2. snapvec.search over a vstash seeded with 517 WHO/ICRC
     protocol chunks.
  3. ClaimVerifier (LLM call): does the draft match the retrieved
     guidelines? If contradicts, what do the guidelines say?
  4. Regenerate (Builder call): re-ask the Builder with a system
     whisper injected -- "note per authoritative guidelines, X".

No training. No local model. Runs in ~30 seconds per question on
Cerebras' fabric. If this proves the concept (at least one baseline
hallucination gets corrected via retrieval), we unlock either Mode
C (local small-model detector embedded in the generation loop) or
a proper Mode A A/B eval on a larger set.

Usage:
  python cerebras_midloop.py seed [chunks.jsonl ...]
  python cerebras_midloop.py run  [--question "..." | --all]

The default ``--all`` runs a hardcoded 5-question smoke set chosen
to hit known-answers in the 517-chunk corpus. Output is a side-by-
side report to stdout + a structured JSON log at
``experiments/midloop_concept/medlocal/cerebras_smoke.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Models -- keep the asymmetry small-builder / big-judge to mirror
# the production midloop topology (Builder is the runtime model,
# Judge is the more capable sidecar).
BUILDER = "llama3.1-8b"
JUDGE = "qwen-3-235b-a22b-instruct-2507"
# Token budgets kept tight -- this pipeline is deliberately thrifty.
# Draft caps where a clinical answer comfortably fits; Judge caps
# where the verdict + quoted evidence + optional rewrite fit.
MAX_TOKENS_DRAFT = 300
MAX_TOKENS_JUDGE = 600

DB_PATH = str(Path.home() / ".merken" / "medlocal_concept.db")
PROJECT = "medlocal_concept"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = [
    REPO_ROOT / "experiments/midloop_pilot/scaleup_out_hf_v1bfrozen/protocols.jsonl",
    REPO_ROOT / "experiments/midloop_pilot/scaleup_out/protocols.jsonl",
]

SMOKE_QUESTIONS = [
    (
        "severe_dehydration_child",
        "A 3-year-old child arrives with sunken eyes, skin pinch "
        "returns very slowly, and lethargy. What IV fluids should "
        "I start and at what dose/rate?",
    ),
    (
        "severe_pneumonia_infant",
        "A 2-month-old infant has severe pneumonia (chest indrawing, "
        "oxygen saturation 89%). What antibiotic should I give and "
        "at what dose?",
    ),
    (
        "postpartum_hemorrhage_txa",
        "A woman with postpartum hemorrhage did not respond to "
        "uterotonics. What is the tranexamic acid dose and how "
        "should I administer it?",
    ),
    (
        "cotrimoxazole_hiv_adult",
        "An adult patient was just diagnosed with HIV and has a "
        "CD4 count of 180. Should I start co-trimoxazole "
        "prophylaxis, and if so, when?",
    ),
    (
        "blood_donor_screening",
        "A potential blood donor appears pale, has a persistent "
        "cough, and a tattoo from last month. Should I accept them?",
    ),
]

# Personal-memory smoke set -- questions about Jay's own project history.
# All five have verbatim answers sitting in the engram vstash. A
# Builder model with no access to that memory will either hallucinate
# plausibly or refuse; either case exercises the Mode A grounding
# path. Domain is deliberately non-clinical so we also test the
# Judge prompt's generalisation beyond WHO guidelines.
PERSONAL_SMOKE_QUESTIONS = [
    (
        "write_filter_baseline",
        "In the user's merken project, what version of the "
        "write-filter classifier is the current graduated "
        "baseline, and what was its key metric improvement "
        "over the previous version?",
    ),
    (
        "consolidation_threshold",
        "What cosine similarity threshold does merken's "
        "PeriodicConsolidator use for clustering, and why was "
        "that specific value chosen?",
    ),
    (
        "silt_rule",
        "What is Silt's rule about proposing new algorithms in "
        "the merken / engram project?",
    ),
    (
        "engram_longmemeval_r5",
        "In the engram project's LongMemEval benchmark at "
        "n=500, what R@5 score did engram achieve?",
    ),
    (
        "four_primitives",
        "What are the four decision primitives defined in "
        "merken's CONSTITUTION, and what are the default "
        "implementations for each?",
    ),
]

# --------------------------------------------------------------- cerebras

def _cerebras_client():
    # Lazy import so `--help` works without the SDK.
    from cerebras.cloud.sdk import Cerebras
    return Cerebras()


def cerebras_chat(model: str, messages: list[dict], max_tokens: int) -> tuple[str, float, dict]:
    """Return (text, wall_seconds, usage). Bubbles SDK exceptions up.

    ``usage`` is a dict of prompt_tokens / completion_tokens /
    total_tokens taken from ``resp.usage`` when Cerebras populates
    it. Empty dict when the SDK response omits the field, so the
    caller can sum defensively.
    """
    client = _cerebras_client()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    dt = time.perf_counter() - t0
    usage = {}
    u = getattr(resp, "usage", None)
    if u is not None:
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }
    return (resp.choices[0].message.content or "").strip(), dt, usage


def _extract_json_object(raw: str) -> dict:
    """Tolerant JSON extract -- mirrors case_generator's approach.

    Handles ``{...}`` objects that may be wrapped in markdown fences
    or preceded by prose. Raises ValueError on total failure so the
    caller can fall back rather than crash the pipeline.
    """
    text = raw.strip()
    m = re.match(r"```(?:json)?\s*", text)
    if m:
        text = text[m.end():]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Locate outermost {...}. Using rfind handles cases where the
    # model appends an explanation object after the answer object;
    # we take the first valid-parse attempt.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM response: {raw[:200]!r}")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}; candidate={candidate[:200]!r}") from e


# --------------------------------------------------------------- midloop stages

# Judge prompt template. The ``{domain_frame}`` placeholder is the
# only per-domain variance -- everything below it (JSON schema,
# verdict rules, verbatim-evidence rule) is invariant across domains.
# Keeping one prompt body avoids drift: a fix to the HARD RULES
# applies to both clinical and personal runs.
JUDGE_SYSTEM_TEMPLATE = (
    "{domain_frame}\n\n"
    "Your job in ONE JSON response:\n"
    "  1. decide the verdict of the draft against the excerpts.\n"
    "  2. if contradicts, produce the corrected answer using only "
    "     information grounded in the excerpts.\n\n"
    "Respond ONLY with a JSON object:\n"
    "{{\n"
    '  "has_claim": bool,              // draft contains a verifiable factual claim?\n'
    '  "verdict": "supports"|"contradicts"|"neutral"|"no_claim",\n'
    '  "cited_excerpt_ids": [int, ...], // 0-based indices into excerpts\n'
    '  "quoted_evidence": "...",       // verbatim sentence(s) from the excerpts\n'
    '  "corrected_text": "..."         // required iff verdict=contradicts; empty otherwise\n'
    "}}\n\n"
    "Rules:\n"
    "- supports: some excerpt VERBATIM matches the draft. Fill "
    "cited_excerpt_ids + quoted_evidence. Leave corrected_text empty.\n"
    "- contradicts: some excerpt contradicts the draft. Fill all "
    "fields; corrected_text is the final user-facing answer.\n"
    "- neutral: excerpts do not cover the claim. Leave cited_excerpt_ids "
    "[] and quoted_evidence \"\".\n"
    "- no_claim: draft has no verifiable factual claim.\n\n"
    "HARD RULES (enforce every time):\n"
    "- quoted_evidence MUST be a substring of one excerpt. If you "
    "cannot find verbatim supporting evidence in the excerpts, "
    "verdict = 'neutral'. Do NOT synthesise evidence from your "
    "own general knowledge.\n"
    "- corrected_text MUST only contain claims grounded in "
    "quoted_evidence. Keep it to <= 120 words, numbered points ok.\n"
    "- No prose outside the JSON. No markdown fences."
)

# Builder modes. The default ("auto") sends the user's question
# with no system prompt -- this is the realistic CHW-agent edge
# case where the Builder often refuses or hedges, so Mode A's
# grounding path mostly fires on "I cannot verify"-shaped drafts.
# "confident" mode forces the Builder to answer authoritatively
# with specific claims, exercising the Mode A hallucination-
# correction path instead. Used for Run 3 adversarial smoke
# (2026-04-21) after Run 2 revealed 4/5 personal drafts were
# refusals and only 1/5 was a real confabulation.
BUILDER_MODES = {
    "auto": None,
    "confident": (
        "You are answering a user question. Provide a direct, "
        "confident, specific answer. Do not hedge. Do not say "
        "'I am not sure', 'I cannot verify', 'I do not have "
        "information', or anything similar -- answer with "
        "concrete facts, numbers, names, and steps. If you are "
        "uncertain, commit to your best informed guess as if it "
        "were correct. Keep the answer under 200 words."
    ),
}

DOMAIN_FRAMES = {
    "clinical": (
        "You are the verification + rewriter stage of a retrieval-"
        "grounded assistant. The user asked a clinical question; a "
        "smaller Builder model produced a draft. You also see excerpts "
        "retrieved from an authoritative guideline memory (WHO/ICRC/"
        "MSF)."
    ),
    "personal": (
        "You are the verification + rewriter stage of a retrieval-"
        "grounded assistant. The user asked a question about their "
        "own project history (decisions, benchmark results, "
        "architecture notes); a smaller Builder model that has NO "
        "access to that history produced a draft. You also see "
        "excerpts retrieved from the user's authoritative project "
        "memory. Treat those excerpts as ground truth about the "
        "user's project -- the Builder's draft is almost certainly "
        "a hallucination unless the excerpts confirm it verbatim."
    ),
}

# Default Judge prompt keeps the clinical framing so existing
# callers (and the clinical smoke set) are unchanged. run_smoke
# overrides this per invocation based on --domain.
JUDGE_SYSTEM = JUDGE_SYSTEM_TEMPLATE.format(domain_frame=DOMAIN_FRAMES["clinical"])


def judge_once(
    question: str, draft: str, excerpts: list[tuple[str, str]],
) -> tuple[dict, float, dict]:
    """One Judge call that performs detect + verify + rewrite in
    a single pass. Returns (parsed_json, wall_s, usage_dict).

    ``excerpts`` is a list of ``(source_id, text)`` so the Judge
    can cite sources by their memory tag rather than by ephemeral
    index alone. Indices remain 0-based in cited_excerpt_ids; the
    source_id is carried through into the deterministic annotator.
    """
    if not excerpts:
        # No retrieval -> no verification possible. Short-circuit
        # without a Judge call to save tokens.
        return (
            {
                "has_claim": False,
                "verdict": "no_retrieval",
                "cited_excerpt_ids": [],
                "quoted_evidence": "",
                "corrected_text": "",
            },
            0.0,
            {},
        )
    joined = "\n---\n".join(
        f"[excerpt {i} | source={src}]\n{text}"
        for i, (src, text) in enumerate(excerpts)
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Draft response:\n{draft}\n\n"
        f"Authoritative excerpts:\n{joined}\n\n"
        "Return the JSON object now."
    )
    text, dt, usage = cerebras_chat(
        JUDGE,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        MAX_TOKENS_JUDGE,
    )
    try:
        parsed = _extract_json_object(text)
    except ValueError:
        # Fail-closed: neutral verdict so we render the draft with
        # a "generated, unverified" tag rather than fabricate a
        # correction.
        parsed = {
            "has_claim": False,
            "verdict": "neutral",
            "cited_excerpt_ids": [],
            "quoted_evidence": "",
            "corrected_text": "",
        }
    return parsed, dt, usage


def annotate_deterministic(
    draft: str,
    judgment: dict,
    excerpts: list[tuple[str, str]],
) -> str:
    """Assemble the final user-facing response with provenance
    footer. Zero extra LLM calls.

    - supports: draft + [confirmed: src | quoted...] footer.
    - contradicts: corrected_text + [corrected from draft; src |
      quoted...] footer.
    - neutral / no_claim / no_retrieval: draft + [generated, no
      memory coverage] footer.

    The footer is a single-line marker so the downstream consumer
    can grep for it or strip it. An inline-per-sentence annotation
    would need another LLM call; the per-question footer gives the
    user the source attribution at minimal token cost.
    """
    verdict = judgment.get("verdict", "neutral")
    cited_ids = judgment.get("cited_excerpt_ids") or []
    quoted = (judgment.get("quoted_evidence") or "").strip()

    def _cite_sources() -> str:
        srcs = []
        for i in cited_ids:
            if isinstance(i, int) and 0 <= i < len(excerpts):
                srcs.append(excerpts[i][0])
        return ", ".join(srcs) if srcs else "unknown"

    if verdict == "contradicts":
        body = (judgment.get("corrected_text") or draft).strip()
        quoted_short = (quoted[:200] + "...") if len(quoted) > 200 else quoted
        footer = (
            f"\n\n>> [corrected from prior draft: {_cite_sources()}]"
            f"\n>> quoted evidence: \"{quoted_short}\""
        )
        return body + footer

    if verdict == "supports":
        quoted_short = (quoted[:200] + "...") if len(quoted) > 200 else quoted
        footer = (
            f"\n\n>> [confirmed: {_cite_sources()}]"
            f"\n>> quoted evidence: \"{quoted_short}\""
        )
        return draft.strip() + footer

    if verdict == "no_claim":
        return draft.strip() + "\n\n>> [no verifiable claim]"

    # neutral, no_retrieval, or fail-closed fallback
    return draft.strip() + "\n\n>> [generated, no memory coverage]"


# --------------------------------------------------------------- seed

def seed_vstash(chunk_paths: list[Path]) -> int:
    from vstash import Memory
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    if Path(DB_PATH).exists():
        print(f"NOTE: {DB_PATH} already exists; remove it first if you want a clean seed")
    mem = Memory(db=DB_PATH, project=PROJECT)
    n = 0
    try:
        for path in chunk_paths:
            path = Path(path)
            if not path.exists():
                print(f"skip {path} (missing)")
                continue
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    mem.remember(
                        text=row["text"],
                        title=row["protocol_id"],
                        tags=f'source:authoritative,protocol:{row["protocol_id"]}',
                    )
                    n += 1
                    if n % 50 == 0:
                        print(f"  seeded {n}...")
    finally:
        mem.close()
    print(f"done: seeded {n} chunks into {DB_PATH}")
    return n


# --------------------------------------------------------------- pipeline

def retrieve(
    mem,
    query: str,
    top_k: int = 5,
    *,
    retrieval_mode: str = "hybrid",
) -> list[tuple[str, str]]:
    """Return list of (source_id, text).

    ``retrieval_mode``:
      - ``"hybrid"`` (default): one vstash.search call with the
        stock adaptive RRF weighting. Cheap, ~100ms.
      - ``"dual"``: run the hybrid search AND an fts_only search,
        then merge. Added 2026-04-21 after the Run 3b diagnostic
        showed that vec-dominant hybrid buries exact-keyword
        matches for dense clinical chunks (e.g. the WOMAN-trial
        TXA chunk, fts top-1 but hybrid not in top-10). Each call
        returns up to ``top_k``; merged list is capped at
        ``2 * top_k`` after dedup. Extra cost: one more search
        call (~100-200ms), no LLM spend.
    """
    try:
        if retrieval_mode == "dual":
            hybrid_hits = mem.search(query, top_k=top_k)
            fts_hits = mem.search(query, top_k=top_k, fts_only=True)
            # Interleave so neither mode monopolises the prefix; the
            # Judge sees a balanced candidate pool and is less
            # likely to fixate on whichever mode ranked first.
            hits: list = []
            for pair in zip(hybrid_hits, fts_hits):
                hits.extend(pair)
            remaining = hybrid_hits[len(fts_hits):] + fts_hits[len(hybrid_hits):]
            hits.extend(remaining)
        else:
            hits = mem.search(query, top_k=top_k)
    except Exception as e:
        print(f"    search error for {query[:60]!r}: {e}")
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for h in hits:
        text = getattr(h, "text", None) or getattr(h, "content", None) or ""
        if not text:
            continue
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)
        # vstash SearchResult exposes title/tags; title is the
        # protocol_id we set at seed time. Fall back to a short
        # prefix tag if the structure shifts.
        src_id = (
            getattr(h, "title", None)
            or getattr(h, "document_title", None)
            or getattr(h, "tags", None)
            or "memory"
        )
        out.append((str(src_id), text))
    return out


def run_one(
    mem,
    question: str,
    *,
    builder_mode: str = "auto",
    retrieval_mode: str = "hybrid",
) -> dict:
    """Two-call pipeline: Builder draft + Judge finalize.

    Retrieval runs once with the draft+question as the query.
    Annotation is deterministic (no extra LLM call).
    ``builder_mode`` selects a Builder system prompt from
    ``BUILDER_MODES``; ``"auto"`` sends no system prompt and is
    the realistic production default.
    """
    print(f"\n{'='*72}\nQ: {question}\n{'='*72}")

    # --- stage 1: Builder draft ------------------------------------
    b_sys = BUILDER_MODES.get(builder_mode)
    b_messages: list[dict] = []
    if b_sys is not None:
        b_messages.append({"role": "system", "content": b_sys})
    b_messages.append({"role": "user", "content": question})
    draft, b_dt, b_usage = cerebras_chat(BUILDER, b_messages, MAX_TOKENS_DRAFT)
    print(f"\n[BUILDER | {b_dt:.2f}s | tok={b_usage.get('total_tokens', '?')}]\n{draft}")

    # --- stage 2: retrieve (no LLM) --------------------------------
    # Query uses both the question and the draft so the retrieval
    # surfaces chunks relevant to whatever the Builder actually said,
    # not just to the abstract question.
    query = f"{question}\n{draft[:400]}"
    excerpts = retrieve(mem, query, top_k=5, retrieval_mode=retrieval_mode)
    print(
        f"[RETRIEVE] got {len(excerpts)} excerpts"
        + (f"; top_src={excerpts[0][0]}" if excerpts else "")
    )

    # --- stage 3: Judge single call --------------------------------
    judgment, j_dt, j_usage = judge_once(question, draft, excerpts)
    print(
        f"[JUDGE | {j_dt:.2f}s | tok={j_usage.get('total_tokens', '?')}] "
        f"has_claim={judgment.get('has_claim')} "
        f"verdict={judgment.get('verdict')} "
        f"cited={judgment.get('cited_excerpt_ids')}"
    )
    if judgment.get("verdict") == "contradicts":
        print(f"         corrected_text: {judgment.get('corrected_text', '')[:180]}...")

    # --- stage 4: deterministic annotation (no LLM) ----------------
    final = annotate_deterministic(draft, judgment, excerpts)
    print(f"\n[FINAL]\n{final}")

    total_tokens = (b_usage.get("total_tokens", 0) or 0) + (j_usage.get("total_tokens", 0) or 0)
    # Audit-ready row: every Mode A run IS a labeled training example.
    # The builder / judge identifiers + timestamp let downstream
    # trainers slice by model version and freshness.
    return {
        "audit_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "builder_model": BUILDER,
        "judge_model": JUDGE,
        "question": question,
        "draft": draft,
        "draft_s": b_dt,
        "draft_usage": b_usage,
        "retrieved": [{"source_id": s, "text": t} for s, t in excerpts],
        "judgment": judgment,
        "judge_s": j_dt,
        "judge_usage": j_usage,
        "final": final,
        "total_tokens": total_tokens,
        "verdict": judgment.get("verdict"),
        "fired": judgment.get("verdict") == "contradicts",
    }


def run_smoke(
    questions: list[tuple[str, str]],
    *,
    log_name: str = "cerebras_smoke.jsonl",
    builder_mode: str = "auto",
    retrieval_mode: str = "hybrid",
) -> list[dict]:
    from vstash import Memory
    if not Path(DB_PATH).exists():
        sys.exit(
            f"error: vstash db not found at {DB_PATH}. "
            f"run `python {Path(__file__).name} seed` first."
        )
    mem = Memory(db=DB_PATH, project=PROJECT)
    logs: list[dict] = []
    try:
        for qid, q in questions:
            row = run_one(
                mem, q,
                builder_mode=builder_mode,
                retrieval_mode=retrieval_mode,
            )
            row["qid"] = qid
            row["builder_mode"] = builder_mode
            row["retrieval_mode"] = retrieval_mode
            logs.append(row)
    finally:
        mem.close()

    out = Path(__file__).parent / log_name
    with out.open("w") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    print(f"\nlog: {out}")

    # --- summary table ---------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n = len(logs)
    by_verdict: dict[str, int] = {}
    for r in logs:
        v = r.get("verdict") or "unknown"
        by_verdict[v] = by_verdict.get(v, 0) + 1
    total_tok = sum(r.get("total_tokens", 0) for r in logs)
    avg_tok = total_tok / max(n, 1)
    print(f"  questions:            {n}")
    for v, c in sorted(by_verdict.items()):
        print(f"  verdict={v:12s}   {c}/{n}")
    print(f"  total tokens used:    {total_tok}")
    print(f"  avg tokens/question:  {avg_tok:.0f}")
    for r in logs:
        print(
            f"    - {r['qid']:30s} verdict={r.get('verdict'):12s} "
            f"tok={r.get('total_tokens')}"
        )
    return logs


# --------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_seed = sp.add_parser("seed", help="ingest protocol chunks into vstash")
    sp_seed.add_argument("chunks", nargs="*", type=Path)

    sp_run = sp.add_parser("run", help="run baseline + midloop on smoke questions")
    sp_run.add_argument("--question", type=str, help="single free-form question")
    sp_run.add_argument("--all", action="store_true", help="run the full smoke set")
    sp_run.add_argument(
        "--domain",
        choices=sorted(DOMAIN_FRAMES.keys()),
        default="clinical",
        help=(
            "which Judge framing + default smoke set to use. "
            "'clinical' = WHO/ICRC guidelines (default); "
            "'personal' = user's project memory."
        ),
    )
    sp_run.add_argument(
        "--db",
        type=str,
        default=None,
        help="override the vstash db path (default is the clinical medlocal db)",
    )
    sp_run.add_argument(
        "--project",
        type=str,
        default=None,
        help="override the vstash project filter (default medlocal_concept)",
    )
    sp_run.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="override the output jsonl filename (default cerebras_smoke.jsonl)",
    )
    sp_run.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "dual"],
        default="hybrid",
        help=(
            "Retrieval strategy. 'hybrid' (default) = one vstash "
            "adaptive-RRF search. 'dual' = hybrid + fts_only merged, "
            "for corpora where vec-dominant ranking buries exact-"
            "keyword matches (e.g. dense clinical guideline chunks)."
        ),
    )
    sp_run.add_argument(
        "--builder-mode",
        choices=sorted(BUILDER_MODES.keys()),
        default="auto",
        help=(
            "Builder system prompt. 'auto' (default) sends no system "
            "prompt and lets the small model hedge/refuse naturally. "
            "'confident' forces authoritative answers -- used for the "
            "adversarial smoke that exercises Mode A's hallucination-"
            "correction path instead of its refusal path."
        ),
    )

    args = ap.parse_args()
    if args.cmd == "seed":
        paths = args.chunks or DEFAULT_CHUNKS
        seed_vstash(paths)
        return 0

    if args.cmd == "run":
        # Resolve per-domain defaults so a plain --domain personal
        # run picks the right DB, project, Judge prompt, and log.
        global DB_PATH, PROJECT, JUDGE_SYSTEM
        if args.domain == "personal":
            default_db = str(Path.home() / ".vstash" / "memory.db")
            default_project = "engram"
            default_log = "cerebras_personal_smoke.jsonl"
            default_questions = PERSONAL_SMOKE_QUESTIONS
        else:
            default_db = DB_PATH
            default_project = PROJECT
            default_log = "cerebras_smoke.jsonl"
            default_questions = SMOKE_QUESTIONS
        DB_PATH = args.db or default_db
        PROJECT = args.project or default_project
        JUDGE_SYSTEM = JUDGE_SYSTEM_TEMPLATE.format(
            domain_frame=DOMAIN_FRAMES[args.domain]
        )
        log_name = args.log_name or default_log

        if args.question:
            qs = [("ad_hoc", args.question)]
        else:
            qs = default_questions

        # Non-default knobs get suffixed logs so they do not
        # clobber a baseline-mode log for the same domain.
        if args.log_name is None:
            parts = []
            if args.builder_mode != "auto":
                parts.append(args.builder_mode)
            if args.retrieval_mode != "hybrid":
                parts.append(args.retrieval_mode)
            if parts:
                stem = Path(log_name).stem
                suffix = Path(log_name).suffix or ".jsonl"
                log_name = f"{stem}_{'_'.join(parts)}{suffix}"

        print(
            f"[config] domain={args.domain} builder_mode={args.builder_mode} "
            f"retrieval_mode={args.retrieval_mode} "
            f"db={DB_PATH} project={PROJECT} log={log_name}"
        )
        run_smoke(
            qs,
            log_name=log_name,
            builder_mode=args.builder_mode,
            retrieval_mode=args.retrieval_mode,
        )
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
