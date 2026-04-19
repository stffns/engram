"""Recover the 2 protocols that failed during the scale-up case generation.

Background: in the 77-protocol scale-up (commit 06e49f5), 2 protocols
hit Gemini JSON malformation -- the model emitted a JSON array that
became unparseable past the first ~500 chars. The pipeline failed-soft
and skipped them, ending with 375 cases instead of 385.

Fix: use Gemini's structured-output mode (response_mime_type +
response_schema=list[Case]) so the JSON is GUARANTEED valid by the
SDK rather than parsed best-effort from free text.

This script:
  1. Re-runs case generation for cholera-who and dengue-who using
     structured output.
  2. Generates lmstudio responses for those new cases.
  3. Aligns + filters at the calibrated threshold 0.85.
  4. APPENDS the new rows to scaleup_out/{cases,responses,aligned}.jsonl
     instead of overwriting.

If structured output also produces a usable run, the same fix should
land in `merken/training/case_generator.py` as a `default_gemini_client`
factory analogous to `default_anthropic_client` (and the Anthropic
client should grow an equivalent strict-JSON path). Filed as
follow-up; not in scope for this script.

Usage:
  python -m experiments.midloop_pilot.recover_failed
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel

# Default protocol root mirrors run_pilot.py: Jay's local MedLocal
# checkout. Override via $MIDLOOP_PROTOCOL_ROOT env var so the
# script is portable across machines / CI.
DEFAULT_PROTOCOL_ROOT = Path.home() / "Desktop/Personal/Projects/medlocal/data/core/protocols"
PROTOCOL_ROOT = Path(
    os.environ.get("MIDLOOP_PROTOCOL_ROOT", str(DEFAULT_PROTOCOL_ROOT))
)
OUT_DIR = Path(__file__).resolve().parent / "scaleup_out"

FAILED_PROTOCOLS = ["cholera-who.md", "dengue-who.md"]
N_CASES_PER_PROTOCOL = 5
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_S = 60  # per-call hard timeout (mirror run_pilot.gemini_client)
LMSTUDIO_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "gemma-4-e4b-it-mlx"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
THRESHOLD = 0.85


class _Case(BaseModel):
    """Pydantic model for the structured-output schema."""
    prompt: str
    truth: str


def gemini_structured_client(api_key: str):
    """Build a function: (system, user) -> list[dict] with guaranteed JSON shape.

    Unlike the free-text gemini_client in run_pilot.py, this requests
    response_mime_type=application/json + response_schema=list[Case]
    so the SDK enforces the structure. No more JSON parse failures
    from unescaped quotes or runaway prose.

    Wrapped in a per-call concurrent.futures timeout (mirror of
    run_pilot.gemini_client) so a hung Gemini call cannot block the
    whole batch. We previously hit a 23-min SSL_read hang during the
    scale-up before adding this protection.
    """
    import concurrent.futures

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise SystemExit(
            "google-genai not installed. Run `pip install google-genai`. "
            f"[{e}]"
        ) from e

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[_Case],
    )
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _do_call(system: str, user: str) -> list[dict]:
        full = f"{system}\n\n{user}"
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full,
            config=config,
        )
        # parsed gives a list of _Case Pydantic instances when the
        # SDK was able to materialize the schema; otherwise text.
        parsed = resp.parsed
        if parsed is None:
            text = (resp.text or "").strip()
            if not text:
                raise ValueError(
                    "Gemini returned an empty response (no parsed schema, "
                    "no text). Likely a transient API issue; retry."
                )
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Gemini returned non-JSON text after schema "
                    f"parsing failed: {text[:200]!r}"
                ) from e
        return [{"prompt": c.prompt, "truth": c.truth} for c in parsed]

    def _fn(system: str, user: str) -> list[dict]:
        future = pool.submit(_do_call, system, user)
        try:
            return future.result(timeout=GEMINI_TIMEOUT_S)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise TimeoutError(
                f"Gemini structured call exceeded {GEMINI_TIMEOUT_S}s timeout"
            ) from e

    return _fn


def lmstudio_client():
    """Same as run_pilot.lmstudio_client: Session-pooled + safe choices access."""
    try:
        import requests
    except ImportError as e:
        raise SystemExit(
            f"requests not installed. Run `pip install requests`. [{e}]"
        ) from e

    SYSTEM = (
        "You are a community health worker assistant in a low-resource "
        "clinical setting. Respond with ONLY the recommended clinical "
        "action in 1-3 short sentences. Do NOT use markdown headers, "
        "bullet lists, tables, code blocks, or extended explanations. "
        "Do NOT add disclaimers, caveats, or signatures. Be direct, "
        "specific (drug names + doses + durations), and brief."
    )
    session = requests.Session()

    def _fn(prompt: str) -> str:
        resp = session.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={
                "model": LMSTUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"lmstudio returned no choices: {str(data)[:200]}")
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()
    return _fn


_CASE_GEN_SYSTEM = (
    "You generate realistic clinical scenarios for training a memory-"
    "intervention model. Given an authoritative protocol clause, you "
    "produce N scenarios where a clinician would need exactly this "
    "knowledge.\n\n"
    "Each scenario must have:\n"
    "- prompt: a realistic clinician question or situation, "
    "1-3 sentences. Concrete details (age, weight, signs).\n"
    "- truth: the correct answer DERIVED STRICTLY from the protocol "
    "clause. Do not add information not present in the clause."
)

_CASE_GEN_USER_TEMPLATE = (
    "Protocol id: {protocol_id}\n"
    "Protocol clause:\n```\n{protocol_text}\n```\n\n"
    "Generate {n} distinct clinical scenarios. Vary patient details "
    "(age, weight, presentation) but keep the protocol-derived "
    "answer correct in every case."
)


def generate_cases_for(
    protocol_id: str, text: str, gen_fn, source_path: Path,
) -> list[dict]:
    """Format N cases from one protocol clause. Skip malformed rows."""
    from datetime import date

    user = _CASE_GEN_USER_TEMPLATE.format(
        protocol_id=protocol_id,
        protocol_text=text,
        n=N_CASES_PER_PROTOCOL,
    )
    rows = gen_fn(_CASE_GEN_SYSTEM, user)
    out: list[dict] = []
    for i, row in enumerate(rows[:N_CASES_PER_PROTOCOL]):
        if not isinstance(row, dict):
            continue
        prompt = row.get("prompt")
        truth = row.get("truth")
        if not prompt or not truth:
            print(
                f"    skip case {i}: missing prompt or truth",
                file=sys.stderr,
            )
            continue
        out.append({
            "case_id": f"{protocol_id}__case_{i:03d}",
            "prompt": str(prompt).strip(),
            "truth": str(truth).strip(),
            "metadata": {
                "protocol_id": protocol_id,
                "source_file": str(source_path),
                "source": "medlocal_authoritative",
                # source_dir tracks the actual parent directory the
                # .md was loaded from, not a hardcoded "protocols"
                # (matches scale-up's behavior when reading from
                # both protocols/ and other dirs).
                "source_dir": source_path.parent.name,
                "ingested_at_pilot": date.today().isoformat(),
                "recovered_via_structured_output": True,
            },
        })
    return out


def align_cases(cases_with_responses: list[dict]) -> list[dict]:
    from merken.training.midloop_dataset import (
        Aligner,
        Case,
        default_embed_fn,
    )
    aligner = Aligner(
        embed_fn=default_embed_fn(model_name=EMBED_MODEL),
        similarity_threshold=THRESHOLD,
    )
    out = []
    for row in cases_with_responses:
        case = Case(
            truth=row["truth"],
            model_response=row["model_response"],
            prompt=row["prompt"],
            case_id=row["case_id"],
            metadata=row.get("metadata", {}),
        )
        result = aligner.process(case)
        interventions_out = [
            {
                "model_token_start": r.model_start,
                "model_token_end": r.model_end,
                "truth_token_start": r.truth_start,
                "truth_token_end": r.truth_end,
                "truth_text": r.truth_text,
                "model_text": r.model_text,
                "cosine_sim": r.cosine_sim,
            }
            for r in result.interventions
        ]
        drops_out = [
            {
                "truth_text": r.truth_text,
                "model_text": r.model_text,
                "cosine_sim": r.cosine_sim,
            }
            for r in result.semantic_drops
        ]
        out.append({
            "case_id": case.case_id,
            "prompt": case.prompt,
            "truth": case.truth,
            "model_response": case.model_response,
            "metadata": case.metadata,
            "n_divergent": len(result.divergent_regions),
            "n_dropped_semantic": len(result.semantic_drops),
            "n_interventions": len(result.interventions),
            "interventions": interventions_out,
            "semantic_drops": drops_out,
        })
    return out


def main() -> int:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY required", file=sys.stderr)
        return 2

    if not OUT_DIR.exists():
        print(f"ERROR: {OUT_DIR} does not exist; run the scale-up first",
              file=sys.stderr)
        return 2

    gen_fn = gemini_structured_client(api_key)
    resp_fn = lmstudio_client()

    new_cases: list[dict] = []
    for fname in FAILED_PROTOCOLS:
        path = PROTOCOL_ROOT / fname
        if not path.exists():
            print(f"  SKIP {fname}: not found", file=sys.stderr)
            continue
        protocol_id = path.stem
        text = path.read_text(encoding="utf-8")
        print(f"  generating cases for {protocol_id}...", file=sys.stderr)
        t0 = time.time()
        try:
            cases = generate_cases_for(protocol_id, text, gen_fn, path)
        except Exception as e:
            print(f"  STILL FAILED {protocol_id}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        print(f"    {len(cases)} cases in {time.time()-t0:.1f}s",
              file=sys.stderr)
        new_cases.extend(cases)

    if not new_cases:
        print("\nNo cases recovered. Exiting.", file=sys.stderr)
        return 1

    print(f"\nGenerating responses for {len(new_cases)} cases via lmstudio...",
          file=sys.stderr)
    t0 = time.time()
    new_responses = []
    for case in new_cases:
        try:
            response = resp_fn(case["prompt"])
        except Exception as e:
            print(f"  ERR response {case['case_id']}: {e}", file=sys.stderr)
            continue
        new_responses.append({**case, "model_response": response})
    print(f"  {len(new_responses)} responses in {time.time()-t0:.1f}s",
          file=sys.stderr)

    print(f"\nAligning at threshold {THRESHOLD}...", file=sys.stderr)
    aligned = align_cases(new_responses)
    n_div = sum(r["n_divergent"] for r in aligned)
    n_drop = sum(r["n_dropped_semantic"] for r in aligned)
    n_int = sum(r["n_interventions"] for r in aligned)
    print(
        f"  divergent={n_div}, drops={n_drop} "
        f"({100*n_drop/max(n_div,1):.1f}%), interventions={n_int}",
        file=sys.stderr,
    )

    # Append to existing artifacts. Each line is one JSON record.
    cases_path = OUT_DIR / "cases.jsonl"
    responses_path = OUT_DIR / "responses.jsonl"
    aligned_path = OUT_DIR / "aligned.jsonl"

    with cases_path.open("a") as f:
        for c in new_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with responses_path.open("a") as f:
        for r in new_responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with aligned_path.open("a") as f:
        for r in aligned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"\n=== recovery complete ===\n"
        f"  protocols recovered: {len({c['metadata']['protocol_id'] for c in new_cases})}\n"
        f"  cases appended:      {len(new_cases)}\n"
        f"  responses appended:  {len(new_responses)}\n"
        f"  aligned appended:    {len(aligned)}\n"
        f"  intervention labels: +{n_int}\n"
        f"  output:              {aligned_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
