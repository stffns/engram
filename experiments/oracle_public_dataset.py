"""Oracle a sample of assistant text from a public HuggingFace dataset.

Motivation: all our labeled data (1026 shadow_skip + 500 agree_write)
comes from Jay's own Claude Code transcripts. N=1 user. To answer the
"multi-user / multi-domain" generalization question, we need assistant
text from a different distribution. Public instruction-tuning datasets
won't be a perfect match (they're IT-clean, not noisy-agent) but they
give us diversity across tasks and styles at negligible cost.

Default: `LDJnr/Capybara`, a 16k multi-turn conversation dataset. Each
conversation has a `source` tag (domain) and a list of
`{input, output}` turns. We extract the `output` field (assistant
text), split long outputs into paragraphs >=50 chars, sample N items
with reservoir sampling for uniform distribution across all sources,
and oracle each with the same strict prompt used for Jay's labels.

Other datasets can be swapped via --dataset / --split / --text-path.

Output: data/merken_labels_<dataset_slug>.jsonl, idempotent by hash.

Expected class distribution: IT data is substance-heavy, so likely
>70% DECISION. That's a finding, not a bug -- it tells us the
distribution gap between curated-IT and noisy-agent content.
"""

from __future__ import annotations

import torch  # noqa: F401 (Mistake #10 if --filter-pd-range used)

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset
from google import genai

from merken.labeling import _parse_backend_response


STRICT_PROMPT = (
    "You are labeling memory events for an agent's write-filter training set.\n"
    "The filter decides which assistant messages to keep in long-term memory.\n\n"
    "Read the event below and classify it as exactly one of:\n\n"
    "  DECISION  -- a lasting commitment, conclusion, analysis result,\n"
    "               fact discovery, reference material, or explained\n"
    "               reasoning worth keeping. Must contain CONCRETE\n"
    "               content: numbers, file paths, specific findings,\n"
    "               design rationale, or resolution of an open\n"
    "               question.\n\n"
    "  NOISE     -- ephemeral/transient, including:\n"
    "               * transition sentences like 'Let me X', 'Now let me\n"
    "                 Y', 'Let me check Z', 'Now I'll do W' (regardless\n"
    "                 of what X/Y/Z/W refer to).\n"
    "               * task announcements without content: 'Now fix the\n"
    "                 tests', 'Now update Y', 'Let me also add Z'.\n"
    "               * status exclamations: 'Good', 'Perfect!', 'Great!',\n"
    "                 'Excellent!' followed by a next-step.\n"
    "               * intermediate steps that only make sense in the\n"
    "                 context of a longer chain of actions.\n\n"
    "  UNCERTAIN -- genuinely ambiguous; reasonable people disagree.\n\n"
    "Important: the mere fact that the agent is ABOUT TO take an action\n"
    "(e.g. 'Let me check the config file next') is NOT a decision.\n\n"
    "Event:\n"
    "<<<\n"
    "{text}\n"
    ">>>\n\n"
    "Respond on a single line in this exact format (no prose, no code\n"
    "fences):\n"
    "LABEL: <DECISION|NOISE|UNCERTAIN> | CONFIDENCE: <number 0.0-1.0> "
    "| REASON: <one sentence>"
)


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def extract_capybara_chunks(ds, min_len: int = 50, max_len: int = 2000):
    """Yield (source, text) chunks from Capybara assistant outputs.

    Each conversation has a `source` tag and a list of
    {input, output} turns. We split each `output` on double-newline
    (paragraph boundary) and keep chunks in [min_len, max_len].
    """
    for row in ds:
        source = row.get("source") or "unknown"
        conv = row.get("conversation") or []
        if not isinstance(conv, list):
            continue
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            out = turn.get("output") or ""
            if not isinstance(out, str):
                continue
            for chunk in re.split(r"\n\n+", out):
                chunk = chunk.strip()
                if len(chunk) < min_len:
                    continue
                if len(chunk) > max_len:
                    chunk = chunk[:max_len]
                yield source, chunk


def extract_slimorca_chunks(ds, min_len: int = 50, max_len: int = 2000):
    """Yield (source, text) chunks from SlimOrca 'gpt' turns.

    SlimOrca rows have `conversations`: list of turns with `from` in
    {system, human, gpt} and `value`. We only want gpt outputs,
    split by paragraph.
    """
    for row in ds:
        conv = row.get("conversations") or []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            if turn.get("from") != "gpt":
                continue
            out = turn.get("value") or ""
            if not isinstance(out, str):
                continue
            for chunk in re.split(r"\n\n+", out):
                chunk = chunk.strip()
                if len(chunk) < min_len:
                    continue
                if len(chunk) > max_len:
                    chunk = chunk[:max_len]
                yield "SlimOrca", chunk


EXTRACTORS = {
    "LDJnr/Capybara": extract_capybara_chunks,
    "Open-Orca/SlimOrca": extract_slimorca_chunks,
}


def reservoir_sample(iterator, k: int, seed: int):
    rng = random.Random(seed)
    reservoir: list = []
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def build_gemini_client(model: str):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY required")
    return genai.Client(api_key=api_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="LDJnr/Capybara")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--split-limit",
        type=int,
        default=3000,
        help="Cap loaded rows to avoid full dataset downloads (default 3000).",
    )
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path. Default derived from dataset.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--filter-pd-low",
        type=float,
        default=None,
        help="Keep only events where v7 P(D) >= this value. Use with "
        "--filter-pd-high for a bin-specific probe (e.g. 0.0 / 0.3 for "
        "LOW-calibration sampling).",
    )
    parser.add_argument(
        "--filter-pd-high",
        type=float,
        default=None,
        help="Upper bound (exclusive) for v7 P(D) filter.",
    )
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT"),
        help="nanoGPT v7 ckpt path (required when --filter-pd-* is used).",
    )
    parser.add_argument(
        "--meta",
        default=os.environ.get("MERKEN_SHADOW_NANOGPT_META"),
        help="nanoGPT v7 meta path (required when --filter-pd-* is used).",
    )
    args = parser.parse_args()

    filter_by_pd = args.filter_pd_low is not None or args.filter_pd_high is not None
    decider = None
    if filter_by_pd:
        if not args.ckpt or not args.meta:
            raise SystemExit(
                "--filter-pd-* requires --ckpt / --meta (or env vars "
                "MERKEN_SHADOW_NANOGPT_CKPT / _META)."
            )
        from merken.classifiers.nanogpt import NanoGPTWriteDecider
        from merken.policies.types import Event, WriteContext
        print(f"loading v7 for P(D) filter: {args.ckpt}")
        decider = NanoGPTWriteDecider(args.ckpt, args.meta)
        _pd_ctx = WriteContext(project="filter")
        lo = args.filter_pd_low if args.filter_pd_low is not None else 0.0
        hi = args.filter_pd_high if args.filter_pd_high is not None else 1.01

        def pd_of(text: str) -> float | None:
            d = decider.decide(Event(text=text), _pd_ctx)
            m = re.search(r"P\(D\)=([\d.]+)", d.reason or "")
            return float(m.group(1)) if m else None

    if args.dataset not in EXTRACTORS:
        raise SystemExit(
            f"No extractor registered for {args.dataset!r}. "
            f"Known: {list(EXTRACTORS)}. Add one in EXTRACTORS."
        )
    extractor = EXTRACTORS[args.dataset]

    slug = args.dataset.replace("/", "_").lower()
    out_path = Path(args.out) if args.out else Path(f"data/merken_labels_{slug}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.dataset}[{args.split}][:{args.split_limit}]...")
    ds = load_dataset(
        args.dataset,
        split=f"{args.split}[:{args.split_limit}]",
    )
    print(f"loaded {len(ds)} rows; extracting chunks...")

    raw_stream = extractor(ds)
    if filter_by_pd:
        def filtered():
            kept = 0
            seen = 0
            for src, text in raw_stream:
                seen += 1
                p = pd_of(text)
                if p is None:
                    continue
                if lo <= p < hi:
                    kept += 1
                    yield src, text, p
                if seen % 500 == 0:
                    print(f"  scanned {seen}, kept {kept} in P(D) [{lo}, {hi})")
        # For filtered stream we also want to keep P(D) in the record.
        stream_list = list(filtered())
        rng = random.Random(args.seed)
        rng.shuffle(stream_list)
        sample_raw = stream_list[: args.n]
        sample = [(s, t) for s, t, _ in sample_raw]
        pd_lookup = {content_hash(t): p for s, t, p in sample_raw}
    else:
        sample = reservoir_sample(raw_stream, args.n, args.seed)
        pd_lookup = {}
    print(f"final sample size: {len(sample)}")

    # source distribution in the sample
    from collections import Counter
    source_dist = Counter(s for s, _ in sample)
    print("source distribution (top 10):")
    for src, cnt in source_dist.most_common(10):
        print(f"  {src:<40} {cnt:>4}")

    if args.dry_run:
        print("\n--dry-run: skipping labeling. First 3 samples:")
        for s, t in sample[:3]:
            print(f"  [{s}] {t[:150]!r}...")
        return 0

    client = build_gemini_client(args.model)

    # idempotency
    done_hashes: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                if r.get("hash"):
                    done_hashes.add(r["hash"])
            except Exception:
                continue

    counts = {"labeled": 0, "skipped_done": 0, "errors": 0,
              "DECISION": 0, "NOISE": 0, "UNCERTAIN": 0}

    with out_path.open("a", encoding="utf-8") as out:
        for i, (source, text) in enumerate(sample, 1):
            h = content_hash(text)
            if h in done_hashes:
                counts["skipped_done"] += 1
                continue
            try:
                prompt = STRICT_PROMPT.format(text=text[:3000])
                resp = client.models.generate_content(
                    model=args.model, contents=prompt
                )
                label = _parse_backend_response(resp.text or "", args.model)
            except Exception as e:
                counts["errors"] += 1
                print(f"  [{i}/{len(sample)}] ERROR: {type(e).__name__}: {e}")
                continue

            rec = {
                "hash": h,
                "dataset": args.dataset,
                "dataset_source": source,
                "text": text,
                "label": label.decision,
                "confidence": label.confidence,
                "rationale": label.rationale,
                "backend": label.backend,
                "source": "oracle_public_dataset",
            }
            if pd_lookup.get(h) is not None:
                rec["v7_pd"] = pd_lookup[h]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["labeled"] += 1
            counts[label.decision] = counts.get(label.decision, 0) + 1

            if i % 25 == 0:
                print(
                    f"  [{i}/{len(sample)}] labeled={counts['labeled']} "
                    f"(D:{counts['DECISION']} N:{counts['NOISE']} "
                    f"U:{counts['UNCERTAIN']}) err={counts['errors']}"
                )

    print("\n=== totals ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
