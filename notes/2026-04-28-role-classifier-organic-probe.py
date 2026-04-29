"""One-shot probe: classify a sample of real vstash content with the V2
SCR RoleClassifier. The classifier was validated on authored synthetic
events (3-seed canonical eval, macro-F1 0.929). This probe asks the
"does it survive contact with organic noise?" question.

BGE-small-en-v1.5 is documented as English-only. The first run of this
probe (no language filter) classified 86% of Spanish content as
"investigation" with confidence < 0.05 -- expected behavior for a model
asked to compare Spanish text to English prototypes. The filter below
restricts the sample to mostly-English docs so we can assess the
classifier under the conditions it was actually trained for.

Run: ``python -m notes.2026-04-28-role-classifier-organic-probe``

Read-only against ``~/.vstash/memory.db``. Does not write anywhere.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

import vstash

from merken.role_classifier import RoleClassifier

SAMPLE_SIZE = 50
MIN_CHARS = 100  # skip tiny / smoke-test artifacts
SEED = 42
LOW_CONF_THRESHOLD = 0.05  # PR #43 docstring: "< 0.05 is uncertain"

# Heuristic English filter: BGE-small-en-v1.5 is English-only, so we
# only want to give it text it was trained for. The combination of
# (1) low accented-char ratio, (2) more English stopword tokens than
# Spanish stopword tokens by a comfortable margin, picks up Jay's
# Claude-Code transcripts, paper notes, and code-discussion content
# while filtering out mixed / Spanish-dominant docs.
_ENGLISH_STOPWORDS = frozenset({
    "the", "of", "and", "a", "in", "to", "is", "was", "for", "on",
    "with", "as", "this", "that", "by", "are", "be", "from", "or",
    "it", "an", "but", "at", "we", "have", "has", "if", "not", "they",
    "you", "i", "can", "will", "would", "could", "should", "do",
    "does", "all", "more", "than", "into", "when", "where", "what",
    "which", "who", "why", "how", "so", "now", "then", "before",
    "after", "during", "while", "until", "since", "because", "though",
    "although", "however", "therefore", "instead", "without",
})
_SPANISH_STOPWORDS = frozenset({
    "el", "la", "los", "las", "de", "del", "que", "en", "un", "una",
    "unos", "unas", "y", "o", "pero", "es", "ser", "se", "no", "si",
    "para", "por", "con", "sin", "como", "pues", "ya", "nos", "le",
    "les", "te", "me", "su", "sus", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "lo", "al", "porque", "cuando",
    "donde", "quien", "tambien", "solo", "tan", "muy", "mas", "esto",
    "hay", "fue", "han", "ha", "esta", "estan", "todo", "todos",
    "todas", "toda", "siempre", "nunca",
})
_ACCENTED_RE = re.compile(r"[À-ſ]")  # Latin-1 Supplement + Latin Ext-A
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _is_mostly_english(text: str) -> bool:
    accented = len(_ACCENTED_RE.findall(text))
    if accented / max(len(text), 1) > 0.01:
        return False
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if len(tokens) < 20:  # too short for a reliable classification
        return False
    en = sum(1 for t in tokens if t in _ENGLISH_STOPWORDS)
    es = sum(1 for t in tokens if t in _SPANISH_STOPWORDS)
    if en < 5:  # not enough English stopwords to be confidently English
        return False
    return en >= max(2 * es, 5)


def main() -> None:
    rng = random.Random(SEED)

    print(f"Loading vstash from default location...")
    mem = vstash.Memory()
    all_docs = list(mem.list())
    print(f"  {len(all_docs)} total docs.")

    # Drop smoke-test artifacts and tiny docs.
    candidates = [
        d
        for d in all_docs
        if d.char_count is not None
        and d.char_count >= MIN_CHARS
        and not d.path.startswith("/tmp/")
        and not d.path.startswith("/private/tmp/")
    ]
    print(f"  {len(candidates)} candidates after size + path filter ({MIN_CHARS}+ chars, no /tmp).")

    # Filter to mostly-English content (the embedder is BGE-small-en-v1.5).
    english_candidates: list = []
    for d in candidates:
        try:
            chunks = mem.get_document_chunks(d.path)
        except Exception:
            continue
        if not chunks:
            continue
        text = chunks[0].strip()
        if _is_mostly_english(text):
            english_candidates.append((d, text))
    print(
        f"  {len(english_candidates)} candidates after English-only filter "
        f"({100*len(english_candidates)/max(len(candidates),1):.1f}% of size-filtered)."
    )

    if len(english_candidates) < SAMPLE_SIZE:
        print(
            f"  WARNING: fewer than {SAMPLE_SIZE} English candidates; "
            f"using all {len(english_candidates)}."
        )

    sample = rng.sample(english_candidates, k=min(SAMPLE_SIZE, len(english_candidates)))
    print(f"  sampling {len(sample)} for classification.")
    print()

    print("Loading RoleClassifier.default()...")
    clf = RoleClassifier.default()
    print(f"  embedder: {clf.model_name}")
    print(f"  taxonomy: {clf.roles}")
    print()

    # Classify on the first chunk text we already pulled during the
    # English filter step (avoids re-reading from sqlite).
    rows: list[dict] = []
    print(f"Classifying {len(sample)} docs...")
    for d, text in sample:
        if len(text) < MIN_CHARS:
            continue
        result = clf.classify(text)
        rows.append(
            {
                "path": d.path,
                "project": d.project,
                "chars": d.char_count,
                "role": result.role,
                "confidence": result.confidence,
                "text": text,
                "all_sims": result.role_similarities,
            }
        )

    print(f"  classified {len(rows)} successfully.")
    print()
    print("=" * 70)

    # Distribution
    role_counts = Counter(r["role"] for r in rows)
    print("Role distribution:")
    for role, count in role_counts.most_common():
        pct = 100 * count / len(rows) if rows else 0
        print(f"  {role:25s} {count:3d}  ({pct:.1f}%)")
    print()

    # Confidence stats
    by_role: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r["confidence"])
    print("Confidence stats per predicted role:")
    print(f"  {'role':25s} {'n':>4s}  {'min':>6s}  {'mean':>6s}  {'max':>6s}")
    for role in clf.roles:
        confs = by_role.get(role, [])
        if not confs:
            print(f"  {role:25s} {0:>4d}")
            continue
        print(
            f"  {role:25s} {len(confs):>4d}  "
            f"{min(confs):>6.3f}  {sum(confs)/len(confs):>6.3f}  {max(confs):>6.3f}"
        )
    print()

    # Low-confidence cases (likely out-of-taxonomy or borderline)
    low_conf = [r for r in rows if r["confidence"] < LOW_CONF_THRESHOLD]
    print(f"Low-confidence cases (< {LOW_CONF_THRESHOLD}): {len(low_conf)}")
    for r in low_conf[:5]:
        snippet = r["text"][:200].replace("\n", " ")
        print(f"  [{r['role']:25s} c={r['confidence']:.3f}] {snippet}...")
    print()

    # Sample 2 examples per role
    print("Sample classifications per role:")
    for role in clf.roles:
        examples = [r for r in rows if r["role"] == role]
        if not examples:
            print(f"\n  -- {role}: none")
            continue
        print(f"\n  -- {role} (n={len(examples)}):")
        # Take highest-confidence and median-confidence
        examples.sort(key=lambda r: -r["confidence"])
        for label, ex in [("top", examples[0]), ("mid", examples[len(examples) // 2])]:
            snippet = ex["text"][:240].replace("\n", " ")
            print(f"    [{label} conf={ex['confidence']:.3f}] {snippet}...")


if __name__ == "__main__":
    main()
