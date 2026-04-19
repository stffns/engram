"""Phase 1: build midloop training dataset from (truth, model_response) pairs.

The midloop spec (`notes/midloop-spec.md`) defines NanoGPTMidloopDecider
as a small classifier trained to decide where in a generation stream
to intervene. Phase 1 builds the labeled dataset for that training:

  1. Take pairs of (truth, model_response) where truth is the
     authoritative answer (derived from a Type B protocol) and
     model_response is what a smaller model produced.
  2. Token-level edit-distance alignment marks divergent regions.
  3. snapvec semantic-similarity filter drops divergent regions
     that are SEMANTICALLY equivalent (paraphrases, unit reformulations,
     "50 mg/kg/dia" vs "50 miligramos por kilo diarios") so the
     midloop labels are NOT polluted by surface-level reformatting.
  4. Remaining divergences become "intervene at this token index"
     labels.

Two API surfaces:

- ``Aligner.align_only(case)`` -- pure edit-distance, no embeddings.
  Useful for unit tests and the case where you do not want network /
  model inference.
- ``Aligner.process(case)`` -- align + batch-embed + cosine filter.
  Per Jay's guidance (2026-04-19): one ``embed_texts`` call per case
  (clear lifecycle, no global memory pressure). 1k cases x 20-50
  divergences => 20k-50k subsequence embeddings -- in batch this is
  fast; one-by-one ``embed_query`` would be ~10x slower.

The aligner does NOT call the large or small model itself. Those are
external steps (Anthropic SDK / HF transformers) producing the
JSONL of cases that this module consumes. CLI orchestration lives in
``merken.training.midloop_dataset_cli`` (Phase 1c).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

# Optional dep: only loaded when a real embedder is requested. The
# pure align_only path does not need it.
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Case:
    """One (truth, model_response) pair to label.

    ``case_id`` is for downstream traceability; if absent at write
    time, the CLI assigns a stable hash. ``prompt`` is preserved on
    output so downstream model training has the original input.

    ``metadata`` is opaque -- carry through whatever the case
    generator produced (protocol_id, severity_level, source_clause,
    etc).
    """
    truth: str
    model_response: str
    prompt: str = ""
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Region:
    """A divergent region between truth and model token sequences.

    Token indices are 0-based, end-exclusive. ``cosine_sim`` is None
    for regions that have not been embedded yet (``align_only`` path)
    and a float in [-1, 1] after the semantic-filter step. A None
    sim for a region returned by ``process`` indicates either side
    of the region was empty (insertion / deletion); semantic
    equivalence does not apply.
    """
    truth_start: int
    truth_end: int
    truth_text: str
    model_start: int
    model_end: int
    model_text: str
    cosine_sim: float | None = None


@dataclass
class AlignmentResult:
    case: Case
    divergent_regions: list[Region]   # all regions found by diff
    semantic_drops: list[Region]      # filtered out as equivalent
    interventions: list[Region]       # remaining: midloop training labels

    @property
    def intervene_at_token_indices(self) -> list[int]:
        """Convenience flat view: model-side start indices of intervention regions."""
        return [r.model_start for r in self.interventions]


# ----------------------------------------------------------------- tokenization

_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    """Word-level whitespace split.

    Phase 1 deliberately does NOT use BPE: the midloop training
    happens at word level until the actual midloop model is built
    (which will define its own tokenizer per the v7 pattern). Keeping
    this in lockstep with whatever the trained model expects is a
    Phase 3 concern; the dataset itself is tokenizer-agnostic at
    word granularity.
    """
    return _TOKEN_RE.findall(text)


# ----------------------------------------------------------------- alignment

def diff_regions(truth: list[str], model: list[str]) -> list[Region]:
    """Return regions where truth and model tokens differ.

    Uses ``difflib.SequenceMatcher`` opcodes -- O((n+m) * D) where D
    is the edit distance, fast for the typical clinical-response
    sizes (truth ~50-300 tokens, model similar). Each non-equal
    opcode (`replace` / `insert` / `delete`) becomes one Region.
    """
    sm = SequenceMatcher(a=truth, b=model, autojunk=False)
    regions: list[Region] = []
    for op, t_lo, t_hi, m_lo, m_hi in sm.get_opcodes():
        if op == "equal":
            continue
        regions.append(Region(
            truth_start=t_lo,
            truth_end=t_hi,
            truth_text=" ".join(truth[t_lo:t_hi]),
            model_start=m_lo,
            model_end=m_hi,
            model_text=" ".join(model[m_lo:m_hi]),
        ))
    return regions


# ----------------------------------------------------------------- semantic filter

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _filter_by_semantic_similarity(
    regions: list[Region],
    embed_fn: EmbedFn,
    threshold: float,
) -> tuple[list[Region], list[Region]]:
    """Batch-embed all regions then split into (interventions, drops).

    Skips regions where either side is empty (pure insert / delete --
    no semantic equivalence possible, always counts as intervention).
    For substantive regions, the cosine sim between truth and model
    text decides: above threshold => semantically equivalent, drop;
    below => keep as intervention.

    Per the spec, ``embed_fn`` is called ONCE per case with ALL
    subsequences batched. This is ~10x faster than per-subseq calls
    and matches what `vstash.embed.embed_texts` is optimized for.
    """
    if not regions:
        return [], []

    # Substantive regions need both sides non-empty.
    substantive_idx = [
        i for i, r in enumerate(regions)
        if r.truth_text and r.model_text
    ]
    if not substantive_idx:
        # All regions are pure insert / delete -- nothing to filter.
        return list(regions), []

    # Batch: truth subseqs first, then model subseqs in same order.
    truth_subseqs = [regions[i].truth_text for i in substantive_idx]
    model_subseqs = [regions[i].model_text for i in substantive_idx]
    batch = truth_subseqs + model_subseqs
    embeddings = embed_fn(batch)
    if len(embeddings) != len(batch):
        raise RuntimeError(
            f"embed_fn returned {len(embeddings)} embeddings for "
            f"{len(batch)} inputs; expected one-to-one"
        )
    truth_embs = embeddings[:len(substantive_idx)]
    model_embs = embeddings[len(substantive_idx):]

    interventions: list[Region] = []
    drops: list[Region] = []
    sim_by_idx: dict[int, float] = {}
    for k, region_idx in enumerate(substantive_idx):
        sim = _cosine(truth_embs[k], model_embs[k])
        sim_by_idx[region_idx] = sim
        regions[region_idx].cosine_sim = sim

    # Now split, preserving original order, including non-substantive
    # regions as interventions.
    for i, region in enumerate(regions):
        if i not in sim_by_idx:
            interventions.append(region)
            continue
        if sim_by_idx[i] >= threshold:
            drops.append(region)
        else:
            interventions.append(region)
    return interventions, drops


# ----------------------------------------------------------------- aligner

class Aligner:
    """Token-level diff + (optional) snapvec semantic-equivalence filter.

    Construct once per CLI run; reuse across all cases. Stateless
    aside from configured threshold and embedder reference.

    Two paths:

      - ``align_only(case)``: pure edit-distance, no embeddings.
        Use for unit tests, debugging, or runs where the embedder
        is not available.
      - ``process(case)``: align + batch-embed + filter. Requires
        an ``embed_fn`` (vstash.embed.embed_texts is the canonical
        choice).
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        *,
        similarity_threshold: float = 0.90,
    ) -> None:
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold

    def align_only(self, case: Case) -> AlignmentResult:
        truth = tokenize(case.truth)
        model = tokenize(case.model_response)
        regions = diff_regions(truth, model)
        return AlignmentResult(
            case=case,
            divergent_regions=regions,
            semantic_drops=[],
            interventions=list(regions),
        )

    def process(self, case: Case) -> AlignmentResult:
        if self._embed_fn is None:
            raise RuntimeError(
                "Aligner.process requires an embed_fn at construction; "
                "use align_only() for the no-embedding path"
            )
        truth = tokenize(case.truth)
        model = tokenize(case.model_response)
        regions = diff_regions(truth, model)
        interventions, drops = _filter_by_semantic_similarity(
            regions, self._embed_fn, self._threshold,
        )
        return AlignmentResult(
            case=case,
            divergent_regions=regions,
            semantic_drops=drops,
            interventions=interventions,
        )


# ----------------------------------------------------------------- defaults

def default_embed_fn(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbedFn:
    """Build an embed_fn backed by vstash.embed.

    Lazy import so the no-embedding path does not require fastembed
    / vstash to be installed for unit tests.
    """
    from vstash.embed import embed_texts

    def _fn(texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, model_name=model_name)

    return _fn
