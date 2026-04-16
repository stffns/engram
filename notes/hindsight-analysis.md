# Hindsight Analysis -- Relevance to merken

*Captured 2026-04-16. Paper: arXiv 2512.12818 (Dec 2025).*

## What Hindsight is

Agent memory architecture with 4 logical networks + 2 core components:

### 4 Networks (typed memory storage)
- **World** -- objective facts about the external environment
- **Bank** -- agent's own experiences (first person)
- **Opinion** -- subjective judgments with confidence scores, updated with new evidence
- **Observation** -- preference-neutral entity summaries synthesized from facts

### 2 Components
- **TEMPR** (Temporal Entity Memory Priming Retrieval) -- retain + recall
  - 4-way parallel retrieval: semantic, BM25, graph, temporal
  - RRF + cross-encoder reranking
- **CARA** (Coherent Adaptive Reasoning Agents) -- reflect
  - Updates opinions as new evidence arrives
  - Synthesizes observations from underlying facts

### Numbers
- LongMemEval: 91.4% (vs 83.6% prior best)
- LoCoMo: 89.61% (vs 75.78% prior best open system)

## What merken has / doesn't have

| Capability | Hindsight | merken today | Gap |
|-----------|-----------|-------------|-----|
| Typed storage | 4 networks | 2 layers (episodic/semantic) | Need 4 layers |
| Semantic retrieval | Yes | Yes (vstash) | -- |
| BM25 retrieval | Yes | Yes (vstash FTS5) | -- |
| Graph retrieval | Yes | No | Need graph signal |
| Temporal retrieval | Yes | Partial (rerank_by_recency) | Need temporal as first-class |
| Cross-encoder rerank | Yes | No | Need reranker |
| RRF fusion | Yes | Yes (vstash adaptive RRF) | -- |
| Reflect/update | CARA | brief_v1 (manual) | Need automated reflection |
| Opinion tracking | Confidence scores | No | Need opinion primitive |

## How merken could adopt the pattern

### Phase 1: 4 layers in vstash (low cost)
vstash already supports `layer=` parameter. Add:
- `layer="world"` -- objective facts
- `layer="experience"` -- agent actions (currently "episodic")
- `layer="opinion"` -- beliefs with confidence (new)
- `layer="entity"` -- entity summaries (currently "semantic" briefs)

The nanoGPT classifier evolves from signal/noise to world/experience/opinion/entity
classification at ingestion time. Same model, richer labels.

### Phase 2: temporal as retrieval signal (medium cost)
vstash has timestamps on every doc. Add temporal scoring to the RRF:
- Recent events boost for trajectory queries
- Temporal decay for stale facts
- This is already sketched in merken's `rerank_by_recency()` but not
  integrated into vstash's core RRF

### Phase 3: graph signal (higher cost)
Entity co-occurrence graph from the episodic layer:
- "Redis" appears in cache decisions AND Redis incidents
- Graph knows they share an entity but differ in role
- This is the entity-aware clustering from Section 12 of the paper

### Phase 4: cross-encoder reranking (medium cost)
A small cross-encoder (e.g., MiniLM-based) that reranks the top-K
after RRF fusion. vstash's retrain pipeline with hard negatives
could train this from recall disagreements.

## Key insight from Hindsight for merken

Hindsight's "opinion" network with confidence scores that update
is exactly what brief_v1's "Current state" field does manually.
The difference: Hindsight automates the update (CARA), merken
requires explicit consolidate() calls.

Automating this = brief regeneration triggered by event ingestion
that contradicts an existing brief's current state. The nanoGPT
classifier could detect this: "new event about cache topic AND
existing brief says Caffeine AND new event mentions Redis" ->
trigger brief regeneration for cache topic.

## What merken has that Hindsight doesn't

- **Glass-box audit trail** -- every decision logged and reversible
- **brief_v1 dual-channel** -- reasoning artifacts separated from retrieval
  (Hindsight's 4 networks all participate in the same retrieval pool)
- **Self-supervised training loop** -- operational data -> model training
  (Hindsight uses fixed models, no self-improvement)

The dual-channel insight from our paper is complementary to Hindsight's
4-network design. You could have 4 Hindsight-style networks in the
retrieval channel AND a brief layer in the reasoning channel.

## For Paper 2

Hindsight is a key related work for the self-supervised paper:
- Their 4-network classification is what our nanoGPT could learn
- Their CARA reflect is what our brief_v1 does
- The gap: they don't train from their own data (fixed architecture)
- Our contribution: the self-improving loop that Hindsight lacks
