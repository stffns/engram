# LoCoMo End-to-End QA Results

## Overview

End-to-end evaluation on the original LoCoMo dataset (Maharana et al., 2024)
using answer accuracy with a judge model, not just retrieval recall.

Pipeline: ingest dialogue sessions -> optionally consolidate/forget ->
retrieve K=10 -> generate answer (Gemini 2.0 Flash) -> judge vs gold
(Gemini 2.0 Flash).

## Run: 2026-04-16, commit TBD

- **Dataset:** locomo10.json (10 conversations, 2 evaluated)
- **Categories:** temporal (cat 2) + adversarial (cat 5)
- **n = 134** QA pairs across 2 conversations
- **Ingestion:** session-level (each session = 1 document, ~20 turns)
- **Judge model:** gemini-2.0-flash
- **top_k:** 10
- **Conversations:** conv-26 (19 sessions, 84 QA), conv-30 (19 sessions, 50 QA)

### Results

| Config | adversarial | temporal | overall | 95% CI |
|--------|------------|----------|---------|--------|
| vstash-raw | 0.070 | 0.413 | 0.231 | [0.157, 0.306] |
| merken-recall | 0.099 | 0.349 | 0.216 | [0.149, 0.291] |
| merken-consol | 0.070 | 0.190 | 0.127 | [0.075, 0.187] |
| merken-full | 0.000 | 0.016 | 0.007 | [0.000, 0.022] |

### Operational metrics

| Config | turns ingested | facts created | docs forgotten | avg tokens/query |
|--------|---------------|---------------|----------------|------------------|
| vstash-raw | 19/conv | 0 | 0 | ~5800 |
| merken-recall | 19/conv | 0 | 0 | ~4400 |
| merken-consol | 19/conv | 1 | 0 | ~4200 |
| merken-full | 19/conv | 1 | 19 | ~300 |

### Analysis

**1. merken-recall matches vstash-raw (CIs overlap).**
Glass-box overhead (audit trail, write decider, layered recall) is invisible
in answer accuracy. Delta is -1.5pp, well within CI. This confirms the LMEB
retrieval findings: merken's primitives don't degrade the substrate.

**2. Consolidation hurts (-10.4pp vs vstash-raw).**
Root cause: 19 sessions between the same 2 speakers have high embedding
similarity. Complete-linkage clustering at threshold 0.70 merges all 19 into
a single mega-fact. This mega-fact loses the granularity that retrieval needs
to find specific dialogue turns. Lowering the threshold to 0.55 produces the
same result -- the sessions are structurally too similar.

**3. ForgetConsolidated is catastrophic here (-22.4pp vs vstash-raw).**
After consolidation creates 1 fact from 19 sessions, ForgetConsolidated
tombstones all 19 original sessions. Only 1 document remains for retrieval.
This validates NeverForget as the safe default -- forgetting requires strong
evidence that consolidated facts fully capture the original information.

### Why LoCoMo is the wrong benchmark for consolidation

LoCoMo was designed to test long-term conversational memory retrieval, not
information synthesis. Its properties work against consolidation:

- **No redundancy:** each session introduces new facts. Consolidation needs
  overlapping information to add value by deduplication.
- **Homogeneous structure:** all sessions are casual dialogues between the
  same 2 people, making embedding similarity uniformly high regardless of
  topic differences.
- **Fine-grained answers:** gold answers reference specific turns in specific
  sessions. A consolidated fact that merges sessions loses this specificity.

### What this means for merken

- **NeverForget is correct as default.** No change.
- **Consolidation has a precondition:** content must have meaningful semantic
  clusters (different topics, repeated facts, knowledge updates). Homogeneous
  dialogue does not meet this precondition.
- **Next step:** design or find a benchmark with redundant/evolving content
  where consolidation can demonstrate value -- or confirm it doesn't.

### Comparison to published results

Published LoCoMo results use full-context approaches (entire conversation in
LLM context window), not retrieval-augmented generation:
- A-MAC: F1=0.583 (hybrid rule + LLM, all categories)
- SuperLocalMemory: 70.4% (zero-LLM mode, all categories)
- Letta: ~66% (all categories)

Our retrieval-based approach (K=10 sessions as context) is structurally
disadvantaged vs full-context. The 23% accuracy is not comparable to these
numbers -- different methodology, different categories, different metric.
