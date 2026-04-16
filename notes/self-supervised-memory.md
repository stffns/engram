# Self-Supervised Memory Systems

*Captured 2026-04-16 during consolidation experiments session.*

## Core Insight

An agentic memory system (vstash + merken) generates labeled training
data as a byproduct of normal operation. This data can train specialized
small models that replace generic large models at each stage of the
memory pipeline, creating a self-improving loop.

## The Architecture

```
Event arrives (text from agent session)
         |
    [Stage 1: Classifier]
    nanoGPT ~1M params, runs local, <1ms
    Input:  raw event text
    Output: NOISE | DECISION:topic:version | ENTITY | EVENT
    Trained on: merken remember() write/skip decisions
         |
    [Stage 2: Embedder]
    Custom model ~30M params, runs local
    Input:  classified event text
    Output: dense vector for retrieval
    Trained on: vstash recall() hits/misses as hard negatives
         |
    [Stage 3: Brief Generator]
    Gemma 2B, fine-tuned with LoRA, runs local
    Input:  cluster of events on same topic
    Output: temporal brief with "Current state: X"
    Trained on: merken brief_v1 outputs (gold summaries)
         |
    [Stage 4: Answer Model]
    Gemma 2B (same or different LoRA)
    Input:  briefs + retrieved chunks + query
    Output: factual answer
    Trained on: QA pairs from knowledge_update experiments
```

No model is larger than 2B params. No API calls. Entire stack runs
locally on a MacBook or a single GPU.

## The Self-Supervised Loop

Each component generates training data for the others:

```
                    generates labels for
    Classifier  --------------------------->  Embedder
        |                                        |
        | write/skip decisions          hits/misses as
        | = signal/noise labels         hard negatives
        |                                        |
        v                                        v
    Audit Trail  <------ merken ------->  Recall Results
        |                   |                    |
        |            consolidate()               |
        |                   |                    |
        v                   v                    v
    Provenance         Brief Output         Relevance Scores
    (full chain)       (gold summaries)     (for embedder)
        |                   |                    |
        |                   v                    |
        |            Brief Generator             |
        |            training data               |
        |                                        |
        +-------> All feed next training cycle <-+
```

### Cycle 0: Bootstrap (today)

- Classifier: trained on synthetic knowledge_update scenarios (1314 examples)
- Embedder: generic BAAI/bge-small-en-v1.5
- Brief generator: Gemini 2.0 Flash (API, expensive)
- Answer model: Gemini 2.0 Flash (API, expensive)

### Cycle 1: First specialization

- Classifier: nanoGPT trained on cycle 0 data + organic merken audit logs
- Embedder: vstash retrain with hard negatives from merken recall misses
  (Jay already built this pipeline for vstash)
- Brief generator: Gemma 2B + LoRA fine-tuned on Gemini's brief outputs
- Answer model: Gemma 2B + LoRA fine-tuned on QA experiment pairs

### Cycle 2+: Self-improvement

- More organic data from real Claude Code sessions with merken hooks
- Classifier gets better -> fewer noise events pollute the store
- Embedder gets better -> retrieval accuracy improves -> better recall
- Better recall -> better briefs -> better answers
- Better answers -> user trusts the system more -> more usage -> more data

## Training Data Sources (already exist)

| Source | Location | Records | Trains |
|--------|----------|---------|--------|
| knowledge_update scenarios | experiments/loop_quality/scenarios/ | 1314 labeled events | Classifier |
| merken audit trail | merken_audit collection | grows per session | Classifier, provenance |
| vstash search results | recall() return values | per query | Embedder hard negatives |
| brief_v1 outputs | experiments/consolidation/ | 50+ gold briefs | Brief generator fine-tune |
| LoCoMo QA pairs | experiments/retrieval/locomo/ | 1986 QA pairs | Answer model fine-tune |
| LMEB retrieval | experiments/retrieval/lmeb/ | 12,813 queries | Embedder evaluation |
| LongMemEval | experiments/retrieval/longmemeval/ | 500 queries | End-to-end evaluation |

## Why Small Models Work Here

The key insight from today's experiments: **context quality matters more
than model size.** Gemini Flash at 100% accuracy with good briefs vs 40%
without them. The bottleneck was never model capability -- it was getting
the right information to the model.

Small specialized models excel when:
1. The task is narrow (classify, embed, summarize -- not general reasoning)
2. The input is pre-filtered (each stage only sees what the previous
   stage passed through)
3. Training data is abundant and domain-specific (vstash/merken generate
   this naturally)
4. Latency matters (local inference < API call)
5. Cost matters (free inference < $0.001/query)

## Hard Negatives Pipeline

Jay already built a retrain pipeline for vstash embeddings using
disagreement mining (BM25 vs vector search disagreements). For merken:

```
Hard negative sources:
1. Noise events that share vocabulary with decisions
   "Redis memory spike" (noise) vs "Redis as caching layer" (decision)
   -> same words, opposite labels

2. Cross-topic decisions with structural similarity
   "Replaced X with Y" appears in ALL decision topics
   -> same pattern, different topic labels

3. Temporal confusions
   v1 and v3 events for same topic
   -> same topic, different version (need to retrieve v3, not v1)
```

These are exactly the failure modes from today's experiments. An embedder
trained on these negatives would solve the 40% retrieval accuracy problem
without needing briefs at all -- or make briefs even more effective.

## Implementation Roadmap

### Phase 1: Prove the classifier (this session)
- [x] nanoGPT trained on knowledge_update data
- [ ] Evaluate on val set (accuracy, overfitting analysis)
- [ ] Test on unseen events (generalization check)

### Phase 2: Embedder with hard negatives (next)
- [ ] Extract hard negatives from merken experiments
  - Noise events with high cosine to decision events
  - v1 events retrieved instead of v3
- [ ] Retrain vstash embedder (Jay's existing pipeline)
- [ ] Measure retrieval improvement on knowledge_update scenarios

### Phase 3: Gemma 2B fine-tune for briefs (after Phase 2)
- [ ] Collect brief_v1 outputs as gold training data
- [ ] LoRA fine-tune Gemma 2B on brief generation task
- [ ] Compare: fine-tuned Gemma 2B vs Gemini Flash on brief quality
- [ ] Measure: local inference latency vs API latency

### Phase 4: Close the loop
- [ ] Deploy classifier at merken.remember() time (tag on ingestion)
- [ ] Deploy custom embedder in vstash (replace generic model)
- [ ] Deploy Gemma briefs in merken.consolidate(method="brief_v1")
- [ ] Measure end-to-end: accuracy, latency, cost vs all-API baseline

### Phase 5: Paper
- Title: "Self-Supervised Memory Systems: Training Specialized Models
  from Agentic Memory Operations"
- Contributions:
  1. Structural limit of embedding-based consolidation (negative result)
  2. Temporal briefs as alternative (+46pp at 50 topics)
  3. Self-supervised training loop (operational data -> training data)
  4. Cascade of small models replacing large API models
- Datasets: knowledge_update (synthetic), LoCoMo (public), LongMemEval (public)

## Key Numbers to Date

| Metric | Value | Source |
|--------|-------|--------|
| Retrieval baseline (50 topics) | 40% | knowledge_update_50topics |
| Brief injection (direct) | 96-100% | brief_experiment.py |
| Brief-layer search | 86% | consolidation runner |
| Delta | +46pp | the paper number |
| Embedding consolidation | -10pp to neutral | structural limit |
| ForgetConsolidated | -22pp to -75pp | NeverForget validated |
| LongMemEval R@5 | 0.964 | retrieval substrate |
| LMEB cross-dataset delta | -0.5pp | glass-box is free |

## Connection to Jay's Ecosystem

```
snapvec  -> index backend for embeddings (Snap, Residual, PQ, IVFPQ)
vstash   -> retrieval substrate (hybrid BM25 + vector + RRF)
merken   -> memory loop (remember, recall, consolidate, forget)
nanoGPT  -> classifier/router (the glue layer)
Gemma 2B -> brief generator + answer model (the reasoning layer)
```

Each layer is a separate package, MIT licensed, on PyPI. The
self-supervised loop ties them together: merken generates data,
nanoGPT/Gemma consume it, and the improved models make merken better.
