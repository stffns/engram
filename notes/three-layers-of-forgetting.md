# Three Layers of Forgetting in Agent Memory

*Captured 2026-04-16. From analysis of merken's forgetting primitives.*

## The reframing

The real problem is not deleting data from disk. The real problem is
hiding irrelevance from the LLM's attention. Storage is cheap (vstash +
snapvec compression). Attention is expensive. Therefore: don't delete,
deprioritize.

## Layer 1: Preventive (never remember)

- **Mechanism:** nanoGPT classifier at should_remember
- **What it does:** 86% of noise never enters the store
- **Cost:** ~1ms per event, 800K param model, runs local
- **Status:** implemented, validated (+18pp retrieval accuracy)

This is not forgetting -- it's "never remembering." The cheapest
byte to process is the one you never store.

## Layer 2: Brief-aware deprioritization (safe forgetting)

- **Mechanism:** should_forget checks brief coverage before tombstoning
- **Precondition:** brief_v1 must have generated a brief for the topic
- **Verification:** event content must be reflected in the brief
- **If covered:** tombstone the event (reversible, in audit trail)
- **If not covered:** keep the event active
- **Status:** not implemented. ForgetConsolidated (without coverage
  check) was tested and proved catastrophic (-22pp to -75pp).
  Brief-aware version should be safe because it verifies before acting.

Key difference from ForgetConsolidated:
- ForgetConsolidated: "event was consolidated" -> tombstone (blind)
- Brief-aware: "event content is in a brief" -> tombstone (verified)

Implementation sketch:
```python
def should_forget(event, briefs):
    relevant = find_brief_covering(event.topic)
    if not relevant:
        return False
    if event_content_in_brief(event, relevant):
        return Decision(forget=True, reason="covered by brief")
    return Decision(forget=False, reason="not covered by brief")
```

## Layer 3: Self-correction via audit trail

- **Mechanism:** correlate tombstoned events with failed recalls
- **Signal:** if a query fails AND a tombstoned event would have helped,
  that tombstone was a mistake
- **Training data:** tombstone mistakes = hard negatives for the
  forget classifier
- **Loop:** forget classifier improves -> fewer bad tombstones ->
  better recall -> more training signal

This is the third self-supervised loop in the pipeline:
1. Write filter: audit trail -> nanoGPT noise classifier
2. Embedder: recall disagreements -> hard negatives for embedder
3. Forget: tombstone mistakes -> hard negatives for forget classifier

## Vitality scoring (alternative to binary forget)

Instead of binary tombstone/keep, assign a vitality score:

```
vitality = base_weight
         * brief_coverage_factor    # lower if covered by brief
         * recency_factor           # lower if old and superseded
         * query_hit_factor         # higher if frequently retrieved
```

Events with low vitality rank lower in retrieval but are never deleted.
The LLM sees only high-vitality events in its top-K context.

Benefits:
- No information loss (everything stays in store)
- Storage cost managed by snapvec compression
- Retrieval quality improved by deprioritization
- Reversible (adjust weights, no tombstones needed)
- Training signal from weight adjustments

## Connection to Hindsight's opinion network

Hindsight tracks confidence scores on opinions that update with new
evidence. Vitality scoring is the same mechanism applied to events:
an event's "confidence" (vitality) decreases when newer events
supersede it or when a brief absorbs its content. The event doesn't
disappear -- it just becomes less prominent in retrieval.

## For Paper 2

This is Section material for the self-supervised paper:
- Layer 1 (preventive) is already validated
- Layer 2 (brief-aware) is implementable now
- Layer 3 (self-correction) is the research contribution
- Vitality scoring ties all three together
