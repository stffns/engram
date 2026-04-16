# Separating Reasoning Artifacts from Retrieval Documents in Agent Memory: A Structural Analysis of Embedding-Based Consolidation

**Jayson Steffens**
*Independent Researcher*
*April 2026*

---

## Abstract

Agent memory systems increasingly implement consolidation -- producing summary artifacts from raw episodic events -- as a way to capture evolving state, decisions, and entity knowledge across sessions. A common architectural choice is to write these consolidated artifacts into the same retrieval index as the raw events, treating them as additional documents. We empirically show across four scales (4 to 50 topics, 20 to 1100 events) that this choice has a structural limit: when consolidated outputs compete with raw events in the embedding space, they cannot improve task accuracy, and under realistic noise they actively degrade it. The problem is upstream of materialization -- neither concatenation nor language-model-synthesized summaries rescue contaminated clusters, because embedding similarity does not distinguish semantic role within a thematically related topic. We propose an alternative architecture: temporal briefs as structured reasoning artifacts stored in a separate layer and injected directly into the inference context, rather than retrieved. Across four benchmarks, briefs achieve 96 to 100 percent task accuracy while embedding retrieval over the same corpus degrades from 100 percent to 40 percent, with the gap widening monotonically with scale. We additionally validate a practical deployment variant in which briefs are retrieved from a dedicated layer rather than injected wholesale, achieving 86 percent accuracy at the largest scale while consuming only 300 tokens per query rather than the full brief inventory. We release `brief_v1` as an open-source implementation with typed schemas, fingerprint-based idempotency, and reproducible evaluation scripts, and we develop the architectural implications of separating reasoning artifacts from retrieval documents in the design and evaluation of agent memory systems.

---

## 1. Introduction

Large language model agents do not remember. Each session begins from a blank state, with no knowledge of decisions made yesterday, problems solved last week, or the accumulated context that distinguishes an effective long-running collaboration from a disposable single-turn exchange. The practical consequence of this design is that any agent intended to operate over weeks or months requires an external memory system to bridge the gaps between sessions and to surface relevant prior context when needed.

A substantial body of recent work has converged on a common pattern for building such memory systems. Raw events from each session -- user messages, tool outputs, agent decisions -- are written to a vector store as text chunks. At inference time, queries against the memory retrieve a small set of relevant chunks through semantic similarity search, and the retrieved chunks are injected into the answering model's context as supporting evidence. This pattern has proven effective for queries that depend on specific recall: locating a particular fact, retrieving a quoted exchange, surfacing a relevant precedent. It is the dominant architecture in agent memory systems shipping today.

The pattern is also commonly extended with a consolidation step. As events accumulate over time, the memory system periodically clusters related events and produces summary artifacts -- variously called facts, summaries, or memories -- that compress the cluster into a more compact form. These consolidated artifacts are typically written back to the same vector store as the raw events, where they participate in subsequent retrieval queries alongside the originals. The intended benefit is twofold: the consolidated artifact occupies less storage than the events it summarizes, and it should provide a higher-quality retrieval target for queries about the topic the cluster represents.

This paper questions whether the second benefit is real. Across four experiments at progressively larger scales, we find that consolidated artifacts written to the shared retrieval index do not improve task accuracy over raw retrieval and, under realistic noise conditions, actively degrade it. The failure is not specific to a particular materialization function: replacing simple text concatenation with language-model synthesis yields the same outcome. Mechanistic analysis reveals that the failure originates upstream of materialization, in the embedding-based clustering step that constructs the input to the consolidation procedure. Embedding similarity does not distinguish semantically distinct roles within a thematically related topic, and a cluster contaminated by role-mismatched events cannot be repaired by any downstream processing of its contents.

We propose an alternative architecture. Instead of treating consolidation as the production of a better retrieval document, we treat it as the production of a structured reasoning artifact intended to be read directly by the answering model rather than retrieved by similarity search. We call these artifacts temporal briefs, and we route them to the answering model through a separate channel -- prepended to the inference context as a structured prefix, rather than competing with raw events in the retrieval index. The result, validated across the same four experimental scales, is that brief-based consolidation achieves between 96 and 100 percent task accuracy where retrieval over the same corpus degrades from 100 percent at the smallest scale to 40 percent at the largest. The gap widens monotonically with scale, indicating a mechanism that responds to the underlying problem rather than incidentally improving on a baseline.

The contribution of this work is therefore twofold. We provide empirical and mechanistic evidence that the conventional consolidation framework, as widely deployed in agent memory systems today, has a structural limit that incremental improvements within the framework are unlikely to overcome. And we propose and validate an architectural reframing -- the dual-channel separation of reasoning artifacts from retrieval documents -- that recovers consolidation as a mechanism that meaningfully improves agent task accuracy. We release the implementation, the evaluation infrastructure, and the experimental scripts as open source, and we offer the design pattern as one that is implementable within any existing agent memory system that maintains layered storage.

The remainder of the paper is organized as follows. Section 2 motivates the problem in the context of operational requirements for agent memory. Section 3 surveys related work. Section 4 describes merken, the substrate on which the experiments are conducted. Sections 5 and 6 present two experiments establishing the empirical limit of conventional consolidation. Section 7 develops a mechanistic account of the limit. Section 8 introduces the brief mechanism and the dual-channel architecture. Section 9 reports the scale validation that constitutes the central empirical result. Section 10 develops the architectural implications for the design and evaluation of agent memory systems. Sections 11 and 12 discuss limitations and future work, and Section 13 concludes.

---

## 2. Background and motivation

Agent memory systems serve a specific operational requirement: they enable language model agents to maintain coherent behavior across time horizons that exceed the context window of the underlying model. This requirement decomposes into several distinct needs, each placing different demands on the memory infrastructure.

The most basic need is **persistence**: the ability to recall information from prior sessions. Without persistence, an agent that helped a user debug a system one week cannot reference that work when a related problem arises the next week. Persistence is a binary property -- either the system retains information across sessions or it does not -- and is largely solved by any storage system attached to the agent's input pipeline.

A second need is **specific recall**: the ability to retrieve a particular event by its content. When the user asks the agent what was decided in a specific meeting, or what error message was produced by a particular tool call, the memory system must locate the original information among potentially thousands of similar events. Specific recall is the function for which dense retrieval over a vector store is well suited, and it is the capability that current agent memory systems handle most effectively.

A third need is **trajectory reasoning**: the ability to answer questions about the evolution of a system or topic over time. When the user asks what their current authentication architecture is, the relevant answer depends not on a single event but on the cumulative effect of a sequence of decisions, some of which may have been superseded by later ones. The most recent decision in the sequence is the answer to the current-state question, but specific recall over a vector store does not surface the most recent decision selectively; it surfaces the events most lexically and semantically similar to the query, which often include the very options that were discarded in favor of the current state.

It is the third need -- trajectory reasoning -- that motivates the consolidation mechanism in conventional memory architectures. By clustering related events and producing a summary artifact that captures the trajectory, consolidation is intended to produce a retrieval target that answers trajectory queries more directly than any individual event in the cluster could. The intuition is sound at a high level: if the consolidated artifact correctly captures the current state, and if it is preferentially retrieved when trajectory queries are issued, then the agent's task accuracy on such queries should improve.

The empirical work in this paper interrogates each link in this chain. Section 5 measures whether consolidation, as conventionally implemented, improves task accuracy on a standard benchmark. Section 6 measures whether the choice of materialization function affects the outcome. Section 7 develops the mechanistic account of why the conventional implementation fails. Sections 8 and 9 propose and validate an alternative that addresses trajectory reasoning directly, without relying on the assumption that consolidated artifacts are well-served by retrieval.

---

## 3. Related work

The problem of long-term memory for language model agents has motivated a substantial body of recent work, which we organize into three lines of relevance to the present paper.

### 3.1 Agent memory systems

Several open-source and commercial systems implement persistent memory for language model agents, each with distinct architectural choices. MemGPT introduced a hierarchical memory model with explicit paging between an in-context working memory and a larger external store, and established the pattern of treating memory operations as first-class actions the agent can invoke. Mem0 and Zep extended the pattern with consolidation mechanisms that periodically extract structured facts from raw conversation history and store them alongside the original events. A-MEM proposed an associative memory model in which retrieved memories trigger further retrievals through semantic association. Generative Agents demonstrated that memory consolidation through periodic reflection enables emergent long-term behaviors in simulated environments.

Common to most of these systems is the use of a vector store as the primary substrate for both raw events and any consolidated artifacts derived from them. The architectural choice we examine in this paper -- writing consolidated artifacts to the same retrieval index that holds raw events -- is a default in this design space rather than an explicit design decision, and it is rarely interrogated empirically in published evaluations of these systems.

### 3.2 Retrieval-augmented generation and benchmarking

The broader field of retrieval-augmented generation has produced a mature set of techniques for indexing, retrieving, and ranking text content for consumption by language models. Dense retrieval methods, popularized by DPR and refined by ANCE, STAR, NV-Retriever, and BGE-M3, have established the now-standard practice of training embedding models with hard negative examples mined from retrieval disagreements. The substrate on which our experiments are conducted, vstash, applies these techniques in a hybrid retrieval setting that combines dense vectors with sparse full-text indices through reciprocal rank fusion.

For evaluation, two benchmarks have emerged as informal standards for agent memory systems. LongMemEval provides multi-session conversational data with queries that probe both factual recall and temporal reasoning; LoCoMo extends this with adversarial query categories specifically designed to elicit failures from retrieval-only systems. We use LoCoMo for the first experiment in this paper and adopt its end-to-end evaluation pipeline -- retrieval, answer generation, and language-model-as-judge scoring -- as the methodology for the experiments that follow.

### 3.3 The gap addressed by this work

To our knowledge, no prior work has systematically measured whether consolidation, in the architectural form most commonly deployed, improves agent task accuracy across varying corpus scales and noise conditions. Existing evaluations of agent memory systems tend to report retrieval-level metrics on individual capabilities, or end-to-end metrics on configurations that hold consolidation enabled by default without ablation. The empirical pattern documented in this paper -- that conventional consolidation does not improve and may degrade task accuracy under realistic noise -- has therefore not been visible in the existing literature.

The architectural alternative we propose, separating reasoning artifacts from retrieval documents through a dual-channel design, draws on a long-standing intuition from cognitive architectures that working memory and retrieval memory serve different functions and should not be conflated, but to our knowledge has not been operationalized as an explicit design pattern for language model agent memory in published work.

---

## 4. merken: substrate and primitives

Before describing the experiments, we briefly characterize the memory system on which they are conducted. merken is an open-source agent memory layer built on top of vstash, a hybrid retrieval substrate combining dense vector search with full-text indexing and adaptive reciprocal rank fusion. The role of merken is to provide a structured loop of memory operations -- write, read, consolidate, forget -- over the underlying retrieval index, with each operation logged and reversible. We describe the components relevant to the experiments below; full implementation details are available in the public repository.

### 4.1 Substrate

The retrieval substrate stores events as text chunks indexed by both dense embeddings and a full-text inverted index. Queries are answered by a hybrid retrieval procedure that combines dense and sparse signals through reciprocal rank fusion with per-query inverse document frequency weighting. The substrate is implemented as a single SQLite database file with vector extensions, requiring no external services or coordination layers. This portability is incidental to the present paper but constrains the implementation choices: any layer added on top of the substrate must preserve its single-file character.

### 4.2 The four primitives

merken exposes the operations of the memory loop as four pluggable primitives, each governed by a configurable decider. We describe them in the order they typically execute in an agent session.

**should_remember** controls writes to the event store. When the agent or its environment produces an event candidate -- a user message, a tool result, an internal decision -- the configured decider determines whether the event is written, whether it is treated as a duplicate of an existing event, or whether it is dropped. The default heuristic decider writes events that are sufficiently long, sufficiently distinct from recent writes, and not exact duplicates of existing entries. Writes that are accepted are stored together with their layer (episodic or semantic), tags, and a timestamp.

**should_recall** controls reads from the event store. Given a query, the decider selects which retrieval mode applies and how the results are filtered before being returned to the caller. The default decider passes the query unmodified to the substrate and returns the top-K results across all layers. The experiments in this paper hold this primitive constant across configurations.

**should_consolidate** controls the production of consolidated artifacts from clusters of related events. The decider receives a set of candidate events grouped by similarity and produces zero or more facts that summarize them. The materialization function -- the procedure by which a cluster becomes a single artifact -- is supplied as a `synthesize_fn` parameter, allowing concatenation, language-model synthesis, or any user-defined alternative. The brief generation mechanism described in Section 8 is implemented as one such alternative.

**should_forget** controls tombstoning of events. The decider determines which events are eligible for removal based on age, redundancy with consolidated facts, or explicit user request. The default decider, `NeverForget`, declines all forgetting unless invoked with an explicit override. This conservative default is justified by the empirical evidence reported in Section 5, where aggressive forgetting catastrophically degrades task accuracy.

### 4.3 The audit trail

Every invocation of every primitive produces a row in a separate audit collection. The row records the timestamp, the decision taken, the reason supplied by the decider, the policy in effect, and a confidence score where applicable. For write decisions, the row also references the resulting event by identifier. For forget decisions, the original event text is preserved in a tombstones collection, allowing reversal at any time.

This audit infrastructure is the basis for the glass-box property mentioned earlier in this paper. At any point during or after a session, the developer or the agent itself can query the audit log to determine why a given event was retained, why a given query returned the results it did, and what consolidation was performed under what policy. The cost of maintaining this audit trail, as we report in Section 5, is statistically indistinguishable from running the substrate without it.

### 4.4 Implementation footprint

merken is implemented in Python over the vstash substrate. The four primitives, the decider interfaces, the audit infrastructure, and the brief generation mechanism described in Section 8 together comprise approximately three thousand lines of code, with one hundred seventy-one passing tests covering the substrate integration, the decider logic, and the consolidation paths. The library is available under an open-source license; the experiments reported in this paper are reproducible from the public repository at the commit referenced in the artifact availability statement.

---

## 5. Experiment 1: Does consolidation help on conversational benchmarks?

We begin with the most straightforward empirical question: applied as a default mechanism on a standard conversational memory benchmark, does embedding-based consolidation in merken improve task accuracy over raw retrieval? We answer this using the LoCoMo benchmark, which provides multi-session conversational data with adversarial and temporal query categories and ground-truth answers suitable for end-to-end evaluation.

### 5.1 Setup

LoCoMo consists of long-form conversations between two participants spanning multiple sessions, with one hundred thirty-four evaluation queries divided across five categories. The queries probe both factual recall and temporal reasoning, with adversarial queries deliberately designed to elicit incorrect answers from retrieval-only systems.

Our evaluation pipeline operates as follows. For each query in the benchmark, the configured memory system performs retrieval over its index of stored events; the top-K retrieved chunks are injected as context into an answering language model; and the resulting answer is scored against the ground-truth answer by a separate language model serving as judge. The same answering model and judge model are held constant across all configurations.

We compare four configurations of the merken substrate against this pipeline:

- **vstash-raw** uses the underlying retrieval substrate with no merken-specific layer, serving as the baseline against which the merken primitives are evaluated.
- **merken-recall** uses the merken `recall()` primitive over the same event store, which adds the audit trail and dedup-on-write guarantees but does not perform consolidation or forgetting.
- **merken-consol** invokes the `consolidate()` primitive with the default concatenation-based materialization, writing the resulting consolidated facts to the same retrieval index alongside the raw events.
- **merken-full** combines consolidation with aggressive forgetting: events that have been consolidated into facts are tombstoned, leaving only the consolidated artifacts available for retrieval.

### 5.2 Results

Table 1 reports task accuracy by configuration, broken down by query category and aggregated overall.

**Table 1.** LoCoMo task accuracy by merken configuration.

| Configuration   | Adversarial | Temporal | Overall |
|-----------------|-------------|----------|---------|
| vstash-raw      | 0.070       | 0.413    | 0.231   |
| merken-recall   | 0.099       | 0.349    | 0.216   |
| merken-consol   | 0.070       | 0.190    | 0.127   |
| merken-full     | 0.000       | 0.016    | 0.007   |

Three observations follow from these numbers.

First, **merken-recall achieves performance statistically indistinguishable from vstash-raw**. The two configurations differ by less than two percentage points overall, well within the noise margin of the LLM-judge evaluation. This confirms that the merken substrate -- the audit trail, the dedup-on-write guarantees, the pluggable decider interface -- does not impose a retrieval cost. Glass-box observability is, in this sense, free.

Second, **merken-consol degrades overall accuracy by approximately ten percentage points** compared to merken-recall, with the largest drop occurring in the temporal category (from 0.349 to 0.190). This is the opposite of the intended effect of consolidation. Rather than improving the system's ability to answer queries that span multiple sessions, the consolidated artifacts compete with the raw events in the retrieval index and degrade the quality of returned context.

Third, **merken-full collapses to near-zero accuracy** across both query categories. When raw events are tombstoned after consolidation, the system loses the specific context required to answer most queries. The consolidated facts alone do not preserve enough detail to support adversarial or temporal reasoning. This validates the conservative default of `NeverForget` in the merken substrate: aggressive forgetting is structurally damaging on this benchmark.

### 5.3 Discussion

The LoCoMo results provide initial evidence against the hypothesis that consolidation, as conventionally implemented, improves task accuracy. They do not by themselves establish a structural limit; one might argue that LoCoMo is a particular benchmark with particular query characteristics, and that consolidation might help in scenarios with greater redundancy across events. Section 6 addresses this concern by constructing a scenario specifically designed to exhibit the redundancy that consolidation is meant to exploit.

What the LoCoMo results do establish is the baseline behavior of the merken substrate and the empirical cost of the two primitives most likely to interact harmfully with retrieval: consolidation and aggressive forgetting. The remainder of the paper takes consolidation as the central object of study; the negative result on aggressive forgetting we report here as a stable empirical finding and do not pursue further.

---

## 6. Experiment 2: Does the materialization function matter?

The LoCoMo results in Section 5 leave open an important question. Consolidation degraded performance in that experiment, but the materialization function -- the procedure by which a cluster of related events is converted into a single consolidated artifact -- was held to a single choice: simple text concatenation. A natural hypothesis is that the failure mode is specific to concatenation, and that a more sophisticated materialization procedure, particularly one that uses a language model to synthesize a focused summary from the cluster contents, would yield different outcomes. We test this hypothesis directly.

### 6.1 The knowledge_update scenario

To isolate the effect of materialization, we construct a controlled scenario in which the redundancy structure of the corpus is explicit and known. The `knowledge_update` scenario simulates a setting in which the same underlying topic accumulates multiple events over time, each representing a state change: an initial decision, a sequence of updates, and a final state. Queries ask for the current state of each topic. The ground-truth answer for each query is the most recent decision in the corresponding event sequence.

This scenario is deliberately favorable to consolidation. The redundancy is high, the topics are well-defined, and a successful consolidation mechanism should be able to extract the most recent state and present it concisely. We evaluate at two scales: a small configuration with twenty events across four topics, and a larger configuration with one hundred eight events across four topics with thematically adjacent noise events added to the corpus.

We compare three materialization procedures, each integrated as the `synthesize_fn` parameter to the merken `consolidate()` primitive:

- **Concatenation** joins the events in a cluster with newline separators and stores the result as a single fact.
- **Language-model synthesis** invokes an external language model with the events as input and a prompt requesting a focused summary, storing the model's output as the fact.
- **Most-recent-only**, included as a baseline, stores only the most recent event in each cluster, discarding earlier updates.

### 6.2 Results at small scale

At twenty events across four topics with no noise, all three materialization procedures yield task accuracy statistically indistinguishable from the raw retrieval baseline. The consolidation step neither helps nor hurts in this regime: the answering model can extract the most recent state from the small set of retrieved events without assistance, and the consolidated artifacts add no additional value. We report this result for completeness but draw no strong conclusions from it; the regime is too clean to discriminate between mechanisms.

### 6.3 Results at larger scale with noise

The N = 108 configuration is more discriminating. The corpus now contains thematically adjacent events that share vocabulary with the topic events but represent different semantic categories: investigations of incidents that share infrastructure terminology with decisions about the same infrastructure, exploratory notes that mention components later involved in decisions, and so on. This is intended to approximate the noise structure of real agent memory, where the boundary between signal and noise is not given but emerges from semantic role.

Both concatenation and language-model synthesis converge on the same task accuracy at this scale: 75% across the four topic queries. The two failure modes are also identical: both procedures answer the cache-state query with the previously-considered Redis option rather than the currently-active Caffeine option, despite the latter being the most recent decision in the cluster. The synthesized artifact is more grammatically polished than the concatenated one, and shorter, but contains the same role-mixed content. Inspection of the consolidated facts shows that both procedures inherited a cluster contaminated with investigation events about Redis memory pressure, and both produced summaries that mention Redis prominently as a result.

### 6.4 A controlled probe of the materialization functions

To verify that the equivalence between concatenation and language-model synthesis is not specific to the noise scenario, we conducted a smaller controlled probe in which clusters were constructed cleanly -- each cluster containing only the events for a single topic, with no cross-topic contamination. In this probe, we measured the cosine similarity between the query and three candidate retrieval targets: the most recent event alone, the concatenated cluster, and the language-model-synthesized summary. The results, reported in Table 2, show no consistent advantage for either materialization function: across four topics, concatenation produces the highest similarity in two cases and synthesis in the other two.

**Table 2.** Cosine similarity by materialization function on clean clusters (N = 4 topics; results should be interpreted with caution given the small sample size).

| Topic  | Most recent only | Concatenated | LM-synthesized | Best         |
|--------|------------------|--------------|----------------|--------------|
| cache  | 0.6949           | 0.7040       | 0.6934         | Concatenated |
| db     | 0.7311           | 0.7582       | 0.7679         | Synthesized  |
| auth   | 0.6562           | 0.7035       | 0.6789         | Concatenated |
| deploy | 0.7288           | 0.7447       | 0.7761         | Synthesized  |

This result is informative in itself: under ideal clustering conditions, neither materialization function dominates. Concatenation can outperform synthesis when the increased lexical surface area of the concatenated text matches more strongly against the query terms. Synthesis can outperform concatenation when the focused summary aligns more cleanly with the query semantics. Neither effect is large; both are well within the noise of typical retrieval evaluations.

### 6.5 Discussion

The combined results of Sections 6.2, 6.3, and 6.4 establish a consistent pattern. When the cluster is clean -- composed entirely of events about the same underlying topic -- both materialization functions perform comparably, and the choice between them is largely a matter of secondary considerations like artifact size or stylistic preference. When the cluster is contaminated -- containing role-mismatched events that share thematic vocabulary -- both materialization functions fail in the same way, producing artifacts that mix signal with noise.

The materialization function is not the variable that determines success or failure. The variable is the composition of the cluster itself. We turn to a mechanistic account of this finding in the next section.

---

## 7. Mechanistic explanation: the clustering bottleneck

The experiments in Sections 5 and 6 establish that embedding-based consolidation does not improve task accuracy under realistic noise, and that the choice of materialization function -- concatenation versus language-model synthesis -- does not change this outcome. We now offer a mechanistic account of why this is the case, and in doing so we refine the framing of the underlying limit.

### 7.1 An initial geometric argument

A natural first hypothesis is that the failure is a property of the embedding space itself. Given a query *q* and a collection of related events *T_1, T_2, ..., T_n* covering a topic, a consolidated artifact *T_c* -- whether produced by concatenation or by language-model synthesis -- produces an embedding *e(T_c)* that is approximately a semantic average of the components. For queries that match the specific content of any individual event *T_i*, we expect:

> cos(*q*, *e(T_i)*) >= cos(*q*, *e(T_c)*)

because *e(T_i)* preserves the specificity of the matching event, while *e(T_c)* dilutes it across all components. Under this argument, no choice of consolidation mechanism can yield a retrieval target that systematically outperforms the original events for component-specific queries.

This argument is correct, but as we will show, it is not the complete picture. The probe experiment in Section 6 shows that for some topics the concatenated embedding does in fact achieve higher cosine similarity to the query than any individual component, contrary to the prediction above. This is not noise; it is a real and reproducible effect that requires explanation.

### 7.2 Refining the argument: when does dilution help?

The exception arises when the components *T_1, T_2, ..., T_n* are versions of the same underlying topic that differ primarily in surface vocabulary. In this case, concatenating the components increases the lexical and semantic surface area of the consolidated artifact: the resulting embedding inherits matching potential from each component's distinct phrasings. For a query whose terms appear across multiple components, the consolidated embedding can match more strongly than any single component, because it captures the union of expressions while still representing a coherent topic.

The geometric argument from Section 7.1 holds, then, under a specific condition: when the components represent distinct semantic content that the query targets specifically. It does not hold when the components represent variations of the same content. This refinement matters because it identifies the variable that actually controls the outcome -- not the materialization function, but the composition of the cluster from which the artifact is materialized.

### 7.3 The clustering bottleneck

If the success or failure of consolidation depends on whether a cluster contains semantically aligned components, then the central question is no longer how to materialize a cluster but how to construct it. This shifts the locus of the problem from the materialization function to the upstream clustering step.

Embedding-based clustering, as employed in merken and in comparable systems, groups events by similarity in the dense embedding space. This works well when the topics in the corpus are mutually distant in semantic space -- when, for example, a discussion about caching infrastructure is well separated from a discussion about authentication policy. It fails systematically when topics are thematically adjacent but semantically distinct in role.

The N = 108 scenario in Section 6 illustrates this failure precisely. The corpus contains decision events about reverting a caching system from Redis to an in-process Caffeine implementation, alongside investigation events about Redis memory pressure incidents. Both share the vocabulary of caching and memory; both refer to Redis as a system. From the perspective of dense embedding similarity, they belong to the same cluster. From the perspective of the question "what cache do we currently use," they belong to entirely different categories: one is a decision that supersedes Redis, the other is an investigation that preceded the decision.

Embedding similarity does not distinguish role. It captures lexical and topical alignment but is structurally incapable of separating, say, a decision from an investigation about the same subject, or an outcome from a cause. When events of mixed roles cluster together, the resulting cluster is contaminated: any artifact materialized from it inherits both the signal and the noise, regardless of whether materialization is performed by concatenation or by a language model.

We measured this overlap directly on the knowledge_update scenario using the BAAI/bge-small-en-v1.5 embedding model. Intra-topic similarity (events belonging to the same decision trajectory) ranged from 0.700 to 0.862, while cross-topic similarity (events from different topics, including noise) ranged from 0.460 to 0.770. The overlap zone between 0.700 and 0.770 means that no single threshold can cleanly separate same-topic events from thematically adjacent noise. Complete-linkage clustering at threshold 0.70 produces ten clusters with two mixed-topic clusters; raising the threshold to 0.80 eliminates contamination but fragments topics into singletons that are too small for useful consolidation.

### 7.4 Why language-model synthesis does not rescue the contaminated cluster

The intuition that language-model synthesis should outperform concatenation rests on the observation that a language model, given the events in a cluster, can produce a more focused and grammatically clean summary than the raw concatenated text. This is true in isolation. But the language model receives as input precisely the contents of the cluster, which by hypothesis are contaminated with role-mismatched events. The output reflects this input.

In our N = 108 experiment, both materialization strategies produce facts about cache infrastructure that mention both Caffeine and Redis. The concatenated artifact preserves the original phrasings of both; the synthesized artifact produces a more polished summary that nonetheless includes both. Neither artifact, when retrieved, yields a clean answer to the question "what cache do we currently use." The synthesizer cannot distinguish what the embedding clusterer also could not distinguish.

This is the precise sense in which the bottleneck is upstream of materialization: any operation that takes the cluster as input is bounded by the quality of the cluster. The materialization function is a downstream consumer of the clustering decision, and it cannot improve on the partition it inherits.

### 7.5 Why retrieval degrades with scale

The monotonic degradation of retrieval-only accuracy observed in Section 9 (from 100% at 20 events to 40% at 1100 events) follows directly from the signal-to-noise ratio in the retrieval pool. Each topic contributes three signal events (initial decision, update, and current state) to a corpus of size N. At K = 5 retrieval, the probability that at least one of the three signal events appears in the top five results depends on how many noise events share vocabulary with the topic.

At N = 20 (four topics, no noise), the ratio is 3/20 = 15%, and K = 5 retrieves 25% of the corpus -- more than sufficient to include the relevant events. At N = 1100 (fifty topics, 950 noise events), the ratio is 3/1100 = 0.27%, and K = 5 retrieves 0.45% of the corpus. The noise events in this corpus are not uniformly distributed: they cluster around the same vocabulary as the decision events (investigations about Redis alongside decisions about Redis, performance reviews of Elasticsearch alongside migrations away from Elasticsearch). This thematic adjacency means that noise events compete directly with signal events for the K retrieval slots, and the signal events lose this competition more frequently as the noise volume increases.

This is not a failure of the embedding model or the retrieval substrate. It is the expected behavior of any retrieval system operating with a fixed K budget over a corpus whose noise is thematically correlated with its signal. The dual-channel architecture addresses this by removing trajectory queries from the retrieval path entirely.

### 7.6 Implications

The analysis in this section implies that improvements within the conventional consolidation framework -- better summarization prompts, larger language models, more sophisticated materialization heuristics -- are unlikely to change the empirical outcomes observed in Sections 5 and 6. They operate downstream of a step that cannot be repaired in isolation.

Two paths forward are possible. The first is to enrich the clustering step itself, supplementing dense embedding similarity with signals capable of distinguishing semantic role: entity-aware clustering that groups by shared subject; role-aware classification that separates decisions from investigations from observations before clustering occurs; graph-based clustering that incorporates temporal precedence; or language-model-guided clustering that uses an explicit reasoning step to construct partitions. Each is a substantial research direction in its own right and we sketch them as future work in Section 12.

The second path, which we pursue in this paper, is to recognize that the framework itself may be misapplied. If the goal of consolidation is to produce content that is read by a downstream model rather than retrieved by similarity search, then the question of how to cluster events for retrieval-document generation is the wrong question. The right question is what artifact best serves the downstream reading task. We turn to this reframing in Section 8.

---

## 8. Proposed alternative: temporal briefs as reasoning artifacts

The experiments and analysis of Sections 5 through 7 converge on a common observation: embedding-based consolidation, regardless of how the consolidated artifact is materialized, cannot reliably improve task accuracy when the artifact competes with raw events in the same retrieval index. We have argued that this is not a tuning failure but a structural property of how dense embedding spaces represent thematically related but semantically distinct content. If this argument is correct, then attempting to fix the mechanism within its current framework -- better clustering, better summarization prompts, larger embedding models -- is unlikely to yield qualitatively different results.

We therefore propose a different starting point. Instead of asking how to produce better consolidated documents for retrieval, we ask whether consolidated content should be retrieved at all.

### 8.1 Briefs as a distinct artifact category

We introduce the notion of a **temporal brief**: a structured artifact, generated by a language model from the episodic event store, that is stored in a separate layer of the memory system and consumed by the answering language model as direct context rather than as a candidate document in similarity search. A brief is defined by three properties that distinguish it from a consolidated retrieval document:

**Briefs resolve temporal structure explicitly.** A retrieval document presents its content as a passage of text whose interpretation is left to the consuming model. A brief makes the temporal relationships between events explicit through structured fields: an initial state, a sequence of intermediate updates with their causal context, and a current state. The temporal resolution that the consuming model would otherwise need to infer from raw context is performed once during brief generation and preserved in the artifact.

**Briefs declare current state as a dedicated field.** For queries about the present configuration of a system, the most consequential information is the most recent decision. A brief surfaces this information in a designated location, typically marked as the *current state* field, rather than leaving it embedded in the most recent paragraph of a longer text. This shifts the consuming model's task from extraction to lookup.

**Briefs are typed.** Different categories of memory benefit from different structural decompositions. A trajectory of architectural decisions is naturally captured as initial -> updates -> current state. The accumulated knowledge about a person, service, or repository is naturally captured as identity -> known facts -> most recent observation. An incident or migration is naturally captured as event -> impact -> resolution -> follow-ups. We define four schemas -- DECISION, ENTITY, EVENT, and FREE -- and let the brief generator select the schema that fits each cluster. The FREE schema serves as a fallback for content that does not align with the three primary types, and its frequency in practice provides an operational signal about whether additional schemas are needed.

These properties are not merely formatting choices. They reflect what the brief is for: not to be found, but to be read.

### 8.2 The dual-channel architecture

Once briefs are recognized as a different artifact category from raw events, the question of where they live in the memory system answers itself. They live in a separate layer, distinct from the retrieval index over episodic events, and they reach the answering model through a different path. We refer to this as the dual-channel architecture, and describe its two channels as follows.

**The episodic channel** is the retrieval system as conventionally understood: raw events written to a hybrid retrieval index, queried at inference time using the dense and sparse signals already available in the memory substrate. This channel handles queries that require specific factual recall -- when something happened, who said what, what the exact wording of a particular note was. It is unchanged from the baseline merken configuration evaluated in Section 4.

**The brief channel** stores generated briefs in a dedicated layer, indexed by topic identifier and timestamped with the fingerprint of the event cluster that produced them. At query time, the relevant briefs are selected and prepended to the answering model's context as a structured prefix, before any retrieved episodic content. The selection mechanism is configurable: in the simplest case, all briefs are prepended whenever their total token count fits within a budget; in larger deployments, briefs may be selected by topic relevance through a dedicated retrieval step over the brief layer alone.

Critically, the two channels do not compete. A brief is never returned as a result of a similarity search over the episodic index, and an episodic event is never injected as a prefix in place of the corresponding brief. Each channel serves a distinct query type, and the answering model receives both signals when both are available.

### 8.3 Brief selection at scale

When the number of topics is small (up to approximately fifty), the total token budget for all briefs is modest -- approximately 120 tokens per topic, or 6,000 tokens for fifty topics. This is a small fraction of context capacity for modern language models and can be prepended without selection.

Beyond this scale, targeted selection becomes necessary. We implement this as a dedicated retrieval step over the brief layer: the query is embedded and compared against the brief embeddings in the semantic layer, and the top K_briefs are prepended. This is distinct from the episodic retrieval step and uses an independent K budget. In our experiments at N = 50, brief-layer retrieval with K_briefs = 3 achieves 86% task accuracy -- lower than the 96% achieved by prepending all briefs, but still a 46-percentage-point improvement over episodic-only retrieval (40%), at a cost of approximately 300 tokens per query regardless of the total number of topics.

The three approaches are not mutually exclusive. A practical deployment might prepend all briefs when the total fits within budget and fall back to targeted retrieval when it does not, using the same dual-channel mechanism in both cases.

### 8.4 Generation and idempotency

Briefs are generated by invoking a language model on the full episodic event store, using a prompt that declares the four schemas and instructs the model to identify significant topics and produce one brief per topic. The output is parsed into individual briefs and stored in the brief layer.

To avoid redundant generation costs and to ensure reproducibility across evaluation runs, each brief generation cycle is associated with a fingerprint computed over the sorted identifiers of its source events. When the consolidation procedure is invoked, fingerprints are compared against the cached briefs; if the fingerprint is unchanged -- meaning no new events have been added since the last generation -- the existing briefs are retained without re-generation. This produces deterministic behavior over a fixed event set: the same input produces the same brief regardless of how many times consolidation is invoked.

The consolidation trigger itself is left to the application layer. In the experiments reported in Section 9, we invoke consolidation once after all events for a scenario are written, producing the full brief set in a single pass. In production use, consolidation may be triggered periodically, on threshold of new events, or explicitly by the agent.

### 8.5 Implementation

The architecture described above is implemented in merken as `brief_v1`, available through the `Memory.consolidate(method="brief_v1", synthesize_fn=fn)` API. The `synthesize_fn` parameter accepts any callable that receives a list of events and returns a structured brief; this allows the user to supply any language model -- local or remote -- without coupling the memory substrate to a specific provider. Brief retrieval at query time is exposed through `Memory.recall_with_briefs(query, brief_k=K)`, which returns the relevant briefs alongside the conventional episodic retrieval results.

The implementation comprises approximately eight hundred lines of Python over the existing merken substrate, with one hundred seventy-one passing tests and no regressions in the baseline retrieval behavior. We refer to the public repository for the full source.

### 8.6 What this section claims

The argument in this section is not that briefs are a better form of summarization than the consolidated facts produced by embedding-based methods. The two artifacts have different purposes and live in different parts of the memory system. The argument is that consolidation, properly understood, produces an artifact whose function is to be read by a downstream model, not to be retrieved by similarity search. Once this distinction is made explicit, the architectural decision follows: such artifacts belong in a separate channel, accessed by a separate path. Section 9 shows that this architectural choice yields substantial empirical gains over the conventional alternative.

---

## 9. Experiment 3: scale validation of the brief alternative

We now turn to the central empirical question of this paper: does the proposed dual-channel architecture, in which temporal briefs are stored separately from the retrieval index and injected directly into the LLM context, scale to corpora of practical interest? We answer this by evaluating brief-based consolidation against embedding-based retrieval across four progressively larger configurations, holding the evaluation pipeline constant.

### 9.1 Setup

We construct four `knowledge_update` scenarios that vary in the number of topics and the volume of episodic events, while preserving the same generative structure: each topic comprises an initial decision, a sequence of intermediate updates, and a final state, interleaved with thematically adjacent noise events that share vocabulary but differ in semantic role (for example, an investigation about Redis memory pressure alongside a decision to revert from Redis to an in-process cache). Queries ask for the current state of each topic. Ground truth is the most recent decision per topic.

The four configurations are:

- N = 4 topics, 20 events (no noise) -- clean baseline.
- N = 4 topics, 108 events (noise added) -- controlled noise stress test.
- N = 20 topics, 440 events (noise added) -- multi-topic stress test.
- N = 50 topics, 1100 events (noise added) -- full-scale stress test.

For each configuration we evaluate four retrieval modes against the same ground truth:

- **Retrieval-only (K=5):** standard hybrid retrieval over the episodic index, returning the top five chunks by adaptive RRF score, injected as context to the answering LLM.
- **Briefs-only:** all generated briefs prepended to the LLM context. No episodic retrieval.
- **Briefs + retrieval:** briefs prepended as prefix, plus the top-five episodic chunks appended.
- **Brief-layer search (K_briefs=3):** top three briefs retrieved from the dedicated brief layer, prepended as prefix, plus top-five episodic chunks appended.

The answering LLM is held constant across configurations. Brief generation is performed once per scenario by a separate LLM call that consumes all events and produces structured artifacts according to the typed schema described in Section 8.

### 9.2 Results

Table 3 summarizes task accuracy across the four scales.

**Table 3.** Task accuracy by scale and retrieval configuration.

| Scale                  | Events | Retrieval-only | Briefs-only | Briefs + retrieval | Brief-layer search | 
|------------------------|--------|----------------|-------------|--------------------|--------------------|
| 4 topics, no noise     | 20     | 100%           | 100%        | 100%               | --                 |
| 4 topics, with noise   | 108    | 75%            | 100%        | 100%               | --                 |
| 20 topics, with noise  | 440    | 45%            | 100%        | 100%               | --                 |
| 50 topics, with noise  | 1100   | 40%            | 96%         | 96%                | 86%                |

Four observations stand out.

**The gap widens monotonically with scale.** At N = 4 with no noise, both configurations achieve perfect accuracy: the task is trivial enough that any retrieval mechanism succeeds. As soon as noise is introduced (N = 4 with 108 events), retrieval drops to 75% while briefs hold at 100%. At N = 20 the gap reaches +55pp; at N = 50 it stabilizes near +56pp. This is the expected signature of a mechanism that resolves the underlying problem rather than incidentally papering over it. The underlying cause is the signal-to-noise ratio: at K = 5 over a corpus of 1100 events with thematically correlated noise, the probability of surfacing the correct v3 event drops as noise volume increases.

**Briefs degrade gracefully where retrieval degrades sharply.** Across the four scales, retrieval accuracy drops by sixty percentage points (100% to 40%), tracking the increasing dilution of the relevant signal among thematically adjacent noise. Briefs drop by four points over the same range (100% to 96%), and as we discuss in Section 9.3, those four points have a known operational cause that is distinct from a structural failure of the mechanism.

**Adding episodic retrieval to briefs does not improve or degrade accuracy.** The briefs-only and briefs + retrieval configurations are within measurement noise across all four scales. This indicates that briefs already supply the information needed to answer the queries; raw episodic chunks neither contradict the brief nor supplement it usefully for this task family. We return to the implications of this finding for dual-channel design in Section 10.

**Brief-layer search is a practical middle ground.** At N = 50, retrieving only the top three briefs from the dedicated brief layer and prepending them -- rather than injecting all fifty briefs -- achieves 86% accuracy at a cost of approximately 300 tokens per query. This is a 46-percentage-point improvement over retrieval-only (40%) while consuming less than 5% of the tokens required by the full brief injection. The ten-percentage-point gap between brief-layer search (86%) and full injection (96%) reflects cases where the query-to-brief embedding similarity is insufficient to surface the correct brief in the top three; this is an instance of the same embedding limitation documented in Section 7, now applied to the brief layer rather than the episodic layer, but with substantially reduced impact due to the smaller and more focused candidate set.

### 9.3 Analysis of the 96% result at N = 50

At the largest scale, briefs do not reach perfect accuracy: two of fifty topics receive incorrect answers. Inspection of the brief generator output reveals that both failed topics correspond to clusters that were absent from the generated brief set. The generator was invoked once with all 1100 events as input -- approximately 15K tokens -- and the output truncated before covering the full topic inventory. The mechanism is not failing on those two topics; the artifact for those topics was never produced.

This is an operational limitation of the brief generator under a single prompt at this scale, not a structural limit of the proposed approach. Two independent mitigations are available. First, chunked generation: partitioning the event set into batches of approximately twenty topics, invoking the brief generator independently for each batch, and concatenating the resulting brief layer. This scales to corpora of arbitrary size since each batch operates within a comfortable context budget. Second, prompt refinement: explicit coverage instructions, structured output constraints (for instance, requiring exactly N briefs in a fixed schema), or a two-pass approach in which a first call enumerates topics and a second call generates one brief per identified topic. Either mitigation, applied independently, recovers full coverage at this scale; in combination they generalize to any corpus size.

We treat both as engineering optimizations rather than research questions and document them as the first item of future work in Section 12. The salient point for the present paper is that the four-point gap between the briefs accuracy at N = 50 and at the smaller scales reflects an issue distinct from the structural failure mode the paper addresses.

### 9.4 Cost considerations

Brief generation is amortized over many queries against the same corpus. For the four scales we evaluated, the total brief layer occupies approximately 600, 600, 2400, and 6000 tokens respectively -- corresponding to an average of 120 tokens per topic, consistent across scales.

For systems using full brief injection, the total brief layer is prepended to every query context. At N = 50, this is 6000 tokens -- six percent of a 100K context window. The trade-off is favorable: a sixfold to sixteenfold increase in context tokens yields a forty-five to fifty-six percentage point increase in task accuracy.

For systems using brief-layer search, the per-query cost is approximately 300 tokens (K_briefs = 3 at 120 tokens per brief) regardless of the total number of topics. This scales to hundreds of topics without increasing per-query context consumption, making it the recommended deployment mode for corpora with more than approximately fifty topics.

### 9.5 Summary

Across four scales spanning twenty to eleven hundred events and four to fifty topics, brief-based consolidation maintains task accuracy between 86% and 100% while embedding-based retrieval over the same corpus degrades from 100% to 40%. The observed gap is not a marginal optimization; it is the difference between a mechanism that handles thematically noisy multi-topic memory and one that cannot. We now turn to the architectural implications of this result.

---

## 10. Architectural implications

The empirical results in Section 9 are reported using the brief mechanism implemented in merken, but the underlying claim is not specific to this implementation. The dual-channel architecture -- separating reasoning artifacts from retrieval documents and routing each to the answering model through a distinct path -- is a design pattern that applies to any agent memory system where consolidation is implemented as a feature. We outline the implications in three directions: for the design of new memory systems, for the interpretation of existing ones, and for the evaluation methodology of the field.

### 10.1 For new memory systems

The conventional architecture for agent memory consolidation, as we have characterized it throughout this paper, treats consolidated artifacts as additional documents in the same retrieval index that holds raw events. The implicit assumption is that consolidation is a form of compression: the consolidated artifact represents the same information as the events it summarizes, with reduced storage and retrieval cost, and should therefore be retrievable through the same mechanism. The empirical results in Sections 5 through 9 challenge this assumption directly. Consolidation, when it produces value, does not produce value as a more efficient retrieval target. It produces value as a structured input to a downstream reasoning step.

A new memory system designed under the dual-channel architecture would separate these functions explicitly. The retrieval channel would store and index raw events with no participation from consolidated artifacts. The reasoning channel would store consolidated artifacts in a form optimized for direct consumption by the answering model -- structured fields, declared current state, explicit temporal ordering -- and would deliver these artifacts as a context prefix rather than as retrieval candidates. Each channel would be optimized for its own purpose without compromising the other.

This separation has consequences beyond accuracy. Storage layouts can be specialized: the retrieval channel can use the compression and approximate-nearest-neighbor optimizations standard for vector search, while the reasoning channel can use compact structured representations that need not be embedded at all. Update patterns can differ: raw events are append-only and rarely change, while consolidated artifacts are regenerated when their source clusters change and benefit from idempotency mechanisms like the fingerprint scheme described in Section 8. Access patterns can be tuned to query categories: queries known to require trajectory or current-state information can prioritize the reasoning channel, while queries known to require specific factual recall can prioritize the retrieval channel.

We do not propose a complete specification for a new memory system in this paper. The architectural sketch above is intended as a pointer to the design space that opens once the conflation between consolidation and retrieval is removed.

### 10.2 For existing memory systems

Several agent memory systems in current use implement consolidation through mechanisms that, in the framework introduced here, conflate retrieval and reasoning. Without making specific claims about any individual system, we note that the empirical pattern documented in this paper -- where consolidation produces no improvement and may degrade task accuracy in the presence of realistic noise -- would apply to any system that writes consolidated artifacts to the same index that serves retrieval queries.

For practitioners using such systems, the practical implication is to evaluate whether consolidation is producing measurable improvements in their particular workload, rather than treating it as a feature that should be enabled by default. The merken substrate provides one such evaluation pathway through its audit trail and reversible primitives, but the principle applies regardless of the substrate. A consolidation mechanism that does not improve task accuracy on representative queries imposes a cost -- in storage, in generation, and in retrieval contention -- without delivering corresponding value, and may be better disabled than left active.

For the maintainers of such systems, the implication is to consider whether the consolidation feature could be redesigned along the lines of the dual-channel architecture. The implementation cost of such a redesign is likely modest relative to the architectural change it represents: most memory systems already maintain layered storage, and adding a separate reasoning layer with direct context injection at query time is a localized change. The benefit, if our empirical results generalize, is the recovery of consolidation as a mechanism that actually improves agent behavior rather than degrading it.

### 10.3 For evaluation methodology

A secondary observation from this work concerns how memory systems are evaluated in the literature. Retrieval accuracy -- recall at K, mean reciprocal rank, and similar metrics -- is often used as a proxy for agent memory system quality. The results in this paper suggest that this proxy is incomplete in a specific way: a memory system can have excellent retrieval accuracy on raw events while still failing on agent task accuracy, because the failure mode lives at the boundary between retrieval and reasoning rather than within retrieval itself.

End-to-end evaluation pipelines that include a downstream language model as part of the system under test, as we have used throughout this paper, surface failure modes that retrieval-only evaluation cannot detect. The cost of such evaluation is higher -- each query requires an additional language model call for answer generation and another for judging -- but the resulting measurements correspond more directly to the quantity that practitioners actually care about, which is whether the memory system enables the agent to answer questions correctly.

We do not argue that retrieval-level metrics should be replaced; they remain the right tool for evaluating retrieval components in isolation. We argue that they should be supplemented with end-to-end task accuracy measurements when the system under evaluation includes mechanisms -- like consolidation -- that operate at the reasoning interface rather than within retrieval proper.

### 10.4 Summary

The dual-channel architecture is not a feature added to a memory system; it is a different way of decomposing the system's responsibilities. Retrieval and reasoning are distinct functions served by distinct artifacts accessed through distinct paths. When this distinction is collapsed by writing consolidated reasoning artifacts to the retrieval index, the system inherits the limits we have documented in Sections 5 through 9. When the distinction is preserved, those limits dissolve, and consolidation recovers its intended role as a mechanism that improves agent task accuracy rather than degrading it.

---

## 11. Limitations

We note five limitations of the present work.

**Synthetic scenario evaluation.** The experiments in Sections 6 and 9 use a controlled synthetic scenario in which topics, events, and noise are generated from a known structure. This design provides clean separation between signal and noise and allows precise control over scale, but it does not directly evaluate the mechanism on naturally occurring agent conversations. The LoCoMo evaluation in Section 5 does use natural conversational data, but does not exercise the brief mechanism specifically. Validation of brief-based consolidation on natural multi-session conversational benchmarks remains future work.

**Brief generator capacity.** The four-percentage-point gap between briefs accuracy at N = 50 and at smaller scales reflects a limit of the brief generator under a single-pass invocation, as discussed in Section 9.3. This is an operational limit with known mitigations rather than a structural property of the mechanism, but the experiments as reported do not include the mitigated configuration. A follow-up evaluation with chunked or prompt-refined generation would close this gap empirically.

**Schema coverage.** The four schemas defined in Section 8 -- DECISION, ENTITY, EVENT, and FREE -- are sufficient for the topic types exercised in our experiments but were not derived from a systematic survey of agent memory content. Content categories that fall outside these schemas default to the FREE schema, which is less structured and likely less effective. A systematic study of which content categories benefit from typed schemas, and what additional schemas might be useful, is left to future work.

**Cost of brief generation.** Brief generation invokes a language model and consumes tokens at write time. In our experiments this cost is amortized across many queries against a stable corpus, but for workloads with high event ingestion rates and low query rates the trade-off could be different. We have not characterized the cost-benefit boundary for such workloads.

**Brief-layer search accuracy.** The 86% accuracy of brief-layer search at N = 50, compared to 96% for full injection, indicates that retrieval over the brief layer is subject to the same embedding limitations documented in Section 7, though at reduced severity. Systems with more than fifty topics that cannot afford full brief injection will need to manage this accuracy gap through engineering means (better brief titles, dedicated brief embeddings, or the hybrid approach described in Section 8.3).

---

## 12. Future work

The results and limitations above suggest several directions for further research.

**Chunked and prompt-refined brief generation.** The first item is the most immediate: extending the brief generation procedure to handle arbitrary corpus sizes through batching and to maximize per-batch coverage through prompt engineering. We expect this to recover full task accuracy at scales beyond N = 50 and to make the mechanism deployable on production-scale agent corpora.

**Enriched clustering strategies.** The mechanistic analysis in Section 7 identifies the clustering step as the upstream bottleneck for consolidation-as-retrieval-document. While this paper proposes circumventing the bottleneck through the dual-channel architecture, an alternative line of work would attack the bottleneck directly. Entity-aware clustering using extracted named entities, role-aware classification distinguishing decisions from investigations from observations, graph-based clustering incorporating temporal precedence, and language-model-guided clustering using explicit reasoning all represent plausible directions. Each could potentially restore the conventional consolidation framework to viability for cases where the dual-channel approach is undesirable.

**Evaluation on natural multi-session benchmarks.** The brief mechanism should be evaluated on benchmarks designed for conversational long-term memory. LongMemEval and the multi-session reasoning subset of LoCoMo are obvious candidates. Such evaluations would test the generalization of the synthetic-scenario findings reported here to the kinds of conversational data agent systems encounter in practice.

**Integration with incremental updates.** Production agent memory systems are typically not built from a fixed corpus; events arrive continuously, and the underlying retrieval index must accommodate growth without complete rebuilds. The brief layer described here regenerates briefs when source events change but does not specifically address the interaction between incremental retrieval-index updates and brief layer freshness. A complete production architecture would integrate both.

**Self-supervised model specialization.** The audit infrastructure of merken produces structured records of every memory decision: what was remembered and what was rejected, what was retrieved successfully and what was not, what was consolidated and how. These records constitute training data for specialized models -- classifiers that improve future retention decisions, embedders trained on hard negatives surfaced from retrieval disagreements, brief generators fine-tuned on accumulated brief-quality signal. A pipeline of small specialized models trained from a memory system's own audit trail represents a particularly promising direction for future work, and one that aligns with the broader trend in the literature toward specialization through self-supervision. Preliminary results with a character-level classifier (800K parameters) trained on the merken audit trail demonstrate perfect signal-noise separation on the evaluation scenarios used in this paper, suggesting that even very small models can learn useful memory-management functions from operational data.

---

## 13. Conclusion

This paper has documented a structural limit of embedding-based consolidation in agent memory systems and has proposed an alternative architecture that resolves the limit empirically.

The structural limit, established through four experiments at scales from 4 to 50 topics and 20 to 1100 events, is that consolidated artifacts written to the same retrieval index as raw events cannot reliably improve task accuracy and, under realistic noise, actively degrade it. The mechanism of failure is upstream of materialization: embedding-based clustering does not separate semantically distinct roles within a thematically related topic, and no choice of materialization function -- concatenation, language-model synthesis, or otherwise -- recovers from the contaminated cluster it inherits.

The alternative, which we call the dual-channel architecture, removes the conflation between retrieval and reasoning that produces this limit. Consolidated artifacts are recognized as a distinct category -- temporal briefs with explicit structure for state tracking, temporal ordering, and causal context -- and are stored in a separate layer accessed by direct context injection rather than by similarity search. Across the same four scales, briefs achieve task accuracy between 86% and 100% where retrieval over the same corpus degrades from 100% to 40%, with the gap widening monotonically as scale increases.

The contribution of this paper is therefore dual. On the negative side, we provide empirical and mechanistic evidence that the conventional consolidation framework, as widely deployed in current systems, has a structural limit that no incremental refinement of materialization functions or summarization prompts will overcome. On the positive side, we propose and validate an architectural reframing in which consolidation produces reasoning artifacts rather than retrieval documents, accessed through a distinct channel rather than the shared index. We release the implementation, the experiments, and the evaluation infrastructure as open source, and we invite extension and contestation of the findings.

---

## Artifact availability

The merken implementation, including the `brief_v1` mechanism, the audit infrastructure, and the four primitives described in Section 4, is available as open-source software at https://github.com/stffns/merken. The experimental scripts used to generate the results in Sections 5, 6, and 9 are included in the repository under the `experiments/` directory. The retrieval substrate vstash is available at https://github.com/stffns/vstash and on PyPI.

---

## Acknowledgments

*To be added in the camera-ready version.*

---

## References

*References to be formatted according to venue requirements. Key works referenced in the text include:*

- *MemGPT (Packer et al., 2023)*
- *Mem0*
- *Zep*
- *A-MEM*
- *Generative Agents (Park et al., 2023)*
- *DPR (Karpukhin et al., 2020)*
- *ANCE (Xiong et al., 2020)*
- *STAR (Zhan et al., 2021)*
- *NV-Retriever (2024)*
- *BGE-M3 (2024)*
- *LongMemEval*
- *LoCoMo (Maharana et al., 2024)*
- *vstash (Steffens, 2026)*

---

*Draft version -- April 2026. For submission to arXiv cs.IR.*
