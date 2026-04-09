# engram benchmark strategy

> **Working document, 2026-04-09.** Design decisions only. No results
> here — those live in the per-benchmark `RESULTS.md` under
> `experiments/retrieval/` and `experiments/loop_quality/`. When a
> decision in this doc changes because of data we collected, the
> decision gets updated with a strikethrough and a pointer to the
> commit that changed it. We do not silently revise strategy to
> match outcomes.

## 0. Why this doc exists

engram lives in a small corner of an opinionated space. Every
month a new "AI memory" project publishes a blog post with a
number on LongMemEval or LoCoMo, frames it as state of the art,
and disappears into the backlog six weeks later. Without a
standing discipline for *which* benchmarks engram measures itself
against, *what* the reference claims actually mean, and *how*
engram will report its own results, we drift into two failure
modes simultaneously:

1. **Chasing numbers** — every time a competitor publishes a
   figure, we bolt on a runner that "also measures" the same
   thing, producing a scoreboard with zero load-bearing meaning.
2. **Measuring in isolation** — ignoring public benchmarks
   because they're imperfect, at which point we can't answer
   "how does engram compare to X?" when anyone asks.

The honest middle path: maintain a small set of public
benchmarks as **absolute positioning** against published claims,
and a separate, larger set of **scenario benchmarks** as
engram's actual design bar. Keep them visibly separate. Never
let a public benchmark drive engram's defaults — that's what the
scenarios are for. Never let a scenario result be claimed as a
public-benchmark win — that's what the retrieval/ runners are for.

This doc defines both surfaces, the rules of engagement, and the
phased roadmap.

---

## 1. Landscape of public AI-memory benchmarks

A partial, opinionated survey of what the "AI memory" space
treats as measurable. Citations are mine, from reading papers and
repos as of early April 2026. Numbers I'm less confident about
are marked **(unverified)**.

### 1.1 LongMemEval
- **Paper / repo:** Xiaowu Wu et al., 2024. Dataset:
  `xiaowu0162/longmemeval-cleaned` on HuggingFace.
- **Shape:** ~500 multi-session conversational QA questions with
  haystack of ~50 sessions per question. Each haystack has one
  or more "answer sessions" plus distractors. Metric is R@k —
  whether the top-k retrieved chunks include at least one from
  an answer session.
- **What it measures:** retrieval quality on chat-replay data.
  Not memory decisions, not consolidation, not recall-over-time.
  Offline, single-shot, no loop.
- **Why it's dominant:** it's the most public, most reproducible,
  and has the largest cluster of published claims. Mempalace,
  Mem0, Zep, Mastra, Supermemory, Letta all cite it.
- **Engram status:** runner implemented at
  `experiments/retrieval/longmemeval/runner.py`. Dataset cached
  locally. Ran n=3 (sanity) and n=10 (first signal, CI wide).
  Full n=500 run deferred — estimated 6h per baseline on current
  hardware, scheduled as an overnight run.

### 1.2 LoCoMo
- **Paper:** "Evaluating Very Long-Term Conversational Memory"
  (Maharana et al., 2024).
- **Shape:** ~35 conversations, each ~600 messages over weeks.
  Five question types: single-hop, multi-hop, temporal reasoning,
  knowledge-update, adversarial. Metric is accuracy on a
  generative answer (judge model compares generated answer to
  gold).
- **What it measures:** long-horizon memory *quality*, not pure
  retrieval. Tests whether the system can correctly integrate
  information across weeks, handle updates to earlier facts, and
  refuse adversarial misdirection. Much closer to what a real
  agent needs than LongMemEval.
- **Why it's harder to publish against:** requires a judge model.
  Every published LoCoMo number is implicitly "X's memory system +
  Y's judge model," and the judge choice dominates the numbers.
- **Engram status:** not implemented. Would need a judge model
  path — first honest open question is whether engram ships with
  an Ollama-backed judge option or requires the user to plug in
  their own.

### 1.3 MemBench / MemoryBank
- **Paper:** MemoryBank (Zhong et al., 2023), MemBench (Chen et
  al., 2024) **(unverified — may be conflated in the wild)**.
- **Shape:** unclear to me without re-reading the papers. Both
  are cited by a handful of posts but don't have the same
  repo-level publicity as LongMemEval or LoCoMo.
- **Status:** **deferred pending verification** — I'd want to
  read the actual datasets and runners before deciding if either
  is load-bearing for engram.

### 1.4 DialSim
- **Paper:** "DialSim" (Kim et al., 2024) **(unverified)**.
- **Shape:** multi-character dialogue simulation, stress-tests
  whether a memory system attributes the right statement to the
  right speaker over long spans.
- **Relevance to engram:** low for v1. engram doesn't model
  speaker identity as a first-class dimension. Would need
  `layer="speaker:<name>"` or similar. **Defer** until we have
  a scenario that demands per-speaker recall.

### 1.5 Needle-in-haystack style tests
- **Not really memory benchmarks** but often cited as such.
  NIAH measures whether a *model's context window* can find a
  planted fact in long context. That's LLM capability, not
  memory-system capability. Any memory system will score near
  100% on NIAH because the memory system is doing retrieval, not
  context stuffing.
- **Status:** not a target. Noise.

### 1.6 HotpotQA / NaturalQuestions / MRQA
- General QA benchmarks occasionally repurposed as "memory"
  benchmarks by systems that ingest the corpus and then answer
  questions against it. This is RAG, not memory in the
  agent-loop sense.
- **Status:** not a target. engram's audience isn't RAG-as-a-
  service.

### 1.7 Custom scenario suites (the wild)
- Many memory systems publish their own bespoke scenario files.
  Mem0 has `mem0-bench`, Letta has scenarios in their
  examples directory, mempalace has a `benchmarks/` directory
  with three runners.
- These are useful to *read* for scenario design inspiration but
  are not competitive baselines because there's no shared
  ground truth and they're evaluated by the publishing party.
- **Status:** not targets. Read for ideas.

---

## 2. Published numbers in the wild

A snapshot of what "everyone knows" as of 2026-04-09. All of these
should be treated as **claims** until engram has reproduced them
or engram has a mechanical reason to trust them. Several blew up
in public review within weeks of being announced.

| System | Benchmark | Claimed | Mode | Reproducible? | My read |
|---|---|---|---|---|---|
| **mempalace** | LongMemEval R@5 | **96.6%** | raw (no LLM) | ✓ runner at `benchmarks/longmemeval_bench.py` | Strongest verifiable claim in the space. Independently reproduced by a community member in <5min on an M2 Ultra. Post-launch correction note exists but the raw number survived. |
| mempalace | LongMemEval R@5 | 100% (500/500) | hybrid + Haiku rerank | ✗ rerank pipeline not in public scripts | The correction note explicitly flagged this as unreproducible at launch. Treat as aspirational. |
| Mem0 | LongMemEval (their paper) | ~85% **(unverified)** | hybrid + GPT-4 | partial — paper has methodology, runner incomplete | Their pipeline extracts facts with an LLM before indexing. Much higher wall-clock cost than raw retrieval. Not directly comparable to mempalace's raw mode. |
| Zep | LongMemEval via Graphiti | ~85% **(unverified)** | hybrid + graph + rerank | partial | Zep's numbers lean on a temporal knowledge graph they build at ingest time. Different problem shape than "find the right chunk." |
| Letta | LoCoMo accuracy | ~66% on hardest category **(unverified)** | hybrid | partial — some examples in repo | LoCoMo is harder than LongMemEval. 66% on multi-hop is arguably more impressive than 96% on single-hop R@5. |
| Mastra | LongMemEval R@5 | 94.87% **(unverified)** | hybrid + GPT | partial | Mid-2024 post, not sure if maintained. |
| Supermemory | LongMemEval | ~99% **(unverified)** | hybrid + strong rerank | not public | Highest claim I've seen. Unreproducible in practice without access to their exact stack. |

### Pattern recognition

- **Every claim above 90% uses a hybrid mode with an LLM reranker
  or an LLM-assisted ingest.** The raw-retrieval ceiling on
  LongMemEval appears to be in the mid-90s, and past that point
  you're paying a token budget for the rerank.
- **"Hybrid" hides a lot.** A hybrid number includes an LLM call
  that the user pays for at query time. Raw numbers are cheaper
  per query but often lower. Any honest comparison must label
  the mode.
- **Confidence intervals are essentially absent.** Almost every
  published number is a single point estimate on a specific
  split, without a bootstrap or repeated-run variance. A 94.87%
  on n=500 with no CI is compatible with a true score anywhere
  in roughly [92%, 97%], and nobody shows the spread.
- **Reproducibility varies from "run the script" to "trust us."**
  mempalace publishes their runner and a community member
  reproduced it. Most others do not and cannot be verified by a
  stranger.
- **The mempalace correction note is the public template for
  honesty.** When they discovered internal errors within 48 hours
  of launch, they posted a dated note retracting specific claims
  while preserving the underlying numbers that survived. engram's
  `RESULTS.md` discipline was written on that template.

### My honest read, distilled

> **LongMemEval is a real benchmark with a real ceiling around 96% in
> raw mode, 100% with paid hybrid reranking. Everything else in the
> wild is harder to verify and more dependent on the specific stack
> the publishing team used.** A responsible new entrant reports raw
> mode with a CI, optionally reports hybrid mode with the exact LLM
> and prompt disclosed, and never reports both in the same headline.

---

## 3. Engram's benchmark surfaces

Two directories under `experiments/`, deliberately separated.

### 3.1 `experiments/retrieval/` — absolute positioning

**Answers:** *"Given a fixed haystack and a fixed query, does
engram surface the right chunk relative to what other systems
publish?"*

**Rules:**
- Every sub-directory is a public benchmark (LongMemEval today,
  LoCoMo next).
- Every runner is in the repo at the commit SHA of any result it
  published.
- Every result row in `RESULTS.md` includes: date, commit SHA,
  subset, n, R@k with 95% bootstrap CI, API calls per query,
  wall-clock cost, and a notes column with anything that would
  help a reader reproduce or interpret the number.
- Results are reported in **both raw and hybrid modes separately**
  when both exist. If engram has only one mode, we say so.
- A row that turns out to be wrong is struck through and a
  corrected row is added beneath with a link to the commit that
  caused the correction. The old row is never removed.
- **No targeted fixes** — we do not rewrite a scenario, tune a
  question, or reword a query to make a number go up. If a
  benchmark question is genuinely ambiguous and we think the
  benchmark is wrong, we file an upstream issue and leave
  engram's number alone.

**What this surface does NOT measure:** whether engram's loop
adds value. A system that does 96% on LongMemEval can still be
useless for a live agent because LongMemEval doesn't exercise
live-stream decisions (when to write, when to forget, when to
consolidate). For that, see `loop_quality/`.

### 3.2 `experiments/loop_quality/` — engram's actual design bar

**Answers:** *"Does engram's decision loop add value over raw
vstash on content that looks like what an agent actually lives
through?"*

**Rules:**
- Every scenario is a JSON fixture with events (text + topic
  label) and queries (question + expected_topic + optional
  expect_contains).
- Scenarios come in three flavors, all kept at 100% pass_rate and
  100% purity as the current bar:
  1. **Synthetic control** — vocabulary is clean, the embedder
     cooperates. Regression guard against "we broke the normal
     case."
  2. **Synthetic borderline** — deliberately has overlapping
     cross-topic cosines near the threshold. Tests the loop's
     robustness to imperfect consolidation.
  3. **Real organic** — snapshot from a real vstash (Jay's, for
     now). Content nobody curated for engram. The only surface
     that catches "tests lying by construction."
- **Every new decider is evaluated on every scenario before
  landing**, via the parametrized
  `test_runner_completes_on_every_scenario[<stem>]` in
  `tests/test_loop_quality.py`. A commit that passes only some
  scenarios does not merge.
- **Coverage is a means, not an end** — what users feel is
  pass_rate and purity. A change that drops coverage on one
  scenario but holds pass_rate and purity everywhere is
  acceptable; the trade-off is recorded in RESULTS.md.

**What this surface does NOT measure:** absolute positioning
against other systems. No scenario here matches any published
benchmark. The scenarios are for engram's own safety net, not for
competitive claims.

### 3.3 Why two surfaces, not one

Collapsing the two would break each one:

- If loop_quality scenarios were in the "retrieval" bucket, a
  100% would be misread as a LongMemEval-style claim, which it
  isn't.
- If LongMemEval were in the "loop_quality" bucket, the 100%
  target would be unreachable (it's a retrieval benchmark, not a
  loop benchmark) and the safety net would turn red forever.

The scoreboards for "where are we relative to claims" and "is the
loop doing its job" measure different things. Keep them apart.

---

## 4. Eligibility criteria for new public benchmarks

A public benchmark only enters `experiments/retrieval/` if it
meets all of the following:

1. **Dataset is available.** License allows use, download is
   mechanical, reproducible cache path is documented.
2. **Metric has a known range.** "Accuracy on a judge model" only
   counts if we specify the judge model version (pinning is
   non-negotiable — a drift in the judge invalidates the number).
3. **Runner is cheap enough** to run overnight on a laptop. Not
   "5 minutes" — that was always aspirational for LongMemEval
   full — but "one person, one night, one laptop, reproducible."
4. **At least one competitor has published a number on it** that
   we can compare against. A benchmark where we'd be the only
   participant is interesting but doesn't serve the
   "absolute-positioning" job of this surface.
5. **The benchmark measures retrieval-or-QA quality**, not
   agent-loop quality. Loop-quality tests belong in
   `loop_quality/`.
6. **We can afford wall-clock + API cost** to run it at least
   once in raw mode. If hybrid mode requires a paid API we don't
   have, raw mode alone is enough.

Benchmarks that *look* interesting but fail any of the criteria
above go into a "`maybe_later.md`" note with the reason, so we
don't re-evaluate them from scratch every quarter.

---

## 5. Rules of engagement for reporting

These apply to every row that lands in any `RESULTS.md` under
`experiments/retrieval/`. They are the operational form of
CONSTITUTION §9 ("empirical first") and the honesty discipline we
put in writing after reading mempalace's correction note.

1. **Every row has a commit SHA.** Not a tag, not a branch, the
   SHA. That's the state of engram when the number was measured.
2. **Every row has a date.** UTC, not local. Makes the cross-row
   timeline readable.
3. **Every row has an `n` and a confidence interval**, unless
   it's a sanity row explicitly marked as such. A point estimate
   without a CI is only published if the table says "sanity".
4. **The mode is labeled.** `raw`, `hybrid`, `hybrid+<model>`,
   `with_rerank`, whatever. The reader must be able to tell at a
   glance whether an LLM was in the path.
5. **API calls per query are counted.** Engram's baseline is
   always zero. Competitor numbers we cite should be quoted with
   their reported cost when known.
6. **Wall-clock cost is recorded** separately from the metric.
   Not part of the R@k, but alongside it so a reader can compare
   "how much did this number cost to produce" across rows.
7. **Corrections are append-only.** A wrong row gets a
   strikethrough and a new row beneath it with a commit link to
   the correction. We do not silently edit the old row.
8. **We do not report sample runs as full runs.** If n is 20 on
   a 500-question dataset, the row says `n=20 (sample)` and does
   not appear in any headline summary.
9. **We do not compare across modes in the same cell.** If engram
   raw is 85% and mempalace hybrid is 100%, those are two rows,
   not "engram 85% vs mempalace 100%". The table structure
   enforces this.
10. **The runner that produced the number must be in the repo at
    the same commit SHA.** No "the number came from a script I
    have locally."

---

## 6. Phased roadmap

### Phase A — LongMemEval full run (immediate prerequisite)
- Complete the n=500 run of `longmemeval_s_cleaned` in raw mode.
- Report the number with a bootstrap CI.
- Compare absolute position against the 96.6% mempalace claim in
  the notes column, without putting it in the headline.
- Wall-clock expected ~6h per baseline on current hardware.
- Deliverable: one row in `experiments/retrieval/longmemeval/
  RESULTS.md` that can be cited as "engram's current LongMemEval
  R@5 raw-mode score."
- **Blocker:** willingness to run an overnight job. No code
  changes needed; `experiments/retrieval/longmemeval/runner.py`
  is already complete.

### Phase B — LoCoMo runner (next public benchmark)
- Build `experiments/retrieval/locomo/` with dataset loader,
  runner, and a judge-model adapter.
- First decision in Phase B is the judge: Ollama-backed (local,
  free, slower, smaller model) or user-pluggable via
  `ENGRAM_JUDGE_MODEL`.
- Report on the same table shape as LongMemEval: date, commit,
  n, category breakdown, accuracy with CI, judge model pinned.
- Expected output: engram's first loop-shaped benchmark result,
  since LoCoMo stresses multi-hop temporal integration.
- Honest expectation: engram's consolidation with embedding_v1 +
  complete-link will **probably underperform LLM-consolidation
  systems on LoCoMo's adversarial category**. That would be real
  evidence that the loop needs a LLM-based consolidator tier, and
  is the scenario we said we'd wait for.

### Phase C — LoopQuality expansion (parallel with B)
- Add at least one real-content scenario outside engram's own
  design discussions. Candidates from Jay's vstash: perf
  migration notes, daily reviews, Kafka meeting threads,
  MedLocal hackathon logs. Each becomes its own `*.json` fixture
  with topic labels by honest title-reading.
- Goal: three real-content scenarios, each covering a different
  topic distribution shape. The current `jay_vstash_2026_04_09_
  snapshot` is dense and meta-heavy; add a scenario that's
  more evenly distributed, and one that's dominated by a single
  topic with many facets.
- Every new scenario must enter the parametrized pytest smoke
  test on the same commit.

### Phase D — MemBench / MemoryBank evaluation (optional)
- Re-read the MemBench and MemoryBank papers, decide if either
  meets Section 4 criteria.
- If yes, add runner. If no, write a one-paragraph note in
  `experiments/retrieval/maybe_later.md` explaining which
  criterion failed.

### Phase E — Internal procedural memory benchmark (post-v1)
- Once procedural memory is implemented (CONSTITUTION §5.3),
  build a task-shaped scenario suite under `loop_quality/
  procedural/`. Events are tool calls with outcomes; queries are
  "have we done this before, and did it work?"
- This is the only benchmark that will stress the fourth memory
  layer from CONSTITUTION §5, and it has no public analog — so
  it lives in `loop_quality/`, not `retrieval/`.

---

## 7. What no benchmark can tell us

Writing this list down because the benchmark surface attracts
focus disproportionate to its value, and the things it cannot
answer are the things that matter most to engram's actual users.

1. **Does engram serve Jay day-to-day?** Answered by: using it in
   Claude Code via the CLI or MCP, watching whether it accumulates
   facts that shortcut real work. No public benchmark touches
   this.
2. **Is engram's audit log useful in practice?** Answered by:
   querying `engram audit <reason>` when something goes wrong and
   seeing if the answer is there. Not a quantitative surface.
3. **Does the tombstone design survive long-term?** Answered by:
   running a real store for months and attempting an unforget
   after 60 days. Requires calendar time, not benchmarks.
4. **Are the defaults right for Spanish / multilingual content?**
   All of engram's calibration so far is on English. The jay
   vstash snapshot has some Spanish (MedLocal demos in es) but
   the scenarios are English. Multilingual calibration is a
   separate empirical question that needs separate scenarios.
5. **Does engram's write policy avoid noise?** The only honest
   way to measure this is to compare an episodic layer filtered
   by `HeuristicWriteDecider` against an unfiltered one over
   real use, and ask "is the filtered version better to search
   against in month 3?" That's a longitudinal question.

Benchmarks are one lever. They're the loudest lever, because
they produce numbers that feel comparable. But the quiet levers
— daily usage, audit log inspection, tombstone recovery, multilingual
scaling, longitudinal drift — are where engram will actually
prove or disprove itself. The benchmark surface exists to give us
absolute positioning against loud claims, not to replace the
quiet evidence.

---

## 8. Open questions (decide before Phase B)

1. **Judge model for LoCoMo.** Ollama-backed default, or
   user-provided? If Ollama, which model? I lean toward
   `llama3.1:8b-instruct` as the default because it's small
   enough to run locally and strong enough to judge open-ended
   conversational answers. The exact version gets pinned in
   `experiments/retrieval/locomo/README.md`.
2. **Hybrid-mode support for LongMemEval.** Does engram add a
   reranker option to compete with mempalace's "100% with Haiku"?
   If yes, where does the reranker live — in engram, or as a
   configurable step in the LongMemEval runner only? My lean:
   add `engram.rerank` as a module with pluggable strategies,
   default `NoRerank` (raw mode). Rerankers are bolt-on, not
   part of the loop proper.
3. **Do we spend time reproducing competitor numbers ourselves?**
   Option A: only measure engram. Option B: also run mempalace /
   Mem0 / Letta on the same hardware to produce a side-by-side.
   (B) is more honest but costs wall-clock and setup pain. My
   lean: (A) for v1, add (B) to Phase B as a one-time
   verification if engram's numbers are surprising either up or
   down.
4. **Public leaderboard?** Does engram publish a page that shows
   these numbers next to the competitive claims, or does it
   keep them in RESULTS.md and let users navigate the repo? My
   lean: no public leaderboard yet. A leaderboard is a
   commitment to maintain numbers across commits, and engram is
   pre-v0.1. RESULTS.md with SHAs is enough.

---

## 9. Related docs

- `../CONSTITUTION.md` §9 — the empirical bar rule ("no benchmark,
  no ship") this doc operationalizes.
- `retrieval/README.md` — what retrieval/ is and isn't.
- `loop_quality/README.md` — what loop_quality/ is and isn't.
- `../notes/prior-art.md` — the mempalace retrospective that
  taught engram what a correction note looks like.

---

*Working doc. Next update when Phase A lands a row, or when a
Section 2 claim is verified or disproven, or when a Section 3
rule gets bent and we want to record the bending.*
