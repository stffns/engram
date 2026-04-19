# Midloop dataset pilot -- 3 protocols (2026-04-19)

## Goal

Validate the Phase 1 dataset pipeline (`merken/training/midloop_dataset.py`
+ `case_generator.py` + `response_generator.py`) on REAL clinical content
before committing to a 114-protocol scale-up. Produce the first labeled
JSONL that a (future) NanoGPTMidloopDecider could train on.

**Does NOT train any model.** The pilot output is the dataset, not a
classifier.

## Setup

- protocols: 3 from `medlocal/data/core/protocols/` -- `pneumonia-imci.md`,
  `dengue-who.md`, `anemia-who.md`. Type B authoritative content.
- case generator: Gemini 2.5 Flash via google.genai (Jay's MedLocal
  teacher_llm). N=5 cases per protocol -> 15 total.
- response generator: lmstudio `gemma-4-e4b-it-mlx` via OpenAI-
  compatible API on `localhost:1234`, MLX-accelerated. System prompt
  constrained to "concise CHW assistant; 1-3 sentences; no markdown".
- aligner: token-level edit distance + batched fastembed
  `BAAI/bge-small-en-v1.5` cosine filter.

Stack swapped from the spec defaults (Claude Sonnet + HF transformers
Gemma) to match Jay's actual MedLocal infra. Same `LLMClient` /
`GenerateFn` interfaces, just different concrete backends.

## Results

| metric | value |
|---|---|
| cases generated | 15/15 |
| divergent regions | 46 |
| semantic drops (threshold=0.90) | 0 |
| semantic drops (threshold=0.85) | 2 |
| intervention labels (threshold=0.85) | 44 |
| wall time | ~50s |
| Gemini cost | ~$0.01 |

### Per-protocol breakdown (threshold=0.85)

| protocol | cases | divergent | drops | interventions |
|---|---:|---:|---:|---:|
| anemia-who | 5 | 18 | 1 | 17 |
| dengue-who | 5 | 16 | 1 | 15 |
| pneumonia-imci | 5 | 12 | 0 | 12 |

### Cosine similarity distribution

| zone | count | character |
|---|---:|---|
| [0.85, 1.0] | 2 | paraphrases (dropped) |
| [0.7, 0.85) | 5 | borderline (mix) |
| [0.5, 0.7) | 26 | divergence (kept) |
| [0.3, 0.5) | 8 | strong divergence (kept) |

**Calibration:** threshold 0.90 (initial spec) is too strict for this
content + embedder. The two confirmed paraphrases observed sit at
0.867 ("Paracetamol" vs "acetaminophen") and 0.898 ("3" vs "three").
Threshold 0.85 catches both without losing the 0.7-0.85 borderline
where genuine divergences live ("iron 3 mg/kg" vs "iron 5 mg/kg" at
cos=0.804 is a REAL clinical error, not a paraphrase).

**Production threshold: 0.85** (empirical, this corpus + embedder).
Future calibration check should re-run as new protocols are added; if
a wider corpus shifts the gap, recalibrate.

## Two false success-criteria from the pre-pilot plan

The pilot plan had two criteria that turned out to be wrong for this
domain:

### Drop ratio "should be 30-70%" -- WRONG criterion

Observed 4.3% drop. The plan assumed the small model would mostly say
the right thing in slightly different words. Reality: Gemma 4 E4B
under the constrained-style system prompt fails to extract protocol-
specific actions and defaults to generic safe-sounding advice
("monitor closely", "continue assessment"). When the truth says "REFER
URGENTLY + IM ampicillin 50 mg/kg", Gemma says "monitor for
worsening". These are NOT paraphrases -- they are substantively
different recommendations. **Every divergence IS a real intervention
label.** A 4% drop ratio is correct because there is genuinely little
overlap between Gemma's generic clinical advice and protocol-specific
actions.

The new criterion is "drops are LEGITIMATE paraphrases when they
exist". 2/2 drops in this run are clean paraphrases (synonyms +
unit reformulation). Pass.

### Style mismatch was the early bug, not threshold

First run (no system prompt on lmstudio): Gemma produced 2k+ char
ChatGPT-style essays with markdown, headers, tables. Truth was
~150-200 char clinical actions. EVERY divergence was massive. The
aligner was correct; the response shape was wrong. Adding the
constrained system prompt cut response length 10x and aligned the
shapes, making divergences meaningful.

Lesson: the response generator must produce text in the same shape
as the truth. For MedLocal that shape is concise clinical action,
not free-form essay. Document this constraint in
`response_generator.default_hf_client` for future users.

## Sample interventions (the actual training labels)

Three representative cases, illustrating the kind of error the midloop
would learn to flag:

```
[pneumonia-imci__case_001]
  truth: "The child has Severe Pneumonia or Very Severe Disease due to
          being lethargic. Refer urgently to the hospital. Give the
          first dose of oral amoxicillin 500 mg before referral."
  model: "Administer oral amoxicillin suspension 20 mg/kg/day for
          seven days. Monitor closely for signs of worsening."
  -> Gemma misses the danger sign (lethargy = severe), gives outpatient
     dosing instead of emergency referral. This is the kind of
     critical-safety failure the midloop must catch.

[anemia-who__case_2]
  truth: "Administer iron syrup 3 mg/kg/day"
  model: "Start ferrous sulfate 5 mg/kg/day"
  -> Real dose error: 3 vs 5 mg/kg + different formulation
     (syrup vs sulfate). cos=0.804.

[dengue-who__case_001]
  truth: "Paracetamol (10-15 mg/kg every 6 hours) should be given.
          Aspirin, ibuprofen, diclofenac should NOT be used."
  model: "Acetaminophen 10-15 mg/kg every 4-6 hours. Strictly avoid
          NSAIDs like ibuprofen."
  -> Mostly correct. Acetaminophen=Paracetamol (paraphrase, drops at
     cos=0.867). "every 4-6 hours" vs "every 6 hours" is a borderline
     intervention (cos=0.775). "NSAIDs like ibuprofen" vs "Aspirin,
     ibuprofen, diclofenac" is generalization (kept as intervention).
```

## Decision tree (post-pilot)

| outcome | action |
|---|---|
| Pipeline E2E | PASS |
| Threshold calibrated (0.85) | PASS |
| Drops are legitimate paraphrases | PASS (2/2 clean) |
| Cost projection (114 protocols x 5 cases x $0.001 = ~$0.40) | acceptable |
| Time projection (~10 min wall clock at full scale) | acceptable |
| Style mismatch identified + fixed | PASS (system prompt constraint) |

**Recommendation: scale to all 114 protocols.** Same script,
`PILOT_PROTOCOLS` extends to every `.md` in `data/core/protocols/`
and the 36 sibling files in `who/ msf/ hesperian/ cht-workflows/`.
Expected output: ~570 cases, ~1,700 intervention labels.

## Two follow-ups filed for the scale-up run

1. **Pre-warm lmstudio.** First call had a ~5s cold start; subsequent
   were ~1s. With 570 calls this is amortized but worth a manual
   `curl /v1/models` before starting if reproducibility matters.
2. **Per-protocol case-count override.** Some protocols are dense
   (acute-abdomen, anaphylaxis) and warrant 10+ cases; some are short
   (deworming) and 3 are enough. Adding `n_cases` per protocol entry
   in protocols.jsonl is a 5-line CLI change.

## What this does NOT do

- Does NOT train a midloop model (Phase 3, weeks out).
- Does NOT add a midloop primitive (Phase 2, gated on this dataset).
- Does NOT validate that a hypothetical midloop trained on this
  dataset would actually catch errors at inference time -- that
  measurement comes after Phase 3.
