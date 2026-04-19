# Midloop pilot report -- 3 protocols (2026-04-19)

## Stack
- cases: Gemini 2.5 Flash via google.genai
- responses: lmstudio gemma-4-e4b-it-mlx via OpenAI-compatible API
- aligner: fastembed BAAI/bge-small-en-v1.5

## Summary
- protocols: 3 (pneumonia-imci.md, dengue-who.md, anemia-who.md)
- cases: 15
- divergent regions: 46
- semantic drops: 2 (4.3%)
- intervention labels: 44 (95.7%)

## Per-protocol breakdown
| protocol | cases | div | drop | intv |
|---|---:|---:|---:|---:|
| anemia-who | 5 | 18 | 1 | 17 |
| dengue-who | 5 | 16 | 1 | 15 |
| pneumonia-imci | 5 | 12 | 0 | 12 |

## Cosine similarity distribution
### Semantic drops (above-threshold paraphrases)
- n=2, min=0.867, median=0.882, mean=0.882, max=0.898
### Interventions (below-threshold real divergences)
- n=39, min=0.407, median=0.567, mean=0.577, max=0.808

## Sample interventions (first 5 cases)
- [pneumonia-imci__case_000] cos=0.528  truth=`The child has Pneumonia. Administer oral amoxicillin 250 mg twice daily` -> model=`Monitor respiratory rate closely`
- [pneumonia-imci__case_000] cos=0.590  truth=`5 days. Advise follow-up in 3 days.` -> model=`any worsening tachypnea or retractions. Continue to assess feeding tolerance and`
- [pneumonia-imci__case_001] cos=0.498  truth=`The child has Severe Pneumonia or Very Severe Disease due to being lethargic. Re` -> model=`Administer`
- [pneumonia-imci__case_001] cos=0.628  truth=`500 mg before referral.` -> model=`suspension 20 mg/kg/day for seven days. Monitor closely for signs of worsening r`
- [pneumonia-imci__case_002] cos=0.580  truth=`The infant has Pneumonia. Refer urgently to the hospital. Before referral, admin` -> model=`Assess for signs of respiratory distress, including nasal flaring`
- [pneumonia-imci__case_002] cos=0.476  truth=`IM gentamicin 7.5 mg/kg. Keep the child warm` -> model=`retractions. Maintain close observation`
- [pneumonia-imci__case_002] cos=0.587  truth=`advise frequent breastfeeding.` -> model=`monitor vital signs frequently.`
- [pneumonia-imci__case_003] cos=0.505  truth=`The child is improving. Advise` -> model=`Continue amoxicillin 250 mg twice daily for`
- [pneumonia-imci__case_003] cos=0.516  truth=`caregiver to complete the` -> model=`prescribed`
- [pneumonia-imci__case_003] cos=0.504  truth=`5 days of oral amoxicillin.` -> model=`course. Monitor breathing rate and temperature closely at each visit.`
- [pneumonia-imci__case_004] cos=0.486  truth=`Given the child's weight is 4.5 kg` -> model=`Start empiric treatment with Amoxicillin suspension 50 mg/kg/day divided into th`
- [pneumonia-imci__case_004] cos=0.501  truth=`age is unknown, treat as a young infant. Refer urgently to the hospital. Before ` -> model=`follow up in 48-72 hours for clinical improvement.`

## Sample semantic drops (first 5 cases)