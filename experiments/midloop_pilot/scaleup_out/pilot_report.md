# Midloop pilot report -- 75 protocols (2026-04-19)

## Stack
- cases: Gemini 2.5 Flash via google.genai
- responses: lmstudio gemma-4-e4b-it-mlx via OpenAI-compatible API
- aligner: fastembed BAAI/bge-small-en-v1.5

## Summary
- protocols: 3 (pneumonia-imci.md, dengue-who.md, anemia-who.md)
- cases: 375
- divergent regions: 1115
- semantic drops: 18 (1.6%)
- intervention labels: 1097 (98.4%)

## Per-protocol breakdown
| protocol | cases | div | drop | intv |
|---|---:|---:|---:|---:|
| acute-abdomen-who | 5 | 15 | 0 | 15 |
| allergic-reactions-who | 5 | 11 | 0 | 11 |
| anaphylaxis-who | 5 | 15 | 0 | 15 |
| anemia-who | 5 | 17 | 0 | 17 |
| asthma-who | 5 | 12 | 0 | 12 |
| breastfeeding-who | 5 | 15 | 0 | 15 |
| burns-who | 5 | 21 | 1 | 20 |
| cardiovascular-emergency-who | 5 | 17 | 0 | 17 |
| chest-pain-who | 5 | 13 | 0 | 13 |
| child-protection-who | 5 | 21 | 0 | 21 |
| childbirth-emergency-who | 5 | 11 | 0 | 11 |
| choking-first-aid | 5 | 12 | 0 | 12 |
| cholestasis-pregnancy-who | 5 | 17 | 0 | 17 |
| depression-mhgap-who | 5 | 13 | 0 | 13 |
| deworming-who | 5 | 15 | 0 | 15 |
| diabetes-who | 5 | 14 | 2 | 12 |
| diarrhea-adults-who | 5 | 15 | 0 | 15 |
| diarrhea-imci | 5 | 16 | 0 | 16 |
| drowning-cpr-who | 5 | 20 | 0 | 20 |
| drug-allergies-alternatives | 5 | 11 | 0 | 11 |
| drug-rash-severity-who | 5 | 15 | 0 | 15 |
| drug-safety-age-pregnancy-who | 5 | 11 | 0 | 11 |
| ear-infections-imci | 5 | 20 | 0 | 20 |
| eczema-dermatitis-who | 5 | 16 | 0 | 16 |
| epilepsy-mhgap-who | 5 | 17 | 0 | 17 |
| eye-infections-who | 5 | 16 | 0 | 16 |
| family-planning-who | 5 | 13 | 0 | 13 |
| fever-assessment-imci | 5 | 14 | 1 | 13 |
| fever-differential-tropical | 5 | 11 | 0 | 11 |
| foreign-body-aspiration-who | 5 | 16 | 0 | 16 |
| fractures-first-aid | 5 | 22 | 0 | 22 |
| gestational-diabetes-who | 5 | 15 | 1 | 14 |
| heat-illness-who | 5 | 15 | 0 | 15 |
| hepatitis-who | 5 | 16 | 0 | 16 |
| herpes-shingles-who | 5 | 10 | 0 | 10 |
| hiv-pep-who | 5 | 15 | 0 | 15 |
| hiv-who | 5 | 16 | 0 | 16 |
| hypertension-pregnancy-who | 5 | 18 | 0 | 18 |
| hypertension-who | 5 | 14 | 0 | 14 |
| immunization-who | 5 | 10 | 0 | 10 |
| infant-safety-home-remedies-who | 5 | 17 | 1 | 16 |
| malaria-pregnancy-who | 5 | 19 | 0 | 19 |
| malaria-who | 5 | 10 | 1 | 9 |
| malnutrition-adults-who | 5 | 13 | 0 | 13 |
| measles-who | 5 | 12 | 0 | 12 |
| medication-side-effects-expected-vs-alarm-who | 5 | 16 | 0 | 16 |
| medications-pregnancy-safety-who | 5 | 9 | 0 | 9 |
| meningitis-who | 5 | 15 | 0 | 15 |
| meningococcemia-who | 5 | 13 | 0 | 13 |
| newborn-care-who | 5 | 9 | 0 | 9 |
| newborn-resuscitation-who | 5 | 9 | 1 | 8 |
| nutrition-who | 5 | 13 | 0 | 13 |
| oral-health-who | 5 | 16 | 0 | 16 |
| pneumonia-imci | 5 | 15 | 0 | 15 |
| poisoning-general-who | 5 | 20 | 1 | 19 |
| poisoning-heavy-metals-who | 5 | 15 | 0 | 15 |
| poisoning-organophosphate-who | 5 | 18 | 1 | 17 |
| preeclampsia-who | 5 | 9 | 0 | 9 |
| pregnancy-antenatal-who | 5 | 14 | 0 | 14 |
| rabies-pep-who | 5 | 22 | 0 | 22 |
| rabies-who | 5 | 13 | 0 | 13 |
| scabies-who | 5 | 17 | 0 | 17 |
| shock-who | 5 | 13 | 0 | 13 |
| skin-infections-who | 5 | 8 | 1 | 7 |
| snakebite-who | 5 | 18 | 0 | 18 |
| sti-who | 5 | 24 | 3 | 21 |
| stroke-who | 5 | 12 | 2 | 10 |
| tetanus-who | 5 | 19 | 0 | 19 |
| tropical-skin-ntds | 5 | 9 | 0 | 9 |
| tuberculosis-who | 5 | 13 | 0 | 13 |
| typhoid-who | 5 | 13 | 1 | 12 |
| uti-who | 5 | 18 | 0 | 18 |
| vaccine-preventable-diseases-who | 5 | 15 | 0 | 15 |
| varicella-chickenpox-who | 5 | 17 | 0 | 17 |
| wound-care-who | 5 | 21 | 1 | 20 |

## Cosine similarity distribution
### Semantic drops (above-threshold paraphrases)
- n=18, min=0.860, median=0.988, mean=0.954, max=1.000
### Interventions (below-threshold real divergences)
- n=987, min=0.362, median=0.567, mean=0.573, max=0.849

## Sample interventions (first 5 cases)
- [acute-abdomen-who__case_000] cos=0.637  truth=`REFER URGENTLY` -> model=`Administer IV fluids and obtain urgent abdominal ultrasound`
- [acute-abdomen-who__case_000] cos=0.525  truth=`a hospital with` -> model=`confirm appendicitis. Start broad-spectrum antibiotics immediately pending`
- [acute-abdomen-who__case_000] cos=0.465  truth=`capacity. Do NOT give antibiotics or NSAIDs. Paracetamol is acceptable for pain ` -> model=`consultation.`
- [acute-abdomen-who__case_001] cos=0.512  truth=`This is an emergency suggesting intussusception. REFER URGENTLY` -> model=`Administer oral iron supplements`
- [acute-abdomen-who__case_001] cos=0.548  truth=`a hospital with surgical capacity. Do NOT give food` -> model=`the infant. Continue close monitoring for signs of worsening anemia`
- [acute-abdomen-who__case_001] cos=0.533  truth=`water (NPO).` -> model=`occult blood loss.`
- [acute-abdomen-who__case_002] cos=0.610  truth=`This is an emergency suggesting volvulus. REFER URGENTLY` -> model=`Administer intravenous fluids`
- [acute-abdomen-who__case_002] cos=0.549  truth=`IMMEDIATELY` -> model=`monitor vital signs closely. Obtain abdominal ultrasound`
- [acute-abdomen-who__case_002] cos=0.446  truth=`a hospital with surgical capacity. Do NOT give food or water (NPO). Monitor` -> model=`evaluate`
- [acute-abdomen-who__case_003] cos=0.665  truth=`REFER URGENTLY` -> model=`Obtain urgent abdominal ultrasound`
- [acute-abdomen-who__case_003] cos=0.530  truth=`a hospital with surgical capacity due to severe localized pain, guarding,` -> model=`evaluate for appendicitis. Administer IV fluids`
- [acute-abdomen-who__case_003] cos=0.454  truth=`rebound tenderness. Do NOT give her water or any food (NPO). Paracetamol is acce` -> model=`monitor vital signs closely.`
- [acute-abdomen-who__case_004] cos=0.642  truth=`REFER URGENTLY` -> model=`Obtain immediate vital signs, including blood pressure and heart rate. Prepare f`
- [acute-abdomen-who__case_004] cos=0.560  truth=`a hospital with surgical capacity. Do NOT give antibiotics or NSAIDs. Do NOT giv` -> model=`suspected acute abdomen.`