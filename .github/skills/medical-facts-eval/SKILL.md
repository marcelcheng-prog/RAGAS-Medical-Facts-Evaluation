---
name: medical-facts-eval
description: 'Run Medical Facts evaluation on RAGFlow agents with verbose output. Use when: evaluating medical facts extraction, running agent comparison, checking medication precision/recall, viewing RAGAS scores, testing agents against German medical transcripts, verbose medical facts output.'
argument-hint: 'Agent ID and optional test case name (e.g., "e1a25a64 hausarzt")'
---

# Medical Facts Evaluation Skill

## When to Use

- User wants to evaluate a Medical Facts extraction agent
- User wants to compare two agents on a test case
- User asks about medication precision, recall, RAGAS scores
- User wants verbose output for medical facts mode
- User mentions agent IDs and medical facts in the same request

## Known Agent IDs

| Agent | ID | Notes |
|-------|----|-------|
| Production | `e1a25a64fdc611f0b3cb4afd40f7103b` | Best quality |
| Default | `df4cb87efd2011f0b3234afd40f7103b` | Default in settings |
| Parallel | `a6ef1157028011f180502a3cdbc575c7` | Experimental, lower quality |

## Available Test Cases

Located in `medical_facts_evaluation/test_cases/`:
- `michael_mueller.json` - Diabetes & back pain (default)
- `hausarzt.json` - Complete GP consultation (longest)
- `diabetes.json` - DPP-4 → Ozempic transition
- `diabetes_hypertonie.json` - With family history
- `medikamentenreview_polypharmazie.json` - Swiss-German ASR errors
- `gyn_pregnancy_gestational_diabetes.json` - Pregnancy, 3 allergies
- `ortho_knee_arthrose.json` - Laterality, µg vs mg
- `korrektur_noise_filter.json` - Corrections & smalltalk
- `magenschmerzen_gastritis.json` - Gastritis
- `bauchschmerzen.json` - Abdominal pain
- `psychiatrie_depression_symptome.json` - Depression
- `notfall_brustschmerzen_vitals.json` - Emergency chest pain
- `634.json` - Additional case

## Procedure

### 1. Single Agent Evaluation (verbose)

```bash
cd P:\MCH\Ragas\RAGAS_Medical_Facts_Agent\RAGAS_Medical_Facts_Agent_refactor
python -m medical_facts_evaluation --agent-a <AGENT_ID> --test-case medical_facts_evaluation/test_cases/<TEST_CASE>.json --verbose
```

### 2. Compare Two Agents

```bash
python -m medical_facts_evaluation --compare --agent-a <ID1> --agent-b <ID2> --test-case medical_facts_evaluation/test_cases/<TEST_CASE>.json --verbose
```

### 3. Consistency Check (multiple iterations)

```bash
python -m medical_facts_evaluation --agent-a <AGENT_ID> --iterations 5 --verbose
```

### 4. Development Thresholds (more relaxed)

```bash
python -m medical_facts_evaluation --agent-a <AGENT_ID> --thresholds development --verbose
```

## Output Interpretation

### Key Metrics
- **Faithfulness** (≥90%): Are extracted facts supported by transcript?
- **Context Recall** (≥85%): Were all relevant facts captured?
- **Answer Relevancy** (≥80%): Is output relevant to medical context?
- **Medication Precision** (≥95%): What % of extracted meds are correct?
- **Medication Recall** (≥90%): What % of actual meds were found?
- **Hallucination Score** (≥98%): 1 - (hallucinated meds / total)

### Quality Score
Weighted composite 0-100. Threshold: ≥70 for production pass.

### Exit Codes
- `0` = All thresholds passed
- `1` = One or more thresholds failed

## Working Directory

Always run from: `P:\MCH\Ragas\RAGAS_Medical_Facts_Agent\RAGAS_Medical_Facts_Agent_refactor`
Or use wrapper: `.\run_soap_eval.ps1` from parent directory (also supports medical_facts mode).

## Troubleshooting

- **API Error**: Check `.env` has valid `MEDICAL_FACTS_AUTH_TOKEN` and `RAGFLOW_BASE_URL`
- **RAGAS Error**: Check `OPENAI_API_KEY` is set. GPT-5.x models may need special handling.
- **ModuleNotFoundError**: Ensure you're in the `RAGAS_Medical_Facts_Agent_refactor/` directory
