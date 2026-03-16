# RAGAS Medical Facts & SOAP Evaluation Framework

## Project Overview

This is an evaluation framework for testing **Medical Facts extraction agents** and **SOAP note generation agents** using the RAGAS (Retrieval Augmented Generation Assessment) framework. The agents extract structured medical information from German doctor-patient transcripts deployed on RAGFlow.

## Two Evaluation Modes

### Medical Facts Mode (`--mode medical_facts`)
Evaluates structured medical data extraction: medications, vitals, symptoms, family history.
Uses RAGAS (Faithfulness, Context Recall, Answer Relevancy) + custom metrics (Precision, Recall, F1, Hallucination detection).

### SOAP Mode (`--mode soap`)
Evaluates SOAP note generation against gold standard references.
Three-tier scoring: Lexical token-F1 (35%) + GPT semantic judge (30%) + RAGAS context recall (35%).
Section aliases map `subjective/objective/assessment/plan` → `S/O/A/P`.

## Web UI
`app.py` — Streamlit dashboard for interactive evaluation, agent favorites, and ranking leaderboard.
Launch: `streamlit run app.py`

## Tech Stack

- **Python 3.10+**
- **RAGAS** - LLM evaluation framework for Faithfulness, Context Recall, Answer Relevancy
- **LangChain + OpenAI** - Powers RAGAS evaluation and GPT semantic scoring
- **Rich** - Terminal output formatting
- **Streamlit** - Web UI dashboard
- **Pydantic** - Data validation
- **python-dotenv** - Environment configuration

## Architecture

```
medical_facts_evaluation/
├── __main__.py          # Entry point
├── cli.py               # Argument parsing (medical_facts + soap modes)
├── evaluator.py         # Medical facts orchestrator
├── clients/             # External API clients (RAGFlow, RAGAS)
├── config/              # Settings and thresholds
├── evaluators/          # SOAP evaluator (3-tier scoring)
├── formatters/          # SOAP formatter (facts → S/O/A/P)
├── metrics/             # Evaluation logic (medications, vitals, symptoms, safety)
├── models/              # Data structures (GroundTruth, EvaluationResult, TestCase)
├── reporters/           # Output formatting (console, JSON)
├── test_cases/          # Ground truth JSON files (13+)
├── gold_soap/           # Gold SOAP references
└── prompts/             # SOAP formatter prompt
app.py                   # Streamlit web UI
```

## Key Commands

```bash
# Medical facts evaluation
python -m medical_facts_evaluation --verbose
python -m medical_facts_evaluation --compare --agent-a <id1> --agent-b <id2>
python -m medical_facts_evaluation --test-case medical_facts_evaluation/test_cases/hausarzt.json

# SOAP evaluation  
python -m medical_facts_evaluation --mode soap --agent-a <id> --verbose-soap
python -m medical_facts_evaluation --mode soap --agent-a <id> --all-test-cases --verbose-soap

# Web UI
streamlit run app.py
```

## Environment Variables

Required in `.env`:
- `OPENAI_API_KEY` - For RAGAS + GPT semantic scoring
- `MEDICAL_FACTS_AUTH_TOKEN` - RAGFlow API auth
- `RAGFLOW_BASE_URL` - RAGFlow endpoint (default: http://172.17.16.150/api/v1/agents_openai)

## Key Agent IDs

| Agent | ID | Type |
|-------|----|------|
| Production Medical Facts | `e1a25a64fdc611f0b3cb4afd40f7103b` | Medical Facts |
| Default Medical Facts | `df4cb87efd2011f0b3234afd40f7103b` | Medical Facts |
| SOAP v1 | `be95d9821ef211f194964348756e437e` | SOAP |
| SOAP v2 | `f44320e61ef011f194964348756e437e` | SOAP |

## What Gets Evaluated

### Medical Facts Mode
1. **Medication Extraction** - Precision, Recall, F1, Hallucination detection
2. **Vital Signs** - Blood pressure, heart rate, etc.
3. **Symptoms** - Patient-reported symptoms
4. **Family History** - RAGAS semantic recall
5. **RAGAS Metrics** - Faithfulness, Context Recall, Answer Relevancy

### SOAP Mode
1. **Structure** - All 4 sections (S/O/A/P) present
2. **Lexical Content** - Token-level F1 per section
3. **GPT Semantic** - Medical equivalence judge per section
4. **RAGAS Semantic** - Context recall per section
5. **Effective Score** - Weighted blend of all three

## Scoring Weights (SOAP)

- Lexical: 35%, GPT Semantic: 30%, RAGAS Semantic: 35%
- Overall: 30% structure + 70% content
- Floor: effective ≥ lexical × 0.8

## Test Cases

13+ German medical transcripts covering:
- Diabetes, hypertension, polypharmacy, pregnancy, orthopedics, psychiatry, gastritis
- ASR transcription errors (Swiss-German)
- Correction recognition (QM-013), noise filtering (QM-015)
- Family history, forbidden medications, laterality

## Development Workflow

1. Create/modify test case JSON with ground truth
2. Run evaluation: `python -m medical_facts_evaluation --test-case <path> -v`
3. Or use web UI: `streamlit run app.py`
4. Review results and adjust agent prompts or thresholds
