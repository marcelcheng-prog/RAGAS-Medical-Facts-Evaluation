# RAGAS Medical Facts Evaluation Framework

## Project Overview

This is an evaluation framework for testing **Medical Facts extraction agents** using the RAGAS (Retrieval Augmented Generation Assessment) framework. The agents extract structured medical information from German doctor-patient transcripts.

> **Scope**: Currently evaluates Medical Facts Extraction only. SOAP note generation evaluation is not yet supported.

## Tech Stack

- **Python 3.10+**
- **RAGAS** - LLM evaluation framework for Faithfulness, Context Recall, Answer Relevancy
- **LangChain + OpenAI** - Powers RAGAS evaluation
- **Rich** - Terminal output formatting
- **Pydantic** - Data validation
- **python-dotenv** - Environment configuration

## Architecture

```
medical_facts_evaluation/
├── __main__.py          # Entry point
├── cli.py               # Argument parsing
├── evaluator.py         # Main orchestrator
├── clients/             # External API clients
├── config/              # Settings and thresholds
├── metrics/             # Evaluation logic (medications, vitals, symptoms)
├── models/              # Data structures
├── reporters/           # Output formatting (console, JSON)
└── test_cases/          # Ground truth JSON files
```

## Key Commands

```bash
# Run evaluation with default agent
python -m medical_facts_evaluation --verbose

# Test specific agent
python -m medical_facts_evaluation --agent-a e1a25a64fdc611f0b3cb4afd40f7103b

# Compare two agents
python -m medical_facts_evaluation --compare --agent-a <id1> --agent-b <id2>

# Use specific test case
python -m medical_facts_evaluation --test-case medical_facts_evaluation/test_cases/hausarzt.json

# Run consistency check
python -m medical_facts_evaluation --iterations 5
```

## Environment Variables

Required in `.env`:
- `OPENAI_API_KEY` - For RAGAS evaluation
- `MEDICAL_FACTS_AUTH_TOKEN` - RAGFlow API auth
- `RAGFLOW_BASE_URL` - RAGFlow endpoint (default: http://172.17.16.150/api/v1/agents_openai)

## What Gets Evaluated

1. **Medication Extraction** - Precision, Recall, F1, Hallucination detection
2. **Vital Signs** - Blood pressure, heart rate, etc.
3. **Symptoms** - Patient-reported symptoms
4. **RAGAS Metrics** - Faithfulness, Context Recall, Answer Relevancy

## Production Agent

Agent ID: `e1a25a64fdc611f0b3cb4afd40f7103b`
- 87-100% Medication Precision
- 80-87% Medication Recall
- 95-100% Faithfulness
- ~97% Quality Score

## Test Cases

| Test Case | Description | Key Features |
|-----------|-------------|--------------|
| `hausarzt.json` | Complete GP consultation | Comprehensive, longest transcript |
| `diabetes.json` | Diabetes therapy change | DPP-4 → Ozempic transition |
| `michael_mueller.json` | Diabetes & back pain | New, stopped, refused medications |
| `diabetes_hypertonie.json` | Diabetes + Hypertension | **Family history** evaluation |
| `medikamentenreview_polypharmazie.json` | Elderly polypharmacy | **Swiss-German ASR errors** |

## Test Case Format

Test cases are JSON files in `test_cases/` containing:
- `transcript` - German doctor-patient conversation
- `ground_truth` - Hand-labeled expected extractions
- `metadata` - Test case info

## Development Workflow

1. Create/modify test case JSON with ground truth
2. Run evaluation: `python -m medical_facts_evaluation --test-case <path> -v`
3. Review results in terminal and `results/` directory
4. Adjust agent prompts or thresholds as needed

## Important Files

- [evaluator.py](medical_facts_evaluation/evaluator.py) - Main evaluation orchestrator
- [config/thresholds.py](medical_facts_evaluation/config/thresholds.py) - Pass/fail thresholds
- [metrics/medication.py](medical_facts_evaluation/metrics/medication.py) - Medication scoring logic
- [models/ground_truth.py](medical_facts_evaluation/models/ground_truth.py) - Expected data structures
