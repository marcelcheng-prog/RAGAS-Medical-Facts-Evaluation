# Workspace Instructions

This is a **RAGAS Medical Facts & SOAP Evaluation Framework** for testing RAGFlow agents that extract structured medical information from German doctor-patient transcripts.

## Project Layout

- `medical_facts_evaluation/` — Core Python package with CLI, evaluators, clients, metrics, models, reporters
- `app.py` — Streamlit web UI for interactive evaluation, favorites, and ranking
- `data/` — Persistent storage for favorites and rankings (JSON)
- `.github/skills/` — Copilot skills for medical-facts-eval and soap-eval

## Two Evaluation Modes

1. **Medical Facts** (`--mode medical_facts`): Medication extraction, vitals, symptoms, RAGAS scoring
2. **SOAP** (`--mode soap`): Three-tier scoring (lexical 35% + GPT-semantic 30% + RAGAS 35%)

## Running Evaluations

```bash
# CLI from RAGAS_Medical_Facts_Agent_refactor/
python -m medical_facts_evaluation --mode medical_facts --agent-a <ID> --verbose
python -m medical_facts_evaluation --mode soap --agent-a <ID> --verbose-soap

# Web UI
python -m streamlit run app.py
```

## Key Agent IDs

- Production Medical Facts: `e1a25a64fdc611f0b3cb4afd40f7103b`
- SOAP v1: `be95d9821ef211f194964348756e437e`
- SOAP v2: `f44320e61ef011f194964348756e437e`

## Environment

Requires `.env` with `OPENAI_API_KEY`, `MEDICAL_FACTS_AUTH_TOKEN`, `RAGFLOW_BASE_URL`.

## Coding Conventions

- Python 3.10+ with type hints
- Dataclasses for models
- German medical terminology in test cases and ground truth
- Rich library for CLI output, Streamlit for web UI
