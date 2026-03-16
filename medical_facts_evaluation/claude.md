# Medical Facts Evaluation Package

## Purpose

Core evaluation package that orchestrates the testing of Medical Facts extraction agents and SOAP note generation agents. Compares agent output against hand-labeled ground truth using both custom metrics and RAGAS framework.

## Package Structure

```
medical_facts_evaluation/
├── __init__.py      # Package exports, lazy loading
├── __main__.py      # Entry: python -m medical_facts_evaluation
├── cli.py           # Argument parsing, main() entry point (medical_facts + soap modes)
├── evaluator.py     # MedicalFactsEvaluator orchestrator
├── clients/         # External API communication (RAGFlow, RAGAS)
├── config/          # Settings, thresholds
├── evaluators/      # SOAP evaluator (3-tier scoring)
├── formatters/      # SOAP formatter (facts → S/O/A/P)
├── metrics/         # Evaluation algorithms
├── models/          # Data structures
├── reporters/       # Output formatting
├── test_cases/      # Ground truth JSON files (13+)
├── gold_soap/       # Gold SOAP references for evaluation
└── prompts/         # SOAP formatter prompt template
```

## Entry Points

1. **CLI**: `python -m medical_facts_evaluation [args]`
2. **Programmatic**: `from medical_facts_evaluation import MedicalFactsEvaluator`
3. **Web UI**: `streamlit run app.py` (parent directory)

## Two Evaluation Modes

### Medical Facts Mode (`--mode medical_facts`)
1. Load test case with transcript + ground truth
2. Call RAGFlow agent API to extract facts
3. Parse agent JSON response
4. Run medication evaluation (precision/recall/F1)
5. Run vital signs, symptoms, family history evaluation
6. Run RAGAS evaluation (faithfulness, recall, relevancy)
7. Build `EvaluationResult` with all scores

### SOAP Mode (`--mode soap`)
1. Load test case + gold SOAP sections from `gold_soap/`
2. Call RAGFlow SOAP agent API
3. Parse SOAP sections (JSON or text, with section alias mapping)
4. Score each S/O/A/P section with three layers:
   - Lexical: Token F1 (35%)
   - GPT Semantic: LLM judge for medical equivalence (30%)
   - RAGAS Semantic: Context recall (35%)
5. Build `SoapEvaluationResult` with blended scores

## Main Classes

### `MedicalFactsEvaluator` (evaluator.py)

```python
evaluator = MedicalFactsEvaluator(client, ragas, thresholds, verbose=True)
result = evaluator.evaluate(test_case)
```

### `evaluate_soap_output()` (evaluators/soap_evaluator.py)

```python
result = evaluate_soap_output(
    output=response.content,
    gold_sections=load_gold_sections(case_path, gold_soap_dir),
    openai_client=openai_client,
    ragas_evaluator=ragas_evaluator,
    transcript=test_case.transcript,
    ...
)
```

## Data Flow

### Medical Facts
```
TestCase JSON → MedicalFactsClient → Agent Response
                                          ↓
                              parse_medical_facts()
                                          ↓
                              evaluate_medications/vitals/symptoms
                                          ↓
                              RagasEvaluator.evaluate()
                                          ↓
                              EvaluationResult → ConsoleReporter + JsonReporter
```

### SOAP
```
TestCase JSON → MedicalFactsClient → SOAP Agent Response
                                          ↓
                              extract_soap_sections() (JSON/text/alias mapping)
                                          ↓
                              Gold sections from gold_soap/
                                          ↓
                              _token_f1() + _llm_semantic_section_score() + RAGAS
                                          ↓
                              _combine_section_score() [weighted blend]
                                          ↓
                              SoapEvaluationResult → CLI/UI output
```

## Key Dependencies

- `clients/medical_facts.py` - RAGFlow API client
- `clients/ragas_client.py` - RAGAS evaluation wrapper
- `evaluators/soap_evaluator.py` - SOAP three-tier scoring
- `formatters/soap_formatter.py` - Medical facts → SOAP conversion
- `config/settings.py` - Environment-based configuration
- `config/thresholds.py` - Quality pass/fail thresholds

## Adding New Features

1. **New metric type**: Add module in `metrics/`, import in `metrics/__init__.py`
2. **New ground truth field**: Update `models/ground_truth.py`
3. **New CLI option**: Update `cli.py` parser
4. **New output format**: Add reporter in `reporters/`
5. **New SOAP scoring signal**: Add to `evaluators/soap_evaluator.py` and update `_combine_section_score()`
