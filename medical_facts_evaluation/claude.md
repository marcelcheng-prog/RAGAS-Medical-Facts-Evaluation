# Medical Facts Evaluation Package

## Purpose

Core evaluation package that orchestrates the testing of Medical Facts extraction agents. Compares agent output against hand-labeled ground truth using both custom metrics and RAGAS framework.

## Package Structure

```
medical_facts_evaluation/
├── __init__.py      # Package exports, lazy loading
├── __main__.py      # Entry: python -m medical_facts_evaluation
├── cli.py           # Argument parsing, main() entry point
├── evaluator.py     # MedicalFactsEvaluator orchestrator
├── clients/         # External API communication
├── config/          # Settings, thresholds
├── metrics/         # Evaluation algorithms
├── models/          # Data structures
├── reporters/       # Output formatting
└── test_cases/      # Ground truth JSON files
```

## Entry Points

1. **CLI**: `python -m medical_facts_evaluation [args]`
2. **Programmatic**: `from medical_facts_evaluation import MedicalFactsEvaluator`

## Main Classes

### `MedicalFactsEvaluator` (evaluator.py)

The main orchestrator. Workflow:
1. Load test case with transcript + ground truth
2. Call RAGFlow agent API to extract facts
3. Parse agent JSON response
4. Run medication evaluation (precision/recall/F1)
5. Run vital signs and symptoms evaluation
6. Run RAGAS evaluation (faithfulness, recall, relevancy)
7. Build `EvaluationResult` with all scores
8. Print results via `ConsoleReporter`
9. Save JSON via `JsonReporter`

```python
evaluator = MedicalFactsEvaluator(client, ragas, thresholds, verbose=True)
result = evaluator.evaluate(test_case)
```

### Key Functions

- `run_agent_comparison()` - Compare two agents side-by-side
- `create_parser()` - CLI argument definitions

## Data Flow

```
TestCase JSON → MedicalFactsClient → Agent Response
                                          ↓
                              parse_medical_facts()
                                          ↓
                              evaluate_medications()
                              evaluate_vital_signs()
                              evaluate_symptoms()
                                          ↓
                              RagasEvaluator.evaluate()
                                          ↓
                              EvaluationResult
                                          ↓
                    ConsoleReporter + JsonReporter
```

## Key Dependencies

- `clients/medical_facts.py` - RAGFlow API client
- `clients/ragas_client.py` - RAGAS evaluation wrapper
- `config/settings.py` - Environment-based configuration
- `config/thresholds.py` - Quality pass/fail thresholds

## Adding New Features

1. **New metric type**: Add module in `metrics/`, import in `metrics/__init__.py`
2. **New ground truth field**: Update `models/ground_truth.py`
3. **New CLI option**: Update `cli.py` parser
4. **New output format**: Add reporter in `reporters/`
