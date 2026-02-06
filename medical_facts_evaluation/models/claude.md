# Models Module

## Purpose

Defines all data structures used throughout the evaluation framework. Uses Python dataclasses for clean, typed data handling.

## Key Classes

### `GroundTruth` (ground_truth.py)

Hand-labeled expected data for a test case.

```python
@dataclass
class GroundTruth:
    medications_taken: list[Medication]      # Currently taking
    medications_planned: list[Medication]    # New/changed/stopped
    all_medication_names: list[str]          # All mentioned meds
    vital_measurements: list[VitalMeasurement]
    symptoms: list[str]
    medical_history: list[str]
    diagnostic_plans: list[str]
    therapeutic_interventions: list[str]
    forbidden_medications: list[str]         # Should NEVER appear
```

### `Medication` (ground_truth.py)

Individual medication with full context:

```python
@dataclass
class Medication:
    name: str
    dose: Optional[str]       # e.g., "500mg"
    frequency: Optional[str]  # e.g., "2x täglich"
    action: Optional[str]     # "new", "stopped", "changed", "refused"
    reason: Optional[str]
    indication: Optional[str]
    notes: Optional[str]
```

### `EvaluationResult` (evaluation.py)

Complete result from evaluating one test case:

```python
@dataclass
class EvaluationResult:
    # Metadata
    test_name: str
    timestamp: str
    agent_id: str
    
    # API metrics
    api_time_seconds: float
    api_error: Optional[str]
    
    # Raw data
    agent_output_raw: str
    agent_facts: Optional[dict]
    
    # RAGAS scores (0.0-1.0)
    faithfulness: Optional[float]
    context_recall: Optional[float]
    answer_relevancy: Optional[float]
    
    # Custom metrics
    medication_eval: MedicationEvaluation
    vital_signs_accuracy: float
    symptoms_completeness: float
    
    # Safety
    critical_hallucinations: list[str]
    
    # Verdict
    passed: bool
    quality_score: float
    failure_reasons: list[str]
```

### `MedicationEvaluation` (evaluation.py)

Detailed medication metrics:

```python
@dataclass
class MedicationEvaluation:
    expected_total: int
    found_total: int
    correct_matches: int
    missing_medications: list[str]
    extra_medications: list[str]
    hallucinations: list[str]
    name_precision: float
    name_recall: float
    f1_score: float
    dose_accuracy: float
    frequency_accuracy: float
```

### `TestCase` (loader.py)

Container for a complete test case:

```python
@dataclass
class TestCase:
    test_id: str
    name: str
    transcript: str          # German doctor-patient conversation
    ground_truth: GroundTruth
    metadata: dict
```

## Loader Functions

- `load_test_case(path)` - Load single JSON test case
- `load_all_test_cases()` - Load all from test_cases directory
- `get_default_test_case()` - Built-in Michael Mueller test

## JSON Serialization

All dataclasses have `to_dict()` methods for JSON export:

```python
result.to_dict()  # Returns serializable dict
```

## Creating Test Cases

Test case JSON structure:

```json
{
  "test_id": "unique_id",
  "name": "Human-readable name",
  "transcript": "Arzt: Guten Tag...",
  "ground_truth": {
    "all_medication_names": ["Ibuprofen", "Metformin"],
    "medications_taken": [...],
    "medications_planned": [...],
    "vital_measurements": [...],
    "symptoms": [...],
    "forbidden_medications": [...]
  },
  "metadata": {
    "language": "de",
    "specialty": "general_practice"
  }
}
```

See `test_cases/schema.json` for full schema definition.
