# Metrics Module

## Purpose

Contains evaluation algorithms for measuring Medical Facts extraction quality. Each metric module focuses on a specific aspect of the extracted data.

## Modules

### `medication.py` - Medication Extraction Metrics

**Primary metric module** - Most important for evaluation.

Key functions:
- `evaluate_medications()` - Main entry point, returns `MedicationEvaluation`
- `normalize_med_name()` - Lowercase, strip for comparison

**Evaluation logic:**
1. Extract medication names from agent response
2. Compare against ground truth `all_medication_names`
3. Calculate precision, recall, F1 score
4. Detect hallucinations (meds not in transcript)
5. Check for forbidden medications (critical safety)
6. Score dose and frequency accuracy

**Hallucination Detection:**
- Checks if extracted medication names appear in original transcript
- Flags medications on `forbidden_medications` list as critical

### `vital_signs.py` - Vital Signs Accuracy

Evaluates extraction of:
- Blood pressure (Blutdruck)
- Heart rate (Herzfrequenz)
- Temperature, weight, etc.

Compares parameter/value pairs against ground truth.

### `symptoms.py` - Symptom Completeness

Measures how many patient-reported symptoms were captured.
Uses fuzzy matching for German symptom descriptions.

### `safety.py` - Safety Checks

- `detect_hallucinations()` - Cross-reference with transcript
- `check_forbidden_medications()` - Critical safety violations

## Scoring Approach

All metrics return scores between 0.0 and 1.0:

```python
precision = correct_matches / found_total if found_total > 0 else 0.0
recall = correct_matches / expected_total if expected_total > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
```

## Adding a New Metric

1. Create new file: `metrics/new_metric.py`
2. Implement evaluation function returning a dataclass or dict
3. Export in `metrics/__init__.py`
4. Call from `evaluator.py` in the `evaluate()` method
5. Add to `EvaluationResult` in `models/evaluation.py`
6. Update console reporter to display results

## Important Considerations

- **German language**: Medication names and symptoms are in German
- **Equivalence patterns**: Some meds have multiple names (e.g., Metamizol = Novalgin)
- **Case-insensitive**: All comparisons use `normalize_med_name()`
- **Hallucinations are critical**: Any fabricated medication is a serious error
