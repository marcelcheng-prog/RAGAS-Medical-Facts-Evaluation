# Formatters Module

## Purpose

Convert structured medical facts JSON into SOAP note format (Subjective/Objective/Assessment/Plan).

## Key File: `soap_formatter.py`

### `build_soap_sections(data: dict) -> dict`

Maps 20+ clinical categories from extractor JSON to S/O/A/P arrays:

- **S (Subjective)**: chief_complaint, symptoms, patient_reported_history, anamnesis
- **O (Objective)**: vital_measurements/vital_signs, physical_examination, current_medications, lab_results
- **A (Assessment)**: diagnostic_hypotheses, diagnoses, clinical_impression
- **P (Plan)**: medications_planned, diagnostic_plans, therapeutic_interventions, follow_up, referrals

### `soap_text_from_sections(sections: dict) -> str`

Formats SOAP sections dictionary into readable `S:\n- item\n...` text.

### CLI Batch Mode

```bash
# Convert single file
python -m medical_facts_evaluation.formatters.soap_formatter input.json output.soap.json

# Batch convert directory
python -m medical_facts_evaluation.formatters.soap_formatter --input-dir test_cases/ --output-dir gold_soap/
```

### Ground Truth Auto-Detection

`load_source_payload()` automatically unwraps `{"ground_truth": {...}}` envelope from test case files.

### Key Aliases

Handles both `vital_measurements` and `vital_signs` as source keys.
