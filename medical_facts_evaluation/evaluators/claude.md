# Evaluators Module

## Purpose

SOAP note evaluation with three-tier scoring (lexical + GPT-semantic + RAGAS).

## Key File: `soap_evaluator.py`

### Three Scoring Layers

1. **Lexical** (`_token_f1`): Token-level F1 between predicted and gold items
2. **GPT Semantic** (`_llm_semantic_section_score`): OpenAI GPT judges medical equivalence (e.g., "morgens" == "1-0-0")
3. **RAGAS Semantic** (`evaluate_semantic_list_recall`): RAGAS context_recall for semantic list matching

### Weighted Blend

`_combine_section_score()` blends all three signals:
- Lexical: 35%, GPT: 30%, RAGAS: 35%
- Floor: blended ≥ lexical × 0.8 (prevents semantic-only perfect scores)

### Section Aliases

SOAP agents return various key names. `SECTION_ALIASES` maps them:
- `subjective/subjektiv` → `S`
- `objective/objektiv` → `O`
- `assessment/beurteilung` → `A`
- `plan/planung` → `P`

### Key Functions

- `extract_soap_sections(output)` - Parse SOAP from JSON, medical facts JSON, or plain text
- `evaluate_soap_output(...)` - Main entry point, returns `SoapEvaluationResult`
- `load_gold_sections(test_case_path, gold_soap_dir)` - Load gold standard

### SoapEvaluationResult Fields

- `section_scores` - Lexical F1 per section
- `semantic_section_scores` - GPT judge per section
- `ragas_section_scores` - RAGAS recall per section
- `effective_section_scores` - Weighted blend per section
- `overall_score` - 30% structure + 70% content (0-100 scale)
