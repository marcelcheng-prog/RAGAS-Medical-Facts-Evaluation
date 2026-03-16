---
name: soap-eval
description: 'Run SOAP note evaluation on RAGFlow agents with verbose output and three-tier scoring. Use when: evaluating SOAP agents, comparing SOAP note quality, viewing section scores (S/O/A/P), running verbose SOAP mode, checking semantic vs lexical scores, SOAP agent diagnostics.'
argument-hint: 'Agent ID and optional test case (e.g., "f44320e6 michael_mueller")'
---

# SOAP Note Evaluation Skill

## When to Use

- User wants to evaluate a SOAP note generation agent
- User wants verbose SOAP output with section analysis
- User asks about SOAP section scores (S/O/A/P)
- User wants to compare predicted vs gold SOAP items
- User mentions SOAP agents, soap mode, or soap evaluation

## Known SOAP Agent IDs

| Agent | ID | Notes |
|-------|----|-------|
| SOAP v1 | `be95d9821ef211f194964348756e437e` | First version |
| SOAP v2 | `f44320e61ef011f194964348756e437e` | Improved version |

## Procedure

### 1. Single Agent, Single Test Case (verbose)

```bash
cd P:\MCH\Ragas\RAGAS_Medical_Facts_Agent\RAGAS_Medical_Facts_Agent_refactor
python -m medical_facts_evaluation --mode soap --agent-a <AGENT_ID> --test-case medical_facts_evaluation/test_cases/<TEST_CASE>.json --verbose-soap
```

Or from parent directory:
```bash
.\run_soap_eval.ps1 --mode soap --agent-a <AGENT_ID> --test-case medical_facts_evaluation/test_cases/<TEST_CASE>.json --verbose-soap
```

### 2. All Test Cases

```bash
python -m medical_facts_evaluation --mode soap --agent-a <AGENT_ID> --all-test-cases --verbose-soap
```

### 3. Compare Two SOAP Agents

```bash
python -m medical_facts_evaluation --mode soap --compare --agent-a <ID1> --agent-b <ID2> --all-test-cases --verbose-soap
```

### 4. Custom Thresholds

```bash
python -m medical_facts_evaluation --mode soap --agent-a <ID> \
  --soap-structure-threshold 0.75 \
  --soap-content-threshold 0.60 \
  --verbose-soap
```

## Three-Tier Scoring

| Layer | Weight | Method |
|-------|--------|--------|
| Lexical | 35% | Token F1 word overlap |
| GPT Semantic | 30% | LLM judges medical equivalence |
| RAGAS Semantic | 35% | Context recall semantic matching |

### Score Calculation
- **Effective section score** = weighted blend of all three layers
- **Overall** = 30% structure + 70% average effective content
- **Floor**: effective ≥ lexical × 0.8

## Output Interpretation

### Verbose Output Shows:
1. **Summary**: Overall score, structure %, content %, lexical %, semantic %
2. **Section Scores**: S, O, A, P breakdown (lexical and effective)
3. **Weakness Summary**: Sections below 60%
4. **Section Analysis**: For each S/O/A/P:
   - Predicted items (what agent produced)
   - Gold items (what was expected)
   - Missing items (in gold but not predicted)
   - Extra items (in predicted but not gold)
   - Overlap count
5. **Top Actionable Fixes**: Priority improvements for agent prompt

### Pass/Fail Thresholds (default)
- Structure: ≥100% (all 4 sections present)
- Content: ≥70% effective average

### Section Aliases
SOAP agents may return keys as `subjective/objective/assessment/plan` instead of `S/O/A/P`. The evaluator handles both automatically.

## Gold SOAP Files

Located in `medical_facts_evaluation/gold_soap/`:
- Auto-generated from test case ground truth
- Format: `<test_name>.soap.json` and `<test_name>.soap.txt`
- Should be reviewed by clinician for production accuracy

## Results Storage

Results are saved to timestamped folders:
```
results/medical_facts_production/soap_agents/<timestamp>_<agent_id_prefix>/
  └── <agent_prefix>/<test_case>.soap_eval.json
```

## Troubleshooting

- **All sections empty**: Agent may return different key names - check raw output in saved JSON
- **RAGAS always 100%**: May indicate RAGAS is matching too broadly - check lexical scores for ground truth
- **GPT scoring errors**: Check `OPENAI_API_KEY` and model availability
- **ModuleNotFoundError**: Run from `RAGAS_Medical_Facts_Agent_refactor/` directory
