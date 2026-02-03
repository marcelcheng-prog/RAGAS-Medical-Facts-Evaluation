# RAGAS Medical Facts Evaluation Framework

A comprehensive evaluation framework for testing Medical Facts extraction agents using the [RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) framework.

## Overview

This tool evaluates the quality of Medical Facts agents deployed on RAGFlow by:

- **Extracting medical facts** from doctor-patient transcripts (German)
- **Comparing extracted facts** against ground truth annotations
- **Computing RAGAS metrics** (Faithfulness, Context Recall, Answer Relevancy)
- **Measuring medication extraction quality** (Precision, Recall, F1, Hallucinations)
- **Comparing multiple agents** side-by-side

## Features

- 🔬 **RAGAS Evaluation**: Industry-standard LLM evaluation metrics
- 💊 **Medication Tracking**: Precision/Recall for medication extraction
- 🆚 **Agent Comparison**: Compare two agents on the same test case
- 📊 **Rich Output**: Beautiful terminal tables with pass/fail indicators
- 📁 **JSON Reports**: Detailed results saved for analysis
- 🧪 **Multiple Test Cases**: German medical transcripts with ground truth

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key (for RAGAS evaluation)
- Access to RAGFlow Medical Facts Agent

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/marcelcheng-prog/RAGAS-Medical-Facts-Evaluation.git
   cd RAGAS-Medical-Facts-Evaluation
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here
MEDICAL_FACTS_AUTH_TOKEN=ragflow-your-token-here

# Optional
RAGFLOW_BASE_URL=http://172.17.16.150/api/v1/agents_openai
OPENAI_MODEL=gpt-4.1  # or gpt-4o, gpt-4o-mini
```

### Supported OpenAI Models

| Model | Quality | Speed | Cost | RAGAS Compatible |
|-------|---------|-------|------|------------------|
| `gpt-4.1` | ⭐⭐⭐⭐⭐ | Medium | $$ | ✅ |
| `gpt-4o` | ⭐⭐⭐⭐ | Fast | $$ | ✅ |
| `gpt-4o-mini` | ⭐⭐⭐ | Fast | $ | ✅ |
| `gpt-4-turbo` | ⭐⭐⭐⭐ | Medium | $$$ | ✅ |
| `gpt-5.x` | ⭐⭐⭐⭐⭐ | Fast | $$$ | ⚠️ Partial |

> **Note**: GPT-5.x models require `max_completion_tokens` instead of `max_tokens`. The framework has partial support, but some RAGAS features may not work.

## Usage

### Basic Evaluation

Evaluate the default agent with the default test case:

```bash
python -m medical_facts_evaluation
```

### Verbose Output

Show detailed extraction results:

```bash
python -m medical_facts_evaluation --verbose
```

### Specific Test Case

Run evaluation on a specific test case:

```bash
python -m medical_facts_evaluation --test-case test_cases/michael_mueller.json
```

### Compare Two Agents

Compare Agent A vs Agent B:

```bash
python -m medical_facts_evaluation --compare \
  --agent-a e1a25a64fdc611f0b3cb4afd40f7103b \
  --agent-b df4cb87efd2011f0b3234afd40f7103b \
  --test-case test_cases/diabetes_kneubühler.json
```

### All Options

```bash
python -m medical_facts_evaluation --help
```

```
options:
  -h, --help            Show this help message and exit
  --verbose, -v         Show detailed output
  --test-case FILE      Path to test case JSON file
  --agent-a ID          Agent A ID for comparison
  --agent-b ID          Agent B ID for comparison
  --compare             Compare two agents
  --iterations N        Run multiple iterations (default: 1)
```

## Test Cases

Test cases are JSON files in `medical_facts_evaluation/test_cases/`:

| Test Case | Description | Medications |
|-----------|-------------|-------------|
| `michael_mueller.json` | Diabetes & back pain consultation | 7 meds (new, stopped, refused) |
| `diabetes_kneubühler.json` | Diabetes therapy change (DPP-4 → Ozempic) | 6 meds including Wegovy |
| `frau_mueller_bauchschmerzen.json` | Abdominal pain, suspected diverticulitis | Multiple antibiotics |
| `magenschmerzen_gastritis.json` | Gastritis with PPI therapy | Esomep, Novalgin, etc. |

### Test Case Structure

```json
{
  "test_id": "unique_id",
  "name": "Patient Name - Condition",
  "description": "Brief description",
  "language": "de",
  "transcript": "Full doctor-patient conversation...",
  "ground_truth": {
    "diagnoses": ["..."],
    "medications": [
      {
        "name": "Medication Name",
        "dosage": "10 mg",
        "frequency": "1-0-1",
        "notes": "Additional context"
      }
    ],
    "vital_signs": [...],
    "lab_values": [...],
    "symptoms": [...],
    "procedures": [...],
    "follow_up": "..."
  }
}
```

## Output

### Console Output

The framework displays rich tables with:

- **Medication Extraction Metrics**: Precision, Recall, F1, Hallucinations
- **RAGAS Scores**: Faithfulness, Context Recall, Answer Relevancy
- **Quality Score**: Weighted composite (0-100)
- **Pass/Fail Status**: Based on configurable thresholds

### JSON Reports

Results are saved to `results/medical_facts_production/`:

```
results/
└── medical_facts_production/
    ├── agent_a_e1a25a64_20260203_143022.json
    ├── agent_b_df4cb87e_20260203_143022.json
    └── comparison_20260203_143022.json
```

## Metrics Explained

### RAGAS Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Faithfulness** | Are extracted facts grounded in the transcript? | ≥ 90% |
| **Context Recall** | Are all relevant facts from transcript captured? | ≥ 85% |
| **Answer Relevancy** | Is the output relevant to the medical context? | ≥ 80% |

### Medication Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Precision** | % of extracted medications that are correct | ≥ 95% |
| **Recall** | % of ground truth medications that were found | ≥ 90% |
| **F1 Score** | Harmonic mean of Precision and Recall | - |
| **Hallucination Score** | % of outputs that are NOT hallucinated | ≥ 98% |

## Project Structure

```
RAGAS_Medical_Facts_Agent_refactor/
├── medical_facts_evaluation/
│   ├── __init__.py
│   ├── __main__.py           # Entry point
│   ├── cli.py                # Command-line interface
│   ├── evaluator.py          # Main orchestrator
│   ├── clients/
│   │   ├── medical_facts.py  # RAGFlow API client
│   │   └── ragas_client.py   # RAGAS evaluation
│   ├── config/
│   │   ├── settings.py       # Environment configuration
│   │   └── thresholds.py     # Quality thresholds
│   ├── metrics/
│   │   ├── medication.py     # Medication evaluation
│   │   ├── vital_signs.py    # Vital signs evaluation
│   │   └── symptoms.py       # Symptom evaluation
│   ├── models/
│   │   ├── ground_truth.py   # Data classes
│   │   ├── evaluation.py     # Result models
│   │   └── loader.py         # Test case loader
│   ├── reporters/
│   │   ├── console.py        # Rich terminal output
│   │   └── json_reporter.py  # JSON file output
│   └── test_cases/
│       ├── schema.json       # Test case JSON schema
│       ├── michael_mueller.json
│       ├── diabetes_kneubühler.json
│       └── ...
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Development

### Adding New Test Cases

1. Create a new JSON file in `medical_facts_evaluation/test_cases/`
2. Follow the schema in `schema.json`
3. Include German medical transcript and complete ground truth

### Adjusting Thresholds

Edit `medical_facts_evaluation/config/thresholds.py`:

```python
PRODUCTION = QualityThresholds(
    medication_precision=0.95,
    medication_recall=0.90,
    faithfulness=0.90,
    # ...
)
```

## Known Limitations

- **Exact String Matching**: Medication names require exact matches. "Esomep" ≠ "Esomeprazol"
- **German Only**: Test cases and prompts are in German
- **GPT-5.x Support**: Partial - RAGAS library needs update for `max_completion_tokens`

## Future Improvements

- [ ] Fuzzy matching for medication names (RapidFuzz)
- [ ] Synonym database for brand/generic names
- [ ] Multi-language support
- [ ] Web UI for results visualization
- [ ] Batch evaluation across all test cases

## License

MIT

## Author

Marcel Cheng - [marcelcheng-prog](https://github.com/marcelcheng-prog)
