# RAGAS Medical Facts Evaluation Framework

Evaluation framework for testing Medical Facts extraction agents using the [RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) framework.

> **⚠️ Scope**: This framework currently evaluates **Medical Facts Extraction** agents only. SOAP note generation evaluation is **not yet supported**.

## Overview

This tool evaluates the quality of Medical Facts agents deployed on RAGFlow by:

- **Extracting medical facts** from doctor-patient transcripts (German)
- **Comparing extracted facts** against ground truth annotations
- **Computing RAGAS metrics** (Faithfulness, Context Recall, Answer Relevancy)
- **Measuring medication extraction quality** (Precision, Recall, F1, Hallucinations)
- **Comparing multiple agents** side-by-side

## Production Prompt Agent

**Agent ID**: `e1a25a64fdc611f0b3cb4afd40f7103b`

This agent uses the **same prompt as the production Medical Facts agent**. It has been tested across multiple test cases and achieves:
- **87-100% Medication Precision** (no hallucinations)
- **80-87% Medication Recall** 
- **95-100% Faithfulness**
- **~97% Quality Score** on most test cases

```bash
# Test the agent with the production prompt
python -m medical_facts_evaluation --agent-a e1a25a64fdc611f0b3cb4afd40f7103b --verbose
```

## Parallel Extraction Agent (Experimental)

**Agent ID**: `a6ef1157028011f180502a3cdbc575c7`

This experimental agent splits Medical Facts extraction into **4 parallel categories**:
1. **Medication** - Drug names, dosages, frequencies
2. **Clinical Assessment** - Diagnoses, symptoms, findings
3. **Measurement & Extraction** - Lab values, vital signs
4. **Patient Context** - Demographics, history, lifestyle

**Pros:**
- ⚡ **Faster performance** due to parallel processing

**Cons:**
- ❌ **Lower quality** than production agent
- Missing equivalence patterns (e.g., "Metamizol = Novalgin")
- ~80% Medication Recall vs 100% on production agent

```bash
# Test the parallel extraction agent
python -m medical_facts_evaluation --agent-a a6ef1157028011f180502a3cdbc575c7 --test-case medical_facts_evaluation/test_cases/hausarzt.json
```

**Recommendation**: Use production agent `e1a25a64fdc611f0b3cb4afd40f7103b` for better accuracy.

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

> **Note**: GPT-5.x and o-series models require RAGAS 0.5+ (current version has a bug with `max_tokens` vs `max_completion_tokens`). Use `gpt-4.1` for best results.

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

Compare Agent A (production) vs Agent B:

```bash
python -m medical_facts_evaluation --compare \
  --agent-a e1a25a64fdc611f0b3cb4afd40f7103b \
  --agent-b df4cb87efd2011f0b3234afd40f7103b \
  --test-case medical_facts_evaluation/test_cases/hausarzt.json \
  --verbose
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

| Test Case | Description | Medications | Complexity |
|-----------|-------------|-------------|------------|
| `michael_mueller.json` | Diabetes & back pain consultation | 8 meds (new, stopped, refused) | Medium |
| `diabetes.json` | Diabetes therapy change (DPP-4 → Ozempic) | 7 meds including Wegovy | Medium |
| `magenschmerzen_gastritis.json` | Gastritis with PPI therapy | 5 meds | Medium |
| `hausarzt.json` | **Hausarzt Praxis** - Complete GP consultation | 5 meds | **High** |
| `diabetes_hypertonie.json` | Diabetes Typ 2 + Hypertonie with family history | 7 meds (Jardins, Ramipril) | **High** |
| `medikamentenreview_polypharmazie.json` | Elderly patient (78y) with heart failure, polypharmacy | 7 meds, Swiss-German ASR errors | **High** |

### Hausarzt Praxis Test Case (Recommended)

The `hausarzt.json` test case is the **longest and most comprehensive transcript**, representing a typical Swiss German GP (Hausarzt) consultation with:
- Detailed patient history taking
- Complete physical examination
- Multiple vital sign measurements
- Lab value discussion (CRP, white blood cells)
- Medication changes (Esomep → Dexilant)
- Patient education

```bash
# Test with the comprehensive Hausarzt Praxis case
python -m medical_facts_evaluation \
  --test-case medical_facts_evaluation/test_cases/hausarzt.json \
  --agent-a e1a25a64fdc611f0b3cb4afd40f7103b \
  --verbose
```

### New Test Cases (February 2026)

#### Diabetes & Hypertonie with Family History

The `diabetes_hypertonie.json` test case features:
- Middle-aged patient with Diabetes Typ 2 and Hypertonie
- **Positive family history** (father: MI at 58, mother: diabetes/hypertension, brother: diabetes)
- Multiple medication adjustments (Amlodipin, Metformin, new Ramipril, Empagliflozin/Jardins)
- Tests **family history extraction** with RAGAS Context Recall

```bash
# Test family history extraction
python -m medical_facts_evaluation --compare \
  --agent-a b6bd2f21034811f19b402a3cdbc575c7 \
  --agent-b e1a25a64fdc611f0b3cb4afd40f7103b \
  --test-case medical_facts_evaluation/test_cases/diabetes_hypertonie.json \
  --verbose
```

#### Polypharmacy & Swiss-German ASR Errors

The `medikamentenreview_polypharmazie.json` test case features:
- **Elderly patient (78 years)** with heart failure, atrial fibrillation, and osteoarthritis
- **Polypharmacy** review with 7 medications
- **Swiss-German ASR transcription errors**: `Koncor` (Concor), `Tafalgan` (Dafalgan), `Witamin D` (Vitamin D)
- Tests agent handling of phonetic transcription artifacts

```bash
# Test polypharmacy and ASR error handling
python -m medical_facts_evaluation --compare \
  --agent-a b6bd2f21034811f19b402a3cdbc575c7 \
  --agent-b e1a25a64fdc611f0b3cb4afd40f7103b \
  --test-case medical_facts_evaluation/test_cases/medikamentenreview_polypharmazie.json \
  --verbose
```

### Test Case Structure

```json
{
  "test_id": "unique_id",
  "name": "Patient Name - Condition",
  "description": "Brief description",
  "language": "de",
  "transcript": "Full doctor-patient conversation...",
  "ground_truth": {
    "diagnoses": ["Diabetes mellitus Typ 2", "..."],
    "medications": [
      {
        "name": "Metformin",
        "dosage": "500 mg",
        "frequency": "1-0-1",
        "notes": "Increased dosage"
      }
    ],
    "vital_signs": [
      {"type": "HbA1c", "value": "7.2", "unit": "%"}
    ],
    "symptoms": ["Fatigue", "Increased thirst"],
    "procedures": ["Blood glucose measurement"],
    "follow_up": "Control appointment in 4 weeks"
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

RAGAS (Retrieval Augmented Generation Assessment) provides LLM-based evaluation metrics. Here's how each metric works:

#### Faithfulness (≥ 90%)

**What it measures**: Whether the agent's extracted facts are actually supported by the source transcript.

**How it works**:
1. The LLM breaks down the agent's output into individual **claims** (e.g., "Patient takes Metformin 500mg")
2. For each claim, the LLM checks if it can be **inferred from the transcript**
3. Claims are classified as either **supported** or **unsupported** (hallucinated)
4. Score = `(Supported Claims) / (Total Claims)`

**Example**:
- Agent outputs: "Patient has diabetes (✓), takes Metformin (✓), allergic to penicillin (✗ - not mentioned)"
- 2 out of 3 claims supported → **Faithfulness = 66.7%**

**Why it matters**: A low faithfulness score indicates **hallucinations** - the agent is making up medical facts not present in the conversation.

---

#### Context Recall (≥ 85%)

**What it measures**: Whether the agent captured all the relevant information from the transcript.

**How it works**:
1. The LLM examines the **ground truth** (expected output)
2. For each piece of ground truth, it checks if it could be **attributed to the transcript**
3. Then checks if this information appears in the **agent's output**
4. Score = `(Ground Truth Items Retrieved) / (Total Ground Truth Items)`

**Example**:
- Ground truth has 5 medications
- Agent correctly extracted 4 of them
- **Context Recall = 80%**

**Why it matters**: A low context recall means the agent is **missing important medical information** that was discussed.

---

#### Answer Relevancy (≥ 80%)

**What it measures**: Whether the agent's output is actually relevant to the medical context and query.

**How it works**:
1. The LLM generates **synthetic questions** that the agent's answer would address
2. These questions are compared to the **original context**
3. Measures how well the answer addresses what was actually asked
4. Penalizes irrelevant or off-topic content

**Why it matters**: Ensures the agent stays focused on relevant medical facts rather than including tangential information.

---

### Medication Metrics

These are deterministic (non-LLM) metrics calculated by comparing extracted medications against ground truth:

| Metric | Formula | Threshold |
|--------|---------|-----------|
| **Precision** | `TP / (TP + FP)` - What % of extracted meds are correct | ≥ 95% |
| **Recall** | `TP / (TP + FN)` - What % of actual meds were found | ≥ 90% |
| **F1 Score** | `2 × (P × R) / (P + R)` - Balanced accuracy | - |
| **Hallucination Score** | `1 - (Hallucinated Meds / Total Extracted)` | ≥ 98% |

**Where**:
- **TP** (True Positives): Medications correctly extracted
- **FP** (False Positives): Medications extracted but not in ground truth (hallucinations)
- **FN** (False Negatives): Medications in ground truth but not extracted (missed)

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
- **GPT-5.x Support**: Requires RAGAS 0.5+ (current 0.4.3 has a bug with `max_completion_tokens`)


Marcel Cheng - [marcelcheng-prog](https://github.com/marcelcheng-prog)
