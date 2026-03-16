# RAGAS Medical Facts & SOAP Evaluation Framework

Evaluation framework for testing **Medical Facts extraction** and **SOAP note generation** agents using the [RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) framework, with a **Streamlit web UI** for interactive evaluation, agent ranking, and favorites management.

## Overview

This tool evaluates the quality of Medical Facts and SOAP agents deployed on RAGFlow by:

- **Extracting medical facts** from doctor-patient transcripts (German)
- **Comparing extracted facts** against ground truth annotations
- **Evaluating SOAP notes** with lexical, GPT-semantic, and RAGAS scoring
- **Computing RAGAS metrics** (Faithfulness, Context Recall, Answer Relevancy)
- **Measuring medication extraction quality** (Precision, Recall, F1, Hallucinations)
- **Comparing multiple agents** side-by-side with ranking leaderboard
- **Interactive web UI** for running evaluations, managing agent favorites, and viewing rankings

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
- 🧼 **SOAP Evaluation**: Three-tier scoring (lexical + GPT-semantic + RAGAS)
- 🆚 **Agent Comparison**: Compare two agents on the same test case
- 📊 **Rich Output**: Beautiful terminal tables with pass/fail indicators
- 📁 **JSON Reports**: Detailed results saved for analysis
- 🧪 **Multiple Test Cases**: 13+ German medical transcripts with ground truth
- 🌐 **Web UI**: Streamlit dashboard for interactive evaluation and ranking
- ⭐ **Favorites**: Save and manage preferred agent configurations
- 🏆 **Leaderboard**: Automatic ranking of agents by score and response time

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

## Quality Metrics (QM) Codes

The evaluation framework tests agents against specific quality metrics. These codes define what aspects of medical extraction are being validated.

### Severity Levels

| Level | Icon | Description |
|-------|:----:|-------------|
| **PASS** | 🟢 | Akzeptiert - meets quality standards |
| **MAJOR** | 🟡 | Nacharbeit nötig - requires manual review/correction |
| **CRITICAL** | 🔴 | Showstopper - patient safety risk, must be fixed |

---

### SAFETY Metrics (QM-001 to QM-008) - Medikation & Patientendaten

| ID | Kategorie | Metrik / Prüfpunkt | 🟢 PASS | 🟡 MAJOR | 🔴 CRITICAL |
|----|-----------|-------------------|---------|----------|-------------|
| **QM-001** | Medikation | **Wirkstoff-Integrität** | Name korrekt erkannt (Verbatim) | Leichter Tippfehler im Namen | Falscher Wirkstoff / Halluzination |
| **QM-002** | Medikation | **Dosierung & Einheit** | Zahl und Einheit (mg/µg) exakt | Einheit fehlt, aber Zahl stimmt | Falsche Dosis oder Einheit (z.B. mg statt µg) |
| **QM-003** | Medikation | **Logik: Bestand vs. Neu** | Korrekt getrennt in Anamnese/Plan | Vermischt, aber als Medikation erkennbar | Bestandsmedikament fälschlich als "neu" |
| **QM-004** | Medikation | **Negativ-Filter** | Abgelehntes Medikament ignoriert | Erwähnt, aber Status unklar | Abgelehntes Medikament als "Verordnung" |
| **QM-005** | Medikation | **Unsicherheits-Handling** | Markiert als [UNCLEAR] | Ungenaue Angabe ohne Markierung | KI rät/erfindet eine Dosis |
| **QM-006** | Patientendaten | **Lateralität (Links/Rechts)** | Seite korrekt (z.B. linkes Knie) | Seite fehlt (nur "Knie") | Seite vertauscht (Rechts statt Links) |
| **QM-007** | Patientendaten | **Allergien (CAVE)** | Allergie korrekt erfasst | Allergie im Fließtext versteckt | Allergie übersehen oder "Keine Allergien" erfunden |
| **QM-008** | Patientendaten | **Ausschluss (Negation)** | Kein Fieber korrekt erkannt | Unpräzise ("Unwohlsein" statt "Grippe") | Hat Fieber (Das "Kein" überlesen) |

---

### ACCURACY Metrics (QM-009 to QM-013) - Diagnostik & Kontext

| ID | Kategorie | Metrik / Prüfpunkt | 🟢 PASS | 🟡 MAJOR | 🔴 CRITICAL |
|----|-----------|-------------------|---------|----------|-------------|
| **QM-009** | Diagnostik | **Hauptdiagnose** | Kernproblem korrekt identifiziert | Unpräzise ("Unwohlsein" statt "Grippe") | Thema verfehlt / Falsche Diagnose erfunden |
| **QM-010** | Diagnostik | **Messwerte (Vitals/Labs)** | Werte exakt (RR 120/80) | Wert da, Parametername unklar | Zahlenfehler (Kommafehler) |
| **QM-011** | Kontext | **Attribution (Wer?)** | Symptome Dritter (Ehefrau) ignoriert | Dritte erwähnt, aber abgegrenzt | Symptome Dritter dem Patienten zugeordnet |
| **QM-012** | Kontext | **Zeitlicher Verlauf** | Zeitangaben ("seit 3 Tagen") korrekt | Zeit fehlt ("seit einiger Zeit") | Falsche Zeit ("seit 3 Wochen" statt "3 Tagen") |
| **QM-013** | Kontext | **Korrektur-Erkennung** | Letztgültige Aussage ("nein doch nicht") zählt | Beide Aussagen (falsch & richtig) gelistet | Nur die falsche (korrigierte) Aussage übernommen |

---

### USABILITY Metrics (QM-014 to QM-018) - Struktur, Inhalt & Stil

| ID | Kategorie | Metrik / Prüfpunkt | 🟢 PASS | 🟡 MAJOR | 🔴 CRITICAL |
|----|-----------|-------------------|---------|----------|-------------|
| **QM-014** | Struktur | **SOAP-Formatierung** | Saubere Trennung S-O-A-P | Infos in falscher Sektion (z.B. Befund in Plan) | Keine Struktur / Fließtext-Block |
| **QM-015** | Inhalt | **Noise Filter (Smalltalk)** | Kein Smalltalk (Wetter/Urlaub) | Kurzer Satz Smalltalk enthalten | Lange Passagen über Irrelevantes (Urlaub, Admin) |
| **QM-016** | Stil | **Medizinischer Jargon** | Fachsprache / Stichpunkte | Umgangssprache / Ganze Sätze | Chatbot-Stil ("Der Arzt sagte dann...") |
| **QM-017** | Stil | **Sprache & Grammatik** | Korrekte deutsche Grammatik | Leichte Grammatikfehler | Englische Wörter gemischt / Sinn entstellt |
| **QM-018** | Inhalt | **Halluzination (Füller)** | Nur Fakten aus Audio | - | Erfundene Untersuchungen ("Abdomen weich"), die nie stattfanden |

---

> **Note**: QM-014 (SOAP Formatting) applies to SOAP note generation, not Medical Facts extraction.

## Test Cases

Test cases are JSON files in `medical_facts_evaluation/test_cases/`:

### Test Case Overview

| Test Case | Description | Medications | QM Codes Tested | Complexity |
|-----------|-------------|:-----------:|-----------------|:----------:|
| `ortho_knee_arthrose.json` | Knee arthritis, NSAID allergy, opioid refusal | 8 | **QM-002 to QM-008** | **High** |
| `gyn_pregnancy_gestational_diabetes.json` | Pregnancy SSW 28, gestational diabetes, 3 allergies | 11 | **QM-002 to QM-008** | **High** |
| `diabetes_hypertonie.json` | Diabetes + Hypertension with family history | 7 | QM-001, QM-003, **QM-010** | **High** |
| `medikamentenreview_polypharmazie.json` | Elderly polypharmacy, heart failure | 7 | QM-001, **QM-009** (ASR) | **High** |
| `hausarzt.json` | Complete GP consultation | 5 | QM-001, QM-003, QM-011 | **High** |
| `michael_mueller.json` | Diabetes & back pain | 8 | QM-001, QM-003, QM-004 | Medium |
| `diabetes.json` | Diabetes therapy change | 7 | QM-001, QM-003 | Medium |
| `magenschmerzen_gastritis.json` | Gastritis with PPI therapy | 5 | QM-001, QM-011, QM-012 | Medium |
| `bauchschmerzen.json` | Abdominal pain & fever | 5 | QM-001, QM-011 | Medium |
| `korrektur_noise_filter.json` | Corrections & smalltalk | 4 | **QM-013**, **QM-015** | High |

### Recommended Test Cases for Comprehensive QM Testing

#### For QM-002 through QM-008 (Critical Accuracy Metrics):

```bash
# Test all critical QM codes (002-008) with orthopedic case
python -m medical_facts_evaluation \
  --agent-a e1a25a64fdc611f0b3cb4afd40f7103b \
  --test-case medical_facts_evaluation/test_cases/ortho_knee_arthrose.json \
  --verbose

# Test all critical QM codes (002-008) with gynecology/pregnancy case
python -m medical_facts_evaluation \
  --agent-a e1a25a64fdc611f0b3cb4afd40f7103b \
  --test-case medical_facts_evaluation/test_cases/gyn_pregnancy_gestational_diabetes.json \
  --verbose
```

#### For QM-013 (Correction Recognition) and QM-015 (Noise Filter):

```bash
# Test correction handling and smalltalk filtering
python -m medical_facts_evaluation \
  --agent-a e1a25a64fdc611f0b3cb4afd40f7103b \
  --test-case medical_facts_evaluation/test_cases/korrektur_noise_filter.json \
  --verbose
```

### QM Code Details by Test Case

#### `korrektur_noise_filter.json` - QM-013 & QM-015

| QM Code | What's Tested | Expected Behavior |
|---------|---------------|-------------------|
| QM-013 | Patient corrections: "Metoprolol...nein, Bisoprolol" | Only final corrected value should appear |
| QM-013 | Dose corrections: "2.5 mg...nein doch nicht, 5 mg" | Only corrected dose |
| QM-013 | Allergen correction: "Penicillin...nein, Amoxicillin" | Only corrected allergen |
| QM-015 | Smalltalk: vacation in Spain, weather, TV | Should NOT appear in output |
| QM-015 | Non-medical: husband fishing, grandchildren | Should NOT appear in output |

**Test Corrections to Detect:**
- Metoprolol → Bisoprolol ✅ (forbidden: Metoprolol)
- Ramipril 2.5 mg → 5 mg
- Penicillin → Amoxicillin ✅ (forbidden: Penicillin)
- Simvastatin 20 mg → 40 mg (action=changed)
- L-Thyroxin 50 → 75 µg (action=changed)
- trockener Husten → produktiver Husten

#### `ortho_knee_arthrose.json` - Orthopädie (QM-002 to QM-008)

| QM Code | What's Tested | Expected Behavior |
|---------|---------------|-------------------|
| QM-002 | L-Thyroxin 75 **µg** (not mg!) | Must recognize Mikrogramm vs Milligramm |
| QM-003 | L-Thyroxin=continued, Dafalgan=changed, Capsaicin=new | Correct action classification |
| QM-004 | Tramadol refused, Voltaren-Gel contraindicated | Must NOT appear in `medications_planned` |
| QM-005 | L-Thyroxin dose unclear (75 or 50 µg) | Should mark as `[UNCLEAR]` |
| QM-006 | Left knee=pain, Right knee=normal | No laterality confusion |
| QM-007 | Ibuprofen allergy (rash) | Must capture with reaction |
| QM-008 | No swelling, no redness, no fever | Must NOT invert negations |

#### `gyn_pregnancy_gestational_diabetes.json` - Gynäkologie (QM-002 to QM-008)

| QM Code | What's Tested | Expected Behavior |
|---------|---------------|-------------------|
| QM-002 | Ferro-Gradumet 105 mg, Dafalgan 500 mg | Correct mg dosages |
| QM-003 | Elevit/Magnesium=continued, Ferro-Gradumet/Dafalgan=new, Pille=stopped | Full action classification |
| QM-004 | Ibuprofen **kontraindiziert** in pregnancy | Critical: Must NOT prescribe |
| QM-005 | Magnesium dose unclear (300 or 400 mg), Elevit timing unclear | Mark uncertainties |
| QM-006 | Right-sided pain, left side pain-free | Correct laterality |
| QM-007 | 3 allergies: Penicillin, Cotrimoxazol, Novalgin | All three with reactions |
| QM-008 | No bleeding, no contractions, no fever | Negations preserved |

### Additional Test Case Details

#### Hausarzt Praxis Test Case

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
│   ├── evaluators/
│   │   └── soap_evaluator.py # SOAP three-tier scoring
│   ├── formatters/
│   │   └── soap_formatter.py # Medical facts → SOAP conversion
│   ├── metrics/
│   │   ├── medication.py     # Medication evaluation
│   │   ├── vital_signs.py    # Vital signs evaluation
│   │   ├── symptoms.py       # Symptom evaluation
│   │   └── safety.py         # Hallucination & safety checks
│   ├── models/
│   │   ├── ground_truth.py   # Data classes
│   │   ├── evaluation.py     # Result models
│   │   └── loader.py         # Test case loader
│   ├── reporters/
│   │   ├── console.py        # Rich terminal output
│   │   └── json_reporter.py  # JSON file output
│   ├── test_cases/           # 13+ ground truth JSON files
│   ├── gold_soap/            # Gold SOAP references
│   └── prompts/
│       └── soap_formatter_prompt.md
├── app.py                    # Streamlit web UI
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

## SOAP Note Evaluation

The framework supports evaluation of SOAP note generation agents with a three-tier scoring system:

### Scoring Layers

| Layer | Weight | Method | Description |
|-------|--------|--------|-------------|
| **Lexical** | 35% | Token F1 | Word-level overlap between predicted and gold sections |
| **GPT Semantic** | 30% | `gpt-4o-mini` | LLM judges medical equivalence (e.g., "morgens" = "1-0-0") |
| **RAGAS Semantic** | 35% | Context Recall | RAGAS framework measures semantic list recall |

### SOAP CLI Commands

```bash
# Evaluate SOAP agent on default test case
python -m medical_facts_evaluation --mode soap --agent-a <soap_agent_id> --verbose-soap

# Evaluate on all test cases
python -m medical_facts_evaluation --mode soap --agent-a <id> --all-test-cases --verbose-soap

# Compare two SOAP agents
python -m medical_facts_evaluation --mode soap --compare --agent-a <id1> --agent-b <id2> --verbose-soap

# Custom thresholds
python -m medical_facts_evaluation --mode soap --agent-a <id> \
  --soap-structure-threshold 0.75 --soap-content-threshold 0.60
```

### SOAP Agent IDs

| Agent | ID | Description |
|-------|----|-------------|
| SOAP v1 | `be95d9821ef211f194964348756e437e` | First SOAP generation agent |
| SOAP v2 | `f44320e61ef011f194964348756e437e` | Improved SOAP generation agent |

### Verbose SOAP Output

When `--verbose-soap` is enabled, the output includes:
- Per-section scores (lexical, GPT-semantic, RAGAS, effective blend)
- Predicted vs Gold item comparison for each S/O/A/P section
- Missing and extra items highlighted
- Weakness summary identifying lowest-scoring sections
- Top actionable fixes to improve agent prompts

### Gold Standard SOAP Files

Gold SOAP references are stored in `medical_facts_evaluation/gold_soap/` as `.soap.json` files, auto-generated from test case ground truth via the SOAP formatter. These should be reviewed by a clinician for production use.

---

## Web UI (Streamlit Dashboard)

An interactive web interface for running evaluations without the command line.

### Launch

```bash
# From the RAGAS_Medical_Facts_Agent_refactor directory
streamlit run app.py
```

### Features

- **Agent Evaluation**: Enter any RAGFlow agent ID and run medical facts or SOAP evaluation
- **Test Case Selection**: Pick from all available test cases or run all
- **Live Results**: View detailed evaluation output with section scores and diagnostics
- **Agent Favorites**: Save frequently-used agents with custom labels for quick access
- **Ranking Leaderboard**: Automatic ranking table sorted by quality score and response time
- **Export**: Download results as JSON

---

## Known Limitations

- **Exact String Matching**: Medication names require exact matches. "Esomep" ≠ "Esomeprazol"
- **German Only**: Test cases and prompts are in German
- **GPT-5.x Support**: Requires RAGAS 0.5+ (current 0.4.3 has a bug with `max_completion_tokens`)


Marcel Cheng - [marcelcheng-prog](https://github.com/marcelcheng-prog)
