---
layout: default
title: Home
---

# 🏥 RAGAS Medical Facts & SOAP Evaluation

Evaluation framework for testing **Medical Facts extraction** and **SOAP note generation** agents using [RAGAS](https://docs.ragas.io/), with a **Streamlit web UI** for interactive evaluation, agent comparison, and gold standard review.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔬 **RAGAS Evaluation** | Industry-standard LLM evaluation metrics (Faithfulness, Context Recall) |
| 💊 **Medication Tracking** | Precision / Recall / F1 for medication extraction with hallucination detection |
| 🧼 **SOAP Evaluation** | Three-tier scoring: lexical + GPT-semantic + RAGAS weighted blend |
| 🆚 **Agent Comparison** | Side-by-side comparison of two agents on the same test cases |
| 📖 **Gold Standard Review** | Browse all transcripts, gold SOAP, and ground truth with feedback |
| 🏆 **Leaderboard** | Automatic ranking by score and response time |
| ⭐ **Favorites** | Save and manage preferred agent configurations |
| 💬 **Feedback System** | Per-test-case feedback with auto-recommendations |
| 📊 **Web UI** | Full Streamlit dashboard at `localhost:8501` |

---

## Quick Start

```bash
# Clone
git clone https://github.com/marcelcheng-prog/RAGAS-Medical-Facts-Evaluation.git
cd RAGAS-Medical-Facts-Evaluation

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env   # add your OPENAI_API_KEY and MEDICAL_FACTS_AUTH_TOKEN

# Run CLI
python -m medical_facts_evaluation --verbose

# Run Web UI
python -m streamlit run app.py --server.headless true
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Web UI (app.py)               │
│  ┌──────────┬───────────┬──────────┬──────────┬────────┐│
│  │ Evaluate │ Data      │ Gold Std │ Rankings │Feedback││
│  │          │ Browser   │ Review   │          │        ││
│  └──────────┴───────────┴──────────┴──────────┴────────┘│
├─────────────────────────────────────────────────────────┤
│                  Evaluation Engine                        │
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │ MedicalFacts     │  │ SOAP Evaluator              │   │
│  │ Evaluator        │  │ (lexical + GPT + RAGAS)     │   │
│  └────────┬────────┘  └──────────┬──────────────────┘   │
│           │                      │                       │
│  ┌────────▼────────┐  ┌─────────▼──────────┐           │
│  │ RAGFlow API     │  │ OpenAI GPT-4       │           │
│  │ (Agent Client)  │  │ (Semantic Scoring)  │           │
│  └─────────────────┘  └────────────────────┘           │
├─────────────────────────────────────────────────────────┤
│                  Test Data                                │
│  test_cases/*.json  │  gold_soap/*.soap.json  │ data/   │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluation Modes

### Medical Facts Mode

Extracts structured medical facts from doctor-patient transcripts and evaluates:

- **Medication extraction** — name, dose, frequency, action (continued/new/changed/stopped)
- **RAGAS Faithfulness** — are extracted facts supported by the transcript?
- **RAGAS Context Recall** — are all ground truth facts captured?
- **Hallucination detection** — medications the agent invented

### SOAP Mode

Generates Subjective / Objective / Assessment / Plan notes and evaluates with a three-tier blend:

| Tier | Weight | What it measures |
|------|--------|-----------------|
| Lexical (token F1) | 35% | Word-level overlap with gold standard |
| GPT-Semantic | 30% | Meaning-level similarity scored by GPT |
| RAGAS Context Recall | 35% | Semantic list recall via RAGAS |

**Thresholds:**
- **Structure Threshold** — minimum fraction of S/O/A/P sections present (default: 1.0 = all required)
- **Content Threshold** — minimum average content quality across sections (default: 0.70)

---

## Quality Metrics (QM Codes)

The framework validates agents against 18 quality metric codes across three categories:

### Safety (QM-001 – QM-008)

| Code | What it checks |
|------|---------------|
| QM-001 | Drug name integrity (verbatim extraction) |
| QM-002 | Dosage & unit correctness (mg vs µg) |
| QM-003 | Existing vs. new medication classification |
| QM-004 | Rejected medication filtering |
| QM-005 | Uncertainty handling ([UNCLEAR] markers) |
| QM-006 | Laterality (left/right) correctness |
| QM-007 | Allergy (CAVE) capture |
| QM-008 | Negation handling ("no fever" ≠ "fever") |

### Accuracy (QM-009 – QM-013)

| Code | What it checks |
|------|---------------|
| QM-009 | Primary diagnosis identification |
| QM-010 | Vital sign / lab value precision |
| QM-011 | Attribution (patient vs. third party) |
| QM-012 | Temporal accuracy ("3 days" vs "3 weeks") |
| QM-013 | Self-correction recognition |

### Usability (QM-014 – QM-018)

| Code | What it checks |
|------|---------------|
| QM-014 | SOAP section formatting |
| QM-015 | Noise / smalltalk filtering |
| QM-016 | Medical terminology style |
| QM-017 | Language & grammar quality |
| QM-018 | Hallucination / filler detection |

---

## Test Cases

13 German medical transcripts with ground truth:

| Test Case | Description | Meds | Complexity |
|-----------|-------------|:----:|:----------:|
| `ortho_knee_arthrose` | Knee arthritis, NSAID allergy, opioid refusal | 8 | High |
| `gyn_pregnancy_gestational_diabetes` | Pregnancy SSW 28, gestational diabetes, 3 allergies | 11 | High |
| `diabetes_hypertonie` | Diabetes + Hypertension with family history | 7 | High |
| `medikamentenreview_polypharmazie` | Elderly polypharmacy, heart failure | 7 | High |
| `hausarzt` | Complete GP consultation | 5 | High |
| `korrektur_noise_filter` | Corrections & smalltalk filtering | 4 | High |
| `michael_mueller` | Diabetes & back pain | 8 | Medium |
| `diabetes` | Diabetes therapy change | 7 | Medium |
| `magenschmerzen_gastritis` | Gastritis with PPI therapy | 5 | Medium |
| `bauchschmerzen` | Abdominal pain & fever | 5 | Medium |
| `notfall_brustschmerzen_vitals` | Emergency chest pain with vitals | — | High |
| `psychiatrie_depression_symptome` | Psychiatry depression symptoms | — | Medium |
| `634` | General consultation | — | Medium |

---

## Predefined Agents

| Label | Agent ID | Purpose |
|-------|----------|---------|
| **DEV-FLH agent** | `ff04c2b01edb11f194964348756e437e` | Development / FLH testing |
| **medicalfactssoapv2** | `f44320e61ef011f194964348756e437e` | SOAP v2 production agent |
| **Production MF** | `e1a25a64fdc611f0b3cb4afd40f7103b` | Production Medical Facts agent |

---

## Web UI Tabs

### 📊 Evaluate
Run Medical Facts or SOAP evaluations with configurable thresholds. Supports single agent or side-by-side comparison.

### 📋 Data Browser
Browse transcripts, ground truth, gold SOAP, and past evaluation outputs per test case.

### 📖 Gold Standard Review
Review all test cases at a glance — transcript, ground truth medical facts, and gold SOAP side by side. Leave prioritized feedback on what should be improved.

### 🏆 Rankings
Agent leaderboard sorted by average/best score, pass rate, and response time.

### 💬 Feedback
Browse all feedback entries with filtering by agent, mode, and test case. Export as JSON.

### ⭐ Favorites
Save and manage preferred agent configurations for quick access.

---

## Pages

- [Home](index) — this page
- [QM Codes Reference](qm-codes) — detailed quality metric definitions
- [Web UI Guide](web-ui-guide) — how to use the Streamlit dashboard
