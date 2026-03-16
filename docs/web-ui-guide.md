---
layout: default
title: Web UI Guide
---

# Streamlit Web UI Guide

The Streamlit dashboard provides an interactive interface for running evaluations, comparing agents, reviewing gold standards, and tracking feedback.

## Launch

```bash
cd RAGAS_Medical_Facts_Agent_refactor
python -m streamlit run app.py --server.headless true
```

Open **http://localhost:8501** in your browser.

---

## Sidebar Configuration

### Agent Selection

Choose from predefined agents via dropdown:

| Label | Agent ID |
|-------|----------|
| DEV-FLH agent | `ff04c2b01edb11f194964348756e437e` |
| medicalfactssoapv2 agent | `f44320e61ef011f194964348756e437e` |

Or select **✏️ Custom Agent ID** to enter any agent ID manually. Saved favorites also appear in the dropdown.

### Compare Mode

Enable **🆚 Compare two agents** to select a second agent. Results are displayed side-by-side with a summary table showing wins/losses.

### Thresholds (SOAP mode)

- **Structure Threshold** — minimum fraction of SOAP sections (S, O, A, P) present. Default: 1.0 (all 4 required).
- **Content Threshold** — minimum average content quality score. Default: 0.70.

---

## Tabs

### 📊 Evaluate

1. Select evaluation mode (Medical Facts or SOAP)
2. Choose agent(s) and test case(s)
3. Click **🚀 Run Evaluation**
4. View detailed results with metrics, auto-recommendations, and feedback forms

**Medical Facts results include:**
- Quality score (0–100)
- Medication precision, recall, F1
- Faithfulness and context recall
- Hallucination list

**SOAP results include:**
- Overall score (structure × 30% + content × 70%)
- Per-section scores (S, O, A, P) with lexical, semantic, and RAGAS breakdown
- Missing/extra items per section

### 📋 Data Browser

Browse per test case:
- Editable transcript
- Ground truth medical facts (formatted + raw JSON)
- Gold standard SOAP (formatted, plain text, raw JSON)
- Auto-derived SOAP from ground truth
- Past evaluation output files

### 📖 Gold Standard Review

Overview of **all** test cases with:
- Summary metrics (total cases, gold files, reviewed count)
- Filter bar and "show only unreviewed" toggle
- For each test case:
  - Resizable transcript viewer
  - Ground truth medical facts (left column)
  - Gold standard SOAP (right column)
  - Previous review feedback
  - Feedback form with target (transcript / ground truth / gold SOAP / general), free text, and priority slider

Feedback is saved to `data/gold_review/` as JSON files — one per test case.

### 🏆 Rankings

Agent leaderboard showing:
- Average, best, and worst scores
- Average and minimum response times
- Pass rate
- Last run timestamp

Filter by mode (Medical Facts / SOAP). Individual run history also available.

### 💬 Feedback

Browse all evaluation feedback entries with filters by agent, mode, and test case. Each entry shows:
- Timestamp, agent, test case, score
- User feedback text
- Auto-recommendations

Export as JSON or clear all.

### ⭐ Favorites

Manage saved agent configurations. Add via sidebar, delete from this tab.

---

## Data Storage

| Directory | Contents |
|-----------|----------|
| `data/favorites.json` | Saved agent favorites |
| `data/rankings.json` | Evaluation run history |
| `data/feedback/` | Per agent+test evaluation feedback |
| `data/gold_review/` | Gold standard review feedback |
| `results/` | Past evaluation output JSON files |

---

[← Back to Home](index)
