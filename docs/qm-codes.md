---
layout: default
title: QM Codes Reference
---

# Quality Metrics (QM) Codes Reference

The evaluation framework validates agent output against 18 quality metric codes organized into three severity categories.

## Severity Levels

| Level | Icon | Description |
|-------|:----:|-------------|
| **PASS** | 🟢 | Akzeptiert — meets quality standards |
| **MAJOR** | 🟡 | Nacharbeit nötig — requires manual review/correction |
| **CRITICAL** | 🔴 | Showstopper — patient safety risk, must be fixed |

---

## SAFETY — Medikation & Patientendaten (QM-001 to QM-008)

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

## ACCURACY — Diagnostik & Kontext (QM-009 to QM-013)

| ID | Kategorie | Metrik / Prüfpunkt | 🟢 PASS | 🟡 MAJOR | 🔴 CRITICAL |
|----|-----------|-------------------|---------|----------|-------------|
| **QM-009** | Diagnostik | **Hauptdiagnose** | Kernproblem korrekt identifiziert | Unpräzise ("Unwohlsein" statt "Grippe") | Thema verfehlt / Falsche Diagnose erfunden |
| **QM-010** | Diagnostik | **Messwerte (Vitals/Labs)** | Werte exakt (RR 120/80) | Wert da, Parametername unklar | Zahlenfehler (Kommafehler) |
| **QM-011** | Kontext | **Attribution (Wer?)** | Symptome Dritter (Ehefrau) ignoriert | Dritte erwähnt, aber abgegrenzt | Symptome Dritter dem Patienten zugeordnet |
| **QM-012** | Kontext | **Zeitlicher Verlauf** | Zeitangaben ("seit 3 Tagen") korrekt | Zeit fehlt ("seit einiger Zeit") | Falsche Zeit ("seit 3 Wochen" statt "3 Tagen") |
| **QM-013** | Kontext | **Korrektur-Erkennung** | Letztgültige Aussage ("nein doch nicht") zählt | Beide Aussagen (falsch & richtig) gelistet | Nur die falsche (korrigierte) Aussage übernommen |

---

## USABILITY — Struktur, Inhalt & Stil (QM-014 to QM-018)

| ID | Kategorie | Metrik / Prüfpunkt | 🟢 PASS | 🟡 MAJOR | 🔴 CRITICAL |
|----|-----------|-------------------|---------|----------|-------------|
| **QM-014** | Struktur | **SOAP-Formatierung** | Saubere Trennung S-O-A-P | Infos in falscher Sektion (z.B. Befund in Plan) | Keine Struktur / Fließtext-Block |
| **QM-015** | Inhalt | **Noise Filter (Smalltalk)** | Kein Smalltalk (Wetter/Urlaub) | Kurzer Satz Smalltalk enthalten | Lange Passagen über Irrelevantes (Urlaub, Admin) |
| **QM-016** | Stil | **Medizinischer Jargon** | Fachsprache / Stichpunkte | Umgangssprache / Ganze Sätze | Chatbot-Stil ("Der Arzt sagte dann...") |
| **QM-017** | Stil | **Sprache & Grammatik** | Korrekte deutsche Grammatik | Leichte Grammatikfehler | Englische Wörter gemischt / Sinn entstellt |
| **QM-018** | Inhalt | **Halluzination (Füller)** | Nur Fakten aus Audio | — | Erfundene Untersuchungen ("Abdomen weich"), die nie stattfanden |

---

## Test Case Coverage

| Test Case | Primary QM Codes | Description |
|-----------|-----------------|-------------|
| `ortho_knee_arthrose` | QM-002 through QM-008 | Comprehensive safety metrics |
| `gyn_pregnancy_gestational_diabetes` | QM-002 through QM-008 | Pregnancy-specific safety |
| `korrektur_noise_filter` | QM-013, QM-015 | Correction & noise handling |
| `diabetes_hypertonie` | QM-001, QM-003, QM-010 | Medication & vitals |
| `medikamentenreview_polypharmazie` | QM-001, QM-009 | Polypharmacy & ASR errors |
| `hausarzt` | QM-001, QM-003, QM-011 | Complete GP consultation |
| `michael_mueller` | QM-001, QM-003, QM-004 | Standard diabetes case |

---

[← Back to Home](index)
