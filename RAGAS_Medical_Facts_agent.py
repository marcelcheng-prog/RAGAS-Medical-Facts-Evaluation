#!/usr/bin/env python3
"""
Medical Facts Agent Production Evaluation
==========================================
Evaluates the Medical Facts Agent in RAGFlow using RAGAS framework.

Tests the agent at: http://172.17.16.150/api/v1/agents_openai/{agent_id}/chat/completions
Model: phi-4

Usage:
    python RAGAS_Medical_Facts_agent.py                              # Run evaluation (default agent)
    python RAGAS_Medical_Facts_agent.py --verbose                    # Show full output
    python RAGAS_Medical_Facts_agent.py --iterations 5               # Run 5 times for consistency check
    python RAGAS_Medical_Facts_agent.py --agent-a <id>               # Test specific agent
    python RAGAS_Medical_Facts_agent.py --compare --agent-a <id> --agent-b <id>  # Compare two agents

Date: January 2026
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

import pandas as pd
import requests
from openai import OpenAI

# RAGAS imports (will show deprecation warnings once at startup)
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, answer_relevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# OpenAI for RAGAS evaluation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# YOUR DEPLOYED MEDICAL FACTS AGENT
MEDICAL_FACTS_AUTH_TOKEN = "ragflow-ILlQy5xh_VthK0H45ia5CqWaftzWnxKMtLqllFdt_2k"
MEDICAL_FACTS_MODEL = "phi-4"

# Output
RESULTS_DIR = Path(r"P:\MCH\Ragas\results\medical_facts_production")

# STRICT Quality Thresholds for Production
THRESHOLDS = {
    # RAGAS metrics
    "faithfulness": 0.90,
    "context_recall": 0.85,
    "answer_relevancy": 0.80,
    
    # Critical safety metrics
    "medication_precision": 0.95,
    "medication_recall": 0.90,
    "hallucination_score": 0.98,
    "null_value_score": 0.60,
    
    # Field accuracy
    "vital_signs_accuracy": 0.95,
    "action_classification": 0.90,
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Medication:
    """Medication with full context"""
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    action: Optional[str] = None
    reason: Optional[str] = None
    indication: Optional[str] = None


@dataclass
class GroundTruth:
    """Hand-labeled ground truth for Medical Facts"""
    medications_taken: List[Medication]
    medications_planned: List[Medication]
    all_medication_names: List[str]
    vital_measurements: List[Dict[str, str]]
    symptoms: List[str]
    medical_history: List[str]
    diagnostic_plans: List[str]
    therapeutic_interventions: List[str]
    forbidden_medications: List[str] = field(default_factory=list)


@dataclass
class MedicationEvaluation:
    """Detailed medication extraction evaluation"""
    expected_total: int
    found_total: int
    correct_matches: int
    missing_medications: List[str]
    extra_medications: List[str]
    hallucinations: List[str]
    name_precision: float
    name_recall: float
    dose_accuracy: float
    frequency_accuracy: float
    action_accuracy: float
    null_doses: List[str]
    null_frequencies: List[str]
    f1_score: float


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    test_name: str
    timestamp: str
    api_time_seconds: float
    api_error: Optional[str]
    agent_output_raw: str
    agent_facts: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None
    medication_eval: Optional[MedicationEvaluation] = None
    vital_signs_accuracy: float = 0.0
    symptoms_completeness: float = 0.0
    diagnostic_plans_accuracy: float = 0.0
    critical_hallucinations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed: bool = False
    quality_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)


# =============================================================================
# GROUND TRUTH - MICHAEL TEST CASE
# =============================================================================

def create_michael_ground_truth() -> Tuple[str, GroundTruth]:
    """Create Michael test case with hand-labeled ground truth"""
    
    transcript = """Dr. Hausmann: „So, Herr Müller. Lange nicht gesehen. Wie geht es Ihnen? Haben Sie das Parkplatzproblem draußen gesehen? Katastrophe heute." Herr Müller: „Ach hören Sie mir auf. Ich musste drei Runden drehen. Aber gut, ich bin ja da. Eigentlich geht es mir ganz gut, nur der Rücken zwickt und mein Zucker macht was er will." Dr. Hausmann: „Okay, schauen wir mal drauf. Nehmen Sie das Metformin noch regelmäßig? Die 1000er?" Herr Müller: „Ja, morgens und abends eine. Aber meine Frau meinte, ich soll mal dieses Ozempic probieren, das nimmt ihre Schwester zum Abnehmen." Dr. Hausmann: „Moment, eins nach dem anderen. Also Metformin 1000 mg, 1-0-1, das bleibt so. Das steht fest. Zu Ozempic: Nein, das verschreibe ich Ihnen heute nicht. Ihre Werte sind dafür nicht schlecht genug und wir haben Lieferengpässe. Vergessen Sie das erst mal wieder." Herr Müller: „Schade. Naja. Und wegen dem Rücken? Ich habe noch diese Tropfen von letztem Jahr... Tramadol oder so?" Dr. Hausmann: „Haben Sie die noch zu Hause?" Herr Müller: „Ja, da ist noch eine halbe Flasche." Dr. Hausmann: „Bitte entsorgen Sie die sofort. Tramadol ist für Ihre jetzigen Beschwerden viel zu stark und macht schwindelig. Nehmen Sie die auf keinen Fall weiter." Herr Müller: „Okay, weg damit. Was dann?" Dr. Hausmann: „Ich schreibe Ihnen Ibuprofen 600 auf... nein, warten Sie. Ich sehe gerade in der Akte, Ihre Nierenwerte waren grenzwertig. GFR unter 60. Da traue ich mich nicht an die 600er ran. Wir machen Folgendes: Ich verschreibe Ihnen Novalgin Tropfen. Metamizol." Herr Müller: „Die bitteren?" Dr. Hausmann: „Genau die. Nehmen Sie bei Bedarf 20 bis 30 Tropfen. Aber maximal viermal am Tag. Nicht mehr." Herr Müller: „Alles klar. Und mein Blutdruck? Das Ramipril?" Dr. Hausmann: „Wie viel nehmen Sie da gerade?" Herr Müller: „Die 5 mg morgens." Dr. Hausmann: „Der Druck ist heute mit 160 zu 90 deutlich zu hoch. Wir verdoppeln das. Ab morgen nehmen Sie bitte die Ramipril 10 mg. Ich gebe Ihnen ein neues Rezept mit." Herr Müller: „Also die 5er wegwerfen?" Dr. Hausmann: „Nein, nehmen Sie einfach zwei von den 5ern, bis die Packung leer ist, danach die neuen 10er. Und ich brauche noch ein Labor. Schwester Ute kommt gleich." Herr Müller: „Gut. Ach, eine Sache noch: Ich fliege nächste Woche nach Mallorca. Brauche ich da eine Thrombosespritze?" Dr. Hausmann: „Für Mallorca? Nein, der Flug ist zu kurz. Trinken Sie viel Wasser, bewegen Sie die Beine. Keine Spritze notwendig."""

    ground_truth = GroundTruth(
        medications_taken=[
            Medication(name="Metformin", dose="1000 mg", frequency="1-0-1", indication="Diabetes"),
            Medication(name="Ramipril", dose="5 mg", frequency="morgens", indication="Blutdruck"),
        ],
        medications_planned=[
            Medication(name="Ozempic", action="refused", reason="Werte nicht schlecht genug, Lieferengpässe"),
            Medication(name="Tramadol", action="stopped", reason="Zu stark, macht schwindelig"),
            Medication(name="Ibuprofen", action="refused", reason="Grenzwertige Nierenwerte (GFR <60)"),
            Medication(name="Novalgin Tropfen", action="new", dose="20-30 Tropfen", 
                      frequency="bei Bedarf max 4x täglich", indication="Rückenschmerzen"),
            Medication(name="Ramipril", action="changed", dose="10 mg", 
                      frequency="täglich", reason="Blutdruck zu hoch"),
        ],
        all_medication_names=[
            "Metformin", "Ozempic", "Tramadol", "Ibuprofen", 
            "Novalgin Tropfen", "Metamizol", "Ramipril"
        ],
        vital_measurements=[
            {"parameter": "Blutdruck", "value": "160/90", "unit": "mmHg", "source": "doctor_measured"}
        ],
        symptoms=[
            "Rücken zwickt",
            "Blutzucker macht was er will"
        ],
        medical_history=[
            "Grenzwertige Nierenwerte (GFR <60)"
        ],
        diagnostic_plans=[
            "Labor"
        ],
        therapeutic_interventions=[
            "Tramadol-Reste sofort entsorgen",
            "Viel Wasser trinken auf Flug",
            "Beine bewegen auf Flug"
        ],
        forbidden_medications=[
            "Risperidon", "Diazepam", "Citalopram", "Alprazolam", 
            "Lorazepam", "Sertralin", "Fluoxetin"
        ]
    )
    
    return transcript, ground_truth


# =============================================================================
# API CLIENT
# =============================================================================

class MedicalFactsClient:
    """Client for deployed Medical Facts Agent"""
    
    def __init__(self, api_url: str, auth_token: str, model: str):
        self.api_url = api_url
        self.auth_token = auth_token
        self.model = model
        self.console = Console() if RICH_AVAILABLE else None
    
    def extract_facts(self, transcript: str, retries: int = 3) -> Tuple[str, float, Optional[str]]:
        """Call deployed Medical Facts Agent. Returns: (content, api_time, error)"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": transcript}]
        }
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(retries + 1):
            try:
                start_time = time.time()
                
                if self.console:
                    self.console.print(f"  [cyan]📡 API call attempt {attempt + 1}/{retries + 1}...[/cyan]")
                
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
                api_time = time.time() - start_time
                response.raise_for_status()
                
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if not content.strip():
                    if attempt < retries:
                        time.sleep(5)
                        continue
                    return "", api_time, "Empty response"
                
                if self.console:
                    self.console.print(f"  [green]✅ Response received ({len(content)} chars)[/green]")
                
                return content, api_time, None
                
            except requests.exceptions.Timeout:
                if attempt < retries:
                    time.sleep(3)
                    continue
                return "", 0, "Request timeout (120s)"
                
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    time.sleep(3)
                    continue
                return "", 0, f"API error: {str(e)}"
        
        return "", 0, "Max retries exceeded"


# =============================================================================
# PARSING
# =============================================================================

def parse_medical_facts(content: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse Medical Facts JSON from agent response."""
    if not content:
        return None, "Empty response"
    
    try:
        return json.loads(content), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_client_for_agent(agent_id: str) -> MedicalFactsClient:
    """Create Medical Facts client for specific agent ID"""
    base_url = "http://172.17.16.150/api/v1/agents_openai"
    api_url = f"{base_url}/{agent_id}/chat/completions"
    
    return MedicalFactsClient(
        api_url=api_url,
        auth_token=MEDICAL_FACTS_AUTH_TOKEN,
        model=MEDICAL_FACTS_MODEL
    )


def normalize_med_name(name: str) -> str:
    """Normalize medication name for comparison"""
    return name.lower().strip()


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_medications(
    agent_facts: Dict[str, Any],
    ground_truth: GroundTruth,
    transcript: str
) -> MedicationEvaluation:
    """Comprehensive medication evaluation."""
    
    agent_taken = agent_facts.get('medications_taken', []) if agent_facts else []
    agent_planned = agent_facts.get('medications_planned', []) if agent_facts else []
    agent_all_names = agent_facts.get('medications_names', []) if agent_facts else []
    
    expected_names = set(normalize_med_name(m) for m in ground_truth.all_medication_names)
    found_names = set(normalize_med_name(m) for m in agent_all_names if m)
    
    correct_matches = expected_names & found_names
    missing = expected_names - found_names
    extra = found_names - expected_names
    
    # Check for hallucinations (NOT in transcript)
    transcript_lower = transcript.lower()
    hallucinations = []
    
    for name in found_names:
        if name not in transcript_lower:
            original = next((m for m in agent_all_names if normalize_med_name(m) == name), name)
            if original.lower() not in transcript_lower:
                hallucinations.append(original)
    
    # Check forbidden medications (CRITICAL SAFETY)
    forbidden = set(normalize_med_name(m) for m in ground_truth.forbidden_medications)
    critical_hallucinations = found_names & forbidden
    if critical_hallucinations:
        hallucinations.extend(list(critical_hallucinations))
    
    # Field-level accuracy
    null_doses = []
    null_frequencies = []
    dose_scores = []
    freq_scores = []
    action_scores = []
    
    for expected in ground_truth.medications_planned:
        expected_norm = normalize_med_name(expected.name)
        
        matching = None
        for found in agent_planned:
            if normalize_med_name(found.get('name', '')) == expected_norm:
                matching = found
                break
        
        if matching:
            if expected.dose:
                if matching.get('dose') == expected.dose:
                    dose_scores.append(1.0)
                elif matching.get('dose') is None or matching.get('dose') == "":
                    dose_scores.append(0.0)
                    null_doses.append(f"{expected.name}.dose")
                else:
                    dose_scores.append(0.5)
            
            if expected.frequency:
                if matching.get('frequency') == expected.frequency:
                    freq_scores.append(1.0)
                elif matching.get('frequency') is None or matching.get('frequency') == "":
                    freq_scores.append(0.0)
                    null_frequencies.append(f"{expected.name}.frequency")
                else:
                    freq_scores.append(0.5)
            
            if expected.action:
                if matching.get('action') == expected.action:
                    action_scores.append(1.0)
                else:
                    action_scores.append(0.0)
    
    def avg(scores):
        return sum(scores) / len(scores) if scores else 1.0
    
    precision = len(correct_matches) / len(found_names) if found_names else 0
    recall = len(correct_matches) / len(expected_names) if expected_names else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return MedicationEvaluation(
        expected_total=len(expected_names),
        found_total=len(found_names),
        correct_matches=len(correct_matches),
        missing_medications=sorted(list(missing)),
        extra_medications=sorted(list(extra)),
        hallucinations=sorted(hallucinations),
        name_precision=precision,
        name_recall=recall,
        dose_accuracy=avg(dose_scores),
        frequency_accuracy=avg(freq_scores),
        action_accuracy=avg(action_scores),
        null_doses=null_doses,
        null_frequencies=null_frequencies,
        f1_score=f1
    )


def evaluate_vital_signs(agent_facts: Dict, ground_truth: GroundTruth) -> float:
    """Evaluate vital signs extraction"""
    if not agent_facts:
        return 0.0
    
    expected_vs = ground_truth.vital_measurements
    found_vs = agent_facts.get('vital_measurements', [])
    
    if not expected_vs:
        return 1.0
    
    correct = 0
    for expected in expected_vs:
        for found in found_vs:
            if (found.get('parameter') == expected['parameter'] and
                found.get('value') == expected['value']):
                correct += 1
                break
    
    return correct / len(expected_vs)


def evaluate_symptoms(agent_facts: Dict, ground_truth: GroundTruth) -> float:
    """Evaluate symptom extraction"""
    if not agent_facts:
        return 0.0
    
    expected = set(s.lower() for s in ground_truth.symptoms)
    found = set(s.lower() for s in agent_facts.get('symptoms', []))
    
    if not expected:
        return 1.0
    
    matches = expected & found
    return len(matches) / len(expected)


# =============================================================================
# RAGAS EVALUATION
# =============================================================================

def setup_ragas(openai_client: OpenAI):
    """Initialize RAGAS with modern llm_factory approach."""
    print("  Creating LLM via llm_factory...")
    
    llm = llm_factory(
        model=OPENAI_MODEL,
        provider="openai",
        client=openai_client,
        max_tokens=10000
    )
    
    print(f"  ✅ LLM ready: {llm}")
    
    print("  Creating embeddings...")
    embeddings = OpenAIEmbeddings(
        client=openai_client,
        model="text-embedding-ada-002"
    )
    print("  ✅ Embeddings ready")
    
    return llm, embeddings


def run_ragas_evaluation(
    transcript: str,
    agent_output: str,
    ground_truth: GroundTruth,
    ragas_components: Tuple
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Run RAGAS evaluation using LEGACY API."""
    print("  Starting RAGAS evaluation...")
    
    try:
        from ragas.metrics import faithfulness, context_recall, answer_relevancy
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        
        llm, embeddings = ragas_components
        llm.max_tokens = 8000
        print("  Creating evaluation sample...")
        
        sample = SingleTurnSample(
            user_input=transcript,
            response=agent_output,
            retrieved_contexts=[transcript],
            reference=json.dumps({
                "medications": ground_truth.all_medication_names,
                "symptoms": ground_truth.symptoms,
                "vital_signs": [f"{v['parameter']}: {v['value']}" 
                               for v in ground_truth.vital_measurements]
            }, ensure_ascii=False)
        )
        
        dataset = EvaluationDataset(samples=[sample])
        
        print("  Running evaluate() with legacy metrics...")
        print(f"    Dataset features: {dataset.features()}")
        
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_recall, answer_relevancy], 
            llm=llm,
            embeddings=embeddings
        )
        
        print("  Extracting scores...")
        df = results.to_pandas()
        print(f"    Result columns: {list(df.columns)}")
        
        faith_score = None
        recall_score = None
        relevancy_score = None
        
        if 'faithfulness' in df.columns:
            val = df['faithfulness'].iloc[0]
            faith_score = float(val) if pd.notna(val) else None
            
        if 'context_recall' in df.columns:
            val = df['context_recall'].iloc[0]
            recall_score = float(val) if pd.notna(val) else None
            
        if 'answer_relevancy' in df.columns:
            val = df['answer_relevancy'].iloc[0]
            relevancy_score = float(val) if pd.notna(val) else None
        
        if faith_score is not None:
            print(f"    ✅ Faithfulness: {faith_score:.1%}")
        if recall_score is not None:
            print(f"    ✅ Context Recall: {recall_score:.1%}")
        if relevancy_score is not None:
            print(f"    ✅ Answer Relevancy: {relevancy_score:.1%}")
        
        return (faith_score, recall_score, relevancy_score)
        
    except Exception as e:
        print(f"  ❌ RAGAS failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_medical_facts_agent(
    transcript: str,
    ground_truth: GroundTruth,
    client: MedicalFactsClient,
    ragas_metrics: Tuple,
    verbose: bool = False
) -> EvaluationResult:
    """Evaluate Medical Facts Agent on test case"""
    
    console = Console() if RICH_AVAILABLE else None
    
    if console:
        console.print(Panel.fit(
            "[bold cyan]🔬 Evaluating Medical Facts Agent[/bold cyan]",
            border_style="cyan"
        ))
    
    # Call agent
    if console:
        console.print("\n[cyan]📡 Calling deployed Medical Facts Agent...[/cyan]")
    
    agent_output, api_time, api_error = client.extract_facts(transcript)
    
    if console:
        console.print(f"[green]⏱️  API response time: {api_time:.2f}s[/green]")
    
    if api_error:
        if console:
            console.print(f"[red]❌ API Error: {api_error}[/red]")
        
        return EvaluationResult(
            test_name="Michael_Production",
            timestamp=datetime.now().isoformat(),
            api_time_seconds=api_time,
            api_error=api_error,
            agent_output_raw=agent_output,
            agent_facts=None,
            parse_error=None,
            passed=False,
            failure_reasons=["API Error"]
        )
    
    # Parse response
    agent_facts, parse_error = parse_medical_facts(agent_output)
    
    if verbose and console:
        console.print("\n[bold]📄 Agent Output:[/bold]")
        syntax = Syntax(json.dumps(agent_facts, indent=2, ensure_ascii=False) if agent_facts else agent_output[:500], 
                       "json", theme="monokai", line_numbers=True)
        console.print(syntax)
    
    if parse_error:
        if console:
            console.print(f"[red]❌ Parse Error: {parse_error}[/red]")
    
    # Evaluate medications
    if console:
        console.print("\n[cyan]💊 Evaluating medications...[/cyan]")
    
    med_eval = evaluate_medications(agent_facts, ground_truth, transcript)
    
    def status(score, threshold):
        return "✅" if score >= threshold else "❌"
    
    if console:
        med_table = Table(title="💊 Medication Extraction", box=box.ROUNDED)
        med_table.add_column("Metric", style="cyan")
        med_table.add_column("Value", style="green")
        med_table.add_column("Threshold", style="yellow")
        med_table.add_column("Status", style="white")
        
        med_table.add_row(
            "Precision", 
            f"{med_eval.name_precision:.1%}",
            f"{THRESHOLDS['medication_precision']:.1%}",
            status(med_eval.name_precision, THRESHOLDS['medication_precision'])
        )
        med_table.add_row(
            "Recall",
            f"{med_eval.name_recall:.1%}",
            f"{THRESHOLDS['medication_recall']:.1%}",
            status(med_eval.name_recall, THRESHOLDS['medication_recall'])
        )
        med_table.add_row(
            "F1 Score",
            f"{med_eval.f1_score:.1%}",
            "-",
            ""
        )
        med_table.add_row(
            "Dose Accuracy",
            f"{med_eval.dose_accuracy:.1%}",
            "-",
            ""
        )
        med_table.add_row(
            "Action Classification",
            f"{med_eval.action_accuracy:.1%}",
            f"{THRESHOLDS['action_classification']:.1%}",
            status(med_eval.action_accuracy, THRESHOLDS['action_classification'])
        )
        
        console.print(med_table)
        
        if med_eval.hallucinations:
            console.print(f"\n[red bold]🚨 HALLUCINATIONS: {', '.join(med_eval.hallucinations)}[/red bold]")
        if med_eval.missing_medications:
            console.print(f"[yellow]⚠️  Missing: {', '.join(med_eval.missing_medications)}[/yellow]")
        if med_eval.null_doses:
            console.print(f"[yellow]⚠️  Null doses: {', '.join(med_eval.null_doses)}[/yellow]")
    
    # Evaluate other fields
    vital_acc = evaluate_vital_signs(agent_facts, ground_truth)
    symptom_comp = evaluate_symptoms(agent_facts, ground_truth)
    
    # Run RAGAS
    if console:
        console.print("\n[cyan]📊 Running RAGAS evaluation...[/cyan]")
    
    faithfulness_score, context_recall_score, answer_relevancy_score = run_ragas_evaluation(
        transcript, agent_output, ground_truth, ragas_metrics
    )
    
    # FIXED: Print RAGAS table
    if console:
        ragas_table = Table(title="📊 RAGAS Scores", box=box.ROUNDED)
        ragas_table.add_column("Metric", style="cyan")
        ragas_table.add_column("Score", style="green")
        ragas_table.add_column("Threshold", style="yellow")
        ragas_table.add_column("Status", style="white")
        
        if faithfulness_score is not None:
            ragas_table.add_row(
                "Faithfulness",
                f"{faithfulness_score:.1%}",
                f"{THRESHOLDS['faithfulness']:.1%}",
                status(faithfulness_score, THRESHOLDS['faithfulness'])
            )
        else:
            ragas_table.add_row("Faithfulness", "Failed", "-", "⚠️")
            
        if context_recall_score is not None:
            ragas_table.add_row(
                "Context Recall",
                f"{context_recall_score:.1%}",
                f"{THRESHOLDS['context_recall']:.1%}",
                status(context_recall_score, THRESHOLDS['context_recall'])
            )
        else:
            ragas_table.add_row("Context Recall", "Failed", "-", "⚠️")
            
        if answer_relevancy_score is not None:
            ragas_table.add_row(
                "Answer Relevancy",
                f"{answer_relevancy_score:.1%}",
                f"{THRESHOLDS['answer_relevancy']:.1%}",
                status(answer_relevancy_score, THRESHOLDS['answer_relevancy'])
            )
        else:
            ragas_table.add_row("Answer Relevancy", "Failed (embeddings)", "-", "⚠️")
        
        console.print(ragas_table)  # <-- FIXED: This was missing!
    
    # Determine pass/fail
    failure_reasons = []
    critical_hallucinations = []
    warnings = []
    
    if api_error:
        failure_reasons.append(f"API Error: {api_error}")
    
    if parse_error:
        failure_reasons.append(f"Parse Error: {parse_error}")
    
    # Critical safety check
    if med_eval.hallucinations:
        for h in med_eval.hallucinations:
            if any(h.lower() in f.lower() for f in ground_truth.forbidden_medications):
                critical_hallucinations.append(f"CRITICAL: {h}")
                failure_reasons.append(f"CRITICAL HALLUCINATION: {h}")
            else:
                warnings.append(f"Hallucination: {h}")
                failure_reasons.append(f"Hallucination: {h}")
    
    # Check thresholds
    checks = {
        "Medication Precision": (med_eval.name_precision, THRESHOLDS['medication_precision']),
        "Medication Recall": (med_eval.name_recall, THRESHOLDS['medication_recall']),
        "Action Classification": (med_eval.action_accuracy, THRESHOLDS['action_classification']),
        "Vital Signs": (vital_acc, THRESHOLDS['vital_signs_accuracy']),
    }
    
    if faithfulness_score is not None:
        checks["Faithfulness"] = (faithfulness_score, THRESHOLDS['faithfulness'])
    if context_recall_score is not None:
        checks["Context Recall"] = (context_recall_score, THRESHOLDS['context_recall'])
    
    for metric_name, (score, threshold) in checks.items():
        if score < threshold:
            failure_reasons.append(f"{metric_name}: {score:.1%} < {threshold:.1%}")
    
    if med_eval.null_doses:
        warnings.append(f"Null doses: {', '.join(med_eval.null_doses)}")
    if med_eval.null_frequencies:
        warnings.append(f"Null frequencies: {', '.join(med_eval.null_frequencies)}")
    
    passed = len(failure_reasons) == 0
    
    # Quality score (0-100)
    quality_components = [
        med_eval.f1_score * 0.30,
        (1.0 - len(med_eval.hallucinations) / max(med_eval.found_total, 1)) * 0.25,
        med_eval.action_accuracy * 0.15,
        vital_acc * 0.10,
    ]
    
    if faithfulness_score is not None:
        quality_components.append(faithfulness_score * 0.15)
    if context_recall_score is not None:
        quality_components.append(context_recall_score * 0.05)
    
    quality_score = sum(quality_components) * 100
    
    result = EvaluationResult(
        test_name="Michael_Production",
        timestamp=datetime.now().isoformat(),
        api_time_seconds=api_time,
        api_error=api_error,
        agent_output_raw=agent_output,
        agent_facts=agent_facts,
        parse_error=parse_error,
        faithfulness=faithfulness_score,
        context_recall=context_recall_score,
        answer_relevancy=answer_relevancy_score,
        medication_eval=med_eval,
        vital_signs_accuracy=vital_acc,
        symptoms_completeness=symptom_comp,
        critical_hallucinations=critical_hallucinations,
        warnings=warnings,
        passed=passed,
        quality_score=quality_score,
        failure_reasons=failure_reasons
    )
    
    # Final verdict
    if console:
        if passed:
            console.print(f"\n[bold green]✅ AGENT PASSED ALL CHECKS[/bold green]")
            console.print(f"[green]Quality Score: {quality_score:.1f}/100[/green]")
        else:
            console.print(f"\n[bold red]❌ AGENT FAILED[/bold red]")
            console.print(f"[red]Quality Score: {quality_score:.1f}/100[/red]")
            
            if critical_hallucinations:
                console.print(f"\n[bold red]🚨 CRITICAL SAFETY ISSUES:[/bold red]")
                for issue in critical_hallucinations:
                    console.print(f"  [red]• {issue}[/red]")
            
            console.print(f"\n[bold]Failure Reasons:[/bold]")
            for reason in failure_reasons:
                console.print(f"  [yellow]• {reason}[/yellow]")
            
            if warnings:
                console.print(f"\n[bold]Warnings:[/bold]")
                for warning in warnings:
                    console.print(f"  [dim]• {warning}[/dim]")
    
    return result


# =============================================================================
# AGENT COMPARISON
# =============================================================================

def run_agent_comparison(
    transcript: str,
    ground_truth: GroundTruth,
    agent_a_id: str,
    agent_b_id: str,
    ragas_components: Tuple,
    output_dir: Path,
    verbose: bool = False
):
    """Compare two Medical Facts agents"""
    
    console = Console() if RICH_AVAILABLE else None
    
    if console:
        console.print(f"\n[bold magenta]🆚 Comparing Two Agents[/bold magenta]\n")
        console.print(f"[cyan]Agent A:[/cyan] {agent_a_id}")
        console.print(f"[cyan]Agent B:[/cyan] {agent_b_id}")
    
    client_a = create_client_for_agent(agent_a_id)
    client_b = create_client_for_agent(agent_b_id)
    
    # Evaluate Agent A
    if console:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Testing Agent A: {agent_a_id[:12]}...[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    
    result_a = evaluate_medical_facts_agent(
        transcript, ground_truth, client_a, ragas_components, verbose=verbose
    )
    
    # Evaluate Agent B
    if console:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Testing Agent B: {agent_b_id[:12]}...[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    
    result_b = evaluate_medical_facts_agent(
        transcript, ground_truth, client_b, ragas_components, verbose=verbose
    )
    
    # Generate comparison
    comparison = {
        "agent_a_id": agent_a_id,
        "agent_b_id": agent_b_id,
        "timestamp": datetime.now().isoformat(),
        "test_name": "Michael_Production",
        "agent_a": {
            "passed": result_a.passed,
            "quality_score": result_a.quality_score,
            "api_time": result_a.api_time_seconds,
            "faithfulness": result_a.faithfulness,
            "context_recall": result_a.context_recall,
            "medication_precision": result_a.medication_eval.name_precision if result_a.medication_eval else None,
            "medication_recall": result_a.medication_eval.name_recall if result_a.medication_eval else None,
            "hallucinations": result_a.medication_eval.hallucinations if result_a.medication_eval else [],
            "missing_medications": result_a.medication_eval.missing_medications if result_a.medication_eval else [],
            "failure_reasons": result_a.failure_reasons
        },
        "agent_b": {
            "passed": result_b.passed,
            "quality_score": result_b.quality_score,
            "api_time": result_b.api_time_seconds,
            "faithfulness": result_b.faithfulness,
            "context_recall": result_b.context_recall,
            "medication_precision": result_b.medication_eval.name_precision if result_b.medication_eval else None,
            "medication_recall": result_b.medication_eval.name_recall if result_b.medication_eval else None,
            "hallucinations": result_b.medication_eval.hallucinations if result_b.medication_eval else [],
            "missing_medications": result_b.medication_eval.missing_medications if result_b.medication_eval else [],
            "failure_reasons": result_b.failure_reasons
        }
    }
    
    # Determine winner
    if result_a.quality_score > result_b.quality_score:
        comparison["winner"] = "agent_a"
        comparison["winner_id"] = agent_a_id
    elif result_b.quality_score > result_a.quality_score:
        comparison["winner"] = "agent_b"
        comparison["winner_id"] = agent_b_id
    else:
        comparison["winner"] = "tie"
        comparison["winner_id"] = None
    
    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"comparison_{timestamp}.json"
    
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    # Save individual results
    save_results([result_a], output_dir / f"agent_a_{agent_a_id[:8]}")
    save_results([result_b], output_dir / f"agent_b_{agent_b_id[:8]}")
    
    # Print comparison summary
    if console:
        print_comparison_summary(comparison, console)
    
    print(f"\n💾 Comparison report saved: {comparison_path}")


def print_comparison_summary(comparison: Dict, console: Console):
    """Print comparison summary for Medical Facts agents"""
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]🆚 COMPARISON SUMMARY[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    if comparison["winner"] == "agent_a":
        console.print(f"[bold green]🏆 Winner: Agent A ({comparison['agent_a_id'][:12]}...)[/bold green]")
    elif comparison["winner"] == "agent_b":
        console.print(f"[bold green]🏆 Winner: Agent B ({comparison['agent_b_id'][:12]}...)[/bold green]")
    else:
        console.print(f"[bold yellow]🤝 Result: Tie[/bold yellow]")
    
    table = Table(title="📊 Metric Comparison", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Agent A", style="green")
    table.add_column("Agent B", style="green")
    table.add_column("Better", style="yellow")
    
    def better(a, b, lower_is_better=False):
        if a is None or b is None:
            return "-"
        if lower_is_better:
            return "A ✅" if a < b else "B ✅" if b < a else "="
        return "A ✅" if a > b else "B ✅" if b > a else "="
    
    a = comparison["agent_a"]
    b = comparison["agent_b"]
    
    table.add_row(
        "Quality Score",
        f"{a['quality_score']:.1f}",
        f"{b['quality_score']:.1f}",
        better(a['quality_score'], b['quality_score'])
    )
    
    table.add_row(
        "Passed",
        "✅" if a['passed'] else "❌",
        "✅" if b['passed'] else "❌",
        "A ✅" if a['passed'] and not b['passed'] else "B ✅" if b['passed'] and not a['passed'] else "="
    )
    
    if a['faithfulness'] is not None and b['faithfulness'] is not None:
        table.add_row(
            "Faithfulness",
            f"{a['faithfulness']:.1%}",
            f"{b['faithfulness']:.1%}",
            better(a['faithfulness'], b['faithfulness'])
        )
    
    if a['context_recall'] is not None and b['context_recall'] is not None:
        table.add_row(
            "Context Recall",
            f"{a['context_recall']:.1%}",
            f"{b['context_recall']:.1%}",
            better(a['context_recall'], b['context_recall'])
        )
    
    if a['medication_precision'] is not None and b['medication_precision'] is not None:
        table.add_row(
            "Med Precision",
            f"{a['medication_precision']:.1%}",
            f"{b['medication_precision']:.1%}",
            better(a['medication_precision'], b['medication_precision'])
        )
    
    if a['medication_recall'] is not None and b['medication_recall'] is not None:
        table.add_row(
            "Med Recall",
            f"{a['medication_recall']:.1%}",
            f"{b['medication_recall']:.1%}",
            better(a['medication_recall'], b['medication_recall'])
        )
    
    table.add_row(
        "API Time",
        f"{a['api_time']:.1f}s",
        f"{b['api_time']:.1f}s",
        better(a['api_time'], b['api_time'], lower_is_better=True)
    )
    
    table.add_row(
        "Hallucinations",
        str(len(a['hallucinations'])),
        str(len(b['hallucinations'])),
        better(len(a['hallucinations']), len(b['hallucinations']), lower_is_better=True)
    )
    
    console.print(table)
    
    # Missing medications detail
    if a['missing_medications'] or b['missing_medications']:
        console.print(f"\n[bold]Missing Medications:[/bold]")
        if a['missing_medications']:
            console.print(f"  [yellow]Agent A missing: {', '.join(a['missing_medications'])}[/yellow]")
        if b['missing_medications']:
            console.print(f"  [yellow]Agent B missing: {', '.join(b['missing_medications'])}[/yellow]")
    
    # Failure details
    if a['failure_reasons']:
        console.print(f"\n[yellow]Agent A Issues:[/yellow]")
        for reason in a['failure_reasons']:
            console.print(f"  • {reason}")
    
    if b['failure_reasons']:
        console.print(f"\n[yellow]Agent B Issues:[/yellow]")
        for reason in b['failure_reasons']:
            console.print(f"  • {reason}")
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")


# =============================================================================
# CONSISTENCY CHECK
# =============================================================================

def run_consistency_check(
    transcript: str,
    ground_truth: GroundTruth,
    client: MedicalFactsClient,
    ragas_metrics: Tuple,
    iterations: int = 5
) -> List[EvaluationResult]:
    """Run agent multiple times to check consistency."""
    console = Console() if RICH_AVAILABLE else None
    
    if console:
        console.print(f"\n[bold magenta]🔄 Running Consistency Check ({iterations} iterations)[/bold magenta]\n")
    
    results = []
    
    for i in range(iterations):
        if console:
            console.print(f"\n[cyan]Iteration {i+1}/{iterations}...[/cyan]")
        else:
            print(f"Iteration {i+1}/{iterations}...")
        
        result = evaluate_medical_facts_agent(
            transcript, ground_truth, client, ragas_metrics, verbose=False
        )
        results.append(result)
    
    if console:
        console.print("\n[bold cyan]📊 Consistency Analysis[/bold cyan]\n")
        
        quality_scores = [r.quality_score for r in results]
        passed_count = sum(1 for r in results if r.passed)
        
        consistency_table = Table(title="Consistency Metrics", box=box.ROUNDED)
        consistency_table.add_column("Metric", style="cyan")
        consistency_table.add_column("Value", style="green")
        
        consistency_table.add_row("Iterations", str(iterations))
        consistency_table.add_row("Passed", f"{passed_count}/{iterations} ({passed_count/iterations:.1%})")
        consistency_table.add_row("Avg Quality Score", f"{sum(quality_scores)/len(quality_scores):.1f}")
        consistency_table.add_row("Min Quality Score", f"{min(quality_scores):.1f}")
        consistency_table.add_row("Max Quality Score", f"{max(quality_scores):.1f}")
        consistency_table.add_row("Std Dev", f"{pd.Series(quality_scores).std():.2f}")
        
        console.print(consistency_table)
        
        if pd.Series(quality_scores).std() > 10:
            console.print("\n[yellow]⚠️  High variance detected - agent produces inconsistent results[/yellow]")
        else:
            console.print("\n[green]✅ Agent produces consistent results[/green]")
    
    return results


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results: List[EvaluationResult], output_dir: Path):
    """Save evaluation results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"medical_facts_production_{timestamp}.json"
    
    results_data = []
    for r in results:
        result_dict = asdict(r)
        if r.medication_eval:
            result_dict['medication_eval'] = asdict(r.medication_eval)
        results_data.append(result_dict)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {json_path}")
    
    return json_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Production Medical Facts Agent"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=1,
        help="Number of iterations for consistency check (default: 1)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare two agents side-by-side"
    )
    parser.add_argument(
        "--agent-a",
        type=str,
        default="df4cb87efd2011f0b3234afd40f7103b",
        help="First agent ID (default: current production agent)"
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        help="Second agent ID for comparison (required with --compare)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Output directory (default: {RESULTS_DIR})"
    )
    
    args = parser.parse_args()
    
    # Validate comparison mode
    if args.compare and not args.agent_b:
        parser.error("--compare requires --agent-b to be specified")
    
    console = Console() if RICH_AVAILABLE else None
    
    if console:
        if args.compare:
            console.print(Panel.fit(
                "[bold cyan]🆚 Medical Facts Agent Comparison[/bold cyan]\n"
                f"[dim]Agent A: {args.agent_a}[/dim]\n"
                f"[dim]Agent B: {args.agent_b}[/dim]\n"
                f"[dim]Model: {MEDICAL_FACTS_MODEL}[/dim]",
                border_style="cyan"
            ))
        else:
            console.print(Panel.fit(
                "[bold cyan]🏥 Medical Facts Agent Production Evaluation[/bold cyan]\n"
                f"[dim]Deployed Agent: {MEDICAL_FACTS_MODEL}[/dim]\n"
                f"[dim]Agent ID: {args.agent_a}[/dim]",
                border_style="cyan"
            ))
    
    # Check OpenAI key
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("   Set with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize
    if console:
        console.print("\n[cyan]🔧 Initializing...[/cyan]")
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    if console:
        console.print("[green]✅ OpenAI client ready[/green]")
    
    # Initialize RAGAS
    if console:
        console.print("[cyan]📊 Initializing RAGAS...[/cyan]")
    
    ragas_components = setup_ragas(openai_client)
    
    if console:
        console.print("[green]✅ RAGAS ready[/green]")
    
    # Load test case
    transcript, ground_truth = create_michael_ground_truth()
    
    if console:
        console.print(f"\n[cyan]📋 Test: Michael Müller (Diabetes, Back Pain)[/cyan]")
        console.print(f"[dim]Expected medications: {len(ground_truth.all_medication_names)} ({', '.join(ground_truth.all_medication_names)})[/dim]")
    
    # Run comparison or single evaluation
    if args.compare:
        run_agent_comparison(
            transcript,
            ground_truth,
            args.agent_a,
            args.agent_b,
            ragas_components,
            Path(args.output_dir),
            verbose=args.verbose
        )
    else:
        agent_client = create_client_for_agent(args.agent_a)
        
        if console:
            console.print("[green]✅ Medical Facts Agent client ready[/green]")
        
        if args.iterations > 1:
            results = run_consistency_check(
                transcript,
                ground_truth,
                agent_client,
                ragas_components,
                args.iterations
            )
        else:
            result = evaluate_medical_facts_agent(
                transcript,
                ground_truth,
                agent_client,
                ragas_components,
                verbose=args.verbose
            )
            results = [result]
        
        save_results(results, Path(args.output_dir))
        
        failed_count = sum(1 for r in results if not r.passed)
        sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()