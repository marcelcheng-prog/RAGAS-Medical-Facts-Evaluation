"""
Medication evaluation metrics.

Comprehensive evaluation of medication extraction accuracy.
"""

from typing import Any, Optional

from ..models.ground_truth import GroundTruth
from ..models.evaluation import MedicationEvaluation


def normalize_med_name(name: str) -> str:
    """Normalize medication name for comparison."""
    return name.lower().strip()


def evaluate_medications(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
    transcript: str,
) -> MedicationEvaluation:
    """
    Comprehensive medication evaluation.
    
    Evaluates:
    - Medication name precision and recall
    - Dose accuracy
    - Frequency accuracy
    - Action classification accuracy
    - Hallucination detection
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        transcript: Original transcript for hallucination detection
        
    Returns:
        MedicationEvaluation with detailed results
    """
    if not agent_facts:
        return MedicationEvaluation.empty(len(ground_truth.all_medication_names))
    
    # Extract from agent response
    agent_taken = agent_facts.get('medications_taken', [])
    agent_planned = agent_facts.get('medications_planned', [])
    agent_all_names = agent_facts.get('medications_names', [])
    
    # If medications_names is empty, extract names from medications_taken and medications_planned
    if not agent_all_names:
        extracted_names = set()
        
        # Extract from medications_taken
        for med in agent_taken:
            if isinstance(med, dict) and med.get('name'):
                extracted_names.add(med['name'])
            elif isinstance(med, str):
                extracted_names.add(med)
        
        # Extract from medications_planned
        for med in agent_planned:
            if isinstance(med, dict) and med.get('name'):
                extracted_names.add(med['name'])
            elif isinstance(med, str):
                extracted_names.add(med)
        
        agent_all_names = list(extracted_names)
    
    # Normalize names for comparison
    expected_names = set(normalize_med_name(m) for m in ground_truth.all_medication_names)
    found_names = set(normalize_med_name(m) for m in agent_all_names if m)
    
    # Calculate matches
    correct_matches = expected_names & found_names
    missing = expected_names - found_names
    extra = found_names - expected_names
    
    # Detect hallucinations (medications not in transcript)
    transcript_lower = transcript.lower()
    hallucinations = _detect_hallucinations(
        found_names=found_names,
        agent_all_names=agent_all_names,
        transcript_lower=transcript_lower,
    )
    
    # Check for forbidden medications (CRITICAL SAFETY)
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
        
        # Find matching medication in agent response
        matching = None
        for found in agent_planned:
            if normalize_med_name(found.get('name', '')) == expected_norm:
                matching = found
                break
        
        if matching:
            # Evaluate dose
            if expected.dose:
                if matching.get('dose') == expected.dose:
                    dose_scores.append(1.0)
                elif matching.get('dose') is None or matching.get('dose') == "":
                    dose_scores.append(0.0)
                    null_doses.append(f"{expected.name}.dose")
                else:
                    dose_scores.append(0.5)  # Partial match
            
            # Evaluate frequency
            if expected.frequency:
                if matching.get('frequency') == expected.frequency:
                    freq_scores.append(1.0)
                elif matching.get('frequency') is None or matching.get('frequency') == "":
                    freq_scores.append(0.0)
                    null_frequencies.append(f"{expected.name}.frequency")
                else:
                    freq_scores.append(0.5)  # Partial match
            
            # Evaluate action classification
            if expected.action:
                if matching.get('action') == expected.action:
                    action_scores.append(1.0)
                else:
                    action_scores.append(0.0)
    
    # Calculate metrics
    precision = len(correct_matches) / len(found_names) if found_names else 0
    recall = len(correct_matches) / len(expected_names) if expected_names else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    def avg(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 1.0
    
    return MedicationEvaluation(
        expected_total=len(expected_names),
        found_total=len(found_names),
        correct_matches=len(correct_matches),
        missing_medications=sorted(list(missing)),
        extra_medications=sorted(list(extra)),
        hallucinations=sorted(set(hallucinations)),
        name_precision=precision,
        name_recall=recall,
        dose_accuracy=avg(dose_scores),
        frequency_accuracy=avg(freq_scores),
        action_accuracy=avg(action_scores),
        null_doses=null_doses,
        null_frequencies=null_frequencies,
        f1_score=f1,
    )


def _detect_hallucinations(
    found_names: set[str],
    agent_all_names: list[str],
    transcript_lower: str,
) -> list[str]:
    """
    Detect medications not present in the transcript.
    
    Args:
        found_names: Normalized medication names found by agent
        agent_all_names: Original medication names from agent
        transcript_lower: Lowercase transcript for matching
        
    Returns:
        List of hallucinated medication names
    """
    hallucinations = []
    
    for name in found_names:
        if name not in transcript_lower:
            # Get original casing
            original = next(
                (m for m in agent_all_names if normalize_med_name(m) == name),
                name
            )
            if original.lower() not in transcript_lower:
                hallucinations.append(original)
    
    return hallucinations
