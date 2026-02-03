"""
Symptom evaluation metrics.
"""

from typing import Any, Optional

from ..models.ground_truth import GroundTruth


def evaluate_symptoms(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
) -> float:
    """
    Evaluate symptom extraction completeness.
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        
    Returns:
        Completeness score from 0.0 to 1.0
    """
    if not agent_facts:
        return 0.0
    
    expected = set(s.lower() for s in ground_truth.symptoms)
    found = set(s.lower() for s in agent_facts.get('symptoms', []))
    
    if not expected:
        return 1.0  # No symptoms expected
    
    matches = expected & found
    return len(matches) / len(expected)


def evaluate_symptoms_detailed(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
) -> dict[str, Any]:
    """
    Detailed symptom evaluation with individual results.
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        
    Returns:
        Dictionary with detailed evaluation results
    """
    if not agent_facts:
        return {
            "completeness": 0.0,
            "expected_count": len(ground_truth.symptoms),
            "found_count": 0,
            "matched": [],
            "missing": ground_truth.symptoms.copy(),
            "extra": [],
        }
    
    expected = set(s.lower() for s in ground_truth.symptoms)
    found = set(s.lower() for s in agent_facts.get('symptoms', []))
    found_original = agent_facts.get('symptoms', [])
    
    if not expected:
        return {
            "completeness": 1.0,
            "expected_count": 0,
            "found_count": len(found),
            "matched": [],
            "missing": [],
            "extra": found_original,
        }
    
    matched = expected & found
    missing = expected - found
    extra = found - expected
    
    # Get original casing for results
    matched_original = [s for s in ground_truth.symptoms if s.lower() in matched]
    missing_original = [s for s in ground_truth.symptoms if s.lower() in missing]
    extra_original = [s for s in found_original if s.lower() in extra]
    
    return {
        "completeness": len(matched) / len(expected),
        "expected_count": len(expected),
        "found_count": len(found),
        "matched": matched_original,
        "missing": missing_original,
        "extra": extra_original,
    }
