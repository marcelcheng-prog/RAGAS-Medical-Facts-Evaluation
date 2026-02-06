"""
Symptom evaluation metrics.
"""

from typing import Any, Optional

from ..models.ground_truth import GroundTruth


def _is_symptom_match(expected: str, found: str) -> bool:
    """
    Check if a found symptom matches an expected symptom.
    
    Uses substring matching to handle cases like:
    - "Bauchschmerzen" matches "Starken Bauchschmerzen"
    - "Diarrhoe" matches "Diarrhoe"
    
    Args:
        expected: Expected symptom (lowercase)
        found: Found symptom (lowercase)
        
    Returns:
        True if match, False otherwise
    """
    # Exact match
    if expected == found:
        return True
    # Expected is substring of found (e.g., "bauchschmerzen" in "starken bauchschmerzen")
    if expected in found:
        return True
    # Found is substring of expected
    if found in expected:
        return True
    return False


def evaluate_symptoms(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
) -> float:
    """
    Evaluate symptom extraction completeness.
    
    Uses substring matching to allow for adjective variations in German
    (e.g., "Starken Bauchschmerzen" matches "Bauchschmerzen").
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        
    Returns:
        Completeness score from 0.0 to 1.0
    """
    if not agent_facts:
        return 0.0
    
    expected = [s.lower() for s in ground_truth.symptoms]
    found = [s.lower() for s in agent_facts.get('symptoms', [])]
    
    if not expected:
        return 1.0  # No symptoms expected
    
    # Count how many expected symptoms have a match in found
    matched_count = 0
    for exp in expected:
        for f in found:
            if _is_symptom_match(exp, f):
                matched_count += 1
                break
    
    return matched_count / len(expected)


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
    
    expected_original = ground_truth.symptoms
    found_original = agent_facts.get('symptoms', [])
    expected = [s.lower() for s in expected_original]
    found = [s.lower() for s in found_original]
    
    if not expected:
        return {
            "completeness": 1.0,
            "expected_count": 0,
            "found_count": len(found),
            "matched": [],
            "missing": [],
            "extra": found_original,
        }
    
    # Track which expected symptoms were matched
    matched_indices = set()
    matched_found_indices = set()
    
    for i, exp in enumerate(expected):
        for j, f in enumerate(found):
            if _is_symptom_match(exp, f):
                matched_indices.add(i)
                matched_found_indices.add(j)
                break
    
    matched_original = [expected_original[i] for i in matched_indices]
    missing_original = [expected_original[i] for i in range(len(expected)) if i not in matched_indices]
    extra_original = [found_original[j] for j in range(len(found)) if j not in matched_found_indices]
    
    return {
        "completeness": len(matched_indices) / len(expected),
        "expected_count": len(expected),
        "found_count": len(found),
        "matched": matched_original,
        "missing": missing_original,
        "extra": extra_original,
    }
