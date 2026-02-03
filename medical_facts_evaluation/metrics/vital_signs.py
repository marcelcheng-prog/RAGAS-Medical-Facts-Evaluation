"""
Vital signs evaluation metrics.
"""

from typing import Any, Optional

from ..models.ground_truth import GroundTruth


def evaluate_vital_signs(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
) -> float:
    """
    Evaluate vital signs extraction accuracy.
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        
    Returns:
        Accuracy score from 0.0 to 1.0
    """
    if not agent_facts:
        return 0.0
    
    expected_vs = ground_truth.vital_measurements
    found_vs = agent_facts.get('vital_measurements', [])
    
    if not expected_vs:
        return 1.0  # No vitals expected, consider it correct
    
    correct = 0
    for expected in expected_vs:
        for found in found_vs:
            if (found.get('parameter') == expected.parameter and
                found.get('value') == expected.value):
                correct += 1
                break
    
    return correct / len(expected_vs)


def evaluate_vital_signs_detailed(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
) -> dict[str, Any]:
    """
    Detailed vital signs evaluation with individual results.
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        
    Returns:
        Dictionary with detailed evaluation results
    """
    if not agent_facts:
        return {
            "accuracy": 0.0,
            "expected_count": len(ground_truth.vital_measurements),
            "found_count": 0,
            "matches": [],
            "missing": [v.parameter for v in ground_truth.vital_measurements],
        }
    
    expected_vs = ground_truth.vital_measurements
    found_vs = agent_facts.get('vital_measurements', [])
    
    if not expected_vs:
        return {
            "accuracy": 1.0,
            "expected_count": 0,
            "found_count": len(found_vs),
            "matches": [],
            "missing": [],
        }
    
    matches = []
    missing = []
    
    for expected in expected_vs:
        found_match = False
        for found in found_vs:
            if (found.get('parameter') == expected.parameter and
                found.get('value') == expected.value):
                matches.append({
                    "parameter": expected.parameter,
                    "expected": expected.value,
                    "found": found.get('value'),
                })
                found_match = True
                break
        
        if not found_match:
            missing.append(expected.parameter)
    
    return {
        "accuracy": len(matches) / len(expected_vs),
        "expected_count": len(expected_vs),
        "found_count": len(found_vs),
        "matches": matches,
        "missing": missing,
    }
