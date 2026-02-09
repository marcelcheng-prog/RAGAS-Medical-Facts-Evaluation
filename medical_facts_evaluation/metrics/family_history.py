"""
Family history (Familienanamnese) evaluation metrics.
"""

from typing import Any, Optional

from ..models.ground_truth import GroundTruth


def normalize_family_history_item(item: str) -> str:
    """Normalize family history item for comparison."""
    return item.lower().strip()


def evaluate_family_history(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
) -> float:
    """
    Evaluate family history extraction completeness.
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        
    Returns:
        Completeness score from 0.0 to 1.0
    """
    if not agent_facts:
        return 0.0 if ground_truth.family_history else 1.0
    
    expected = set(normalize_family_history_item(item) for item in ground_truth.family_history)
    found = set(normalize_family_history_item(item) for item in agent_facts.get('family_history', []))
    
    if not expected:
        return 1.0  # No family history expected, consider it correct
    
    # Calculate recall (what percentage of expected items were found)
    matches = 0
    for exp_item in expected:
        for found_item in found:
            # Use substring matching for flexibility (German medical terms can vary)
            if exp_item in found_item or found_item in exp_item:
                matches += 1
                break
    
    return matches / len(expected)


def evaluate_family_history_detailed(
    agent_facts: Optional[dict[str, Any]],
    ground_truth: GroundTruth,
) -> dict[str, Any]:
    """
    Detailed family history evaluation with individual results.
    
    Args:
        agent_facts: Parsed JSON from agent response
        ground_truth: Expected ground truth data
        
    Returns:
        Dictionary with detailed evaluation results including precision and recall
    """
    if not agent_facts:
        return {
            "completeness": 0.0 if ground_truth.family_history else 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "expected_count": len(ground_truth.family_history),
            "found_count": 0,
            "matched": [],
            "missing": ground_truth.family_history.copy(),
            "extra": [],
        }
    
    expected = ground_truth.family_history
    found = agent_facts.get('family_history', [])
    expected_norm = set(normalize_family_history_item(item) for item in expected)
    found_norm = set(normalize_family_history_item(item) for item in found)
    
    if not expected:
        return {
            "completeness": 1.0,
            "precision": 1.0 if not found else 0.0,  # Precision is 0 if extra items found when none expected
            "recall": 1.0,
            "f1_score": 1.0 if not found else 0.0,
            "expected_count": 0,
            "found_count": len(found),
            "matched": [],
            "missing": [],
            "extra": found,
        }
    
    # Find matches using substring matching
    matched_expected = []
    matched_found = []
    
    for exp_item in expected:
        exp_norm = normalize_family_history_item(exp_item)
        for found_item in found:
            found_norm_item = normalize_family_history_item(found_item)
            if exp_norm in found_norm_item or found_norm_item in exp_norm:
                if exp_item not in matched_expected:
                    matched_expected.append(exp_item)
                if found_item not in matched_found:
                    matched_found.append(found_item)
                break
    
    missing = [item for item in expected if item not in matched_expected]
    extra = [item for item in found if item not in matched_found]
    
    # Calculate metrics
    precision = len(matched_found) / len(found) if found else 0.0
    recall = len(matched_expected) / len(expected) if expected else 1.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "completeness": recall,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "expected_count": len(expected),
        "found_count": len(found),
        "matched": matched_expected,
        "missing": missing,
        "extra": extra,
    }
