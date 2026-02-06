"""
Quality thresholds for Medical Facts Evaluation.

Defines pass/fail thresholds for different environments.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    """Quality thresholds for evaluation metrics."""
    
    # RAGAS metrics
    faithfulness: float = 0.90
    context_recall: float = 0.85
    answer_relevancy: float = 0.80
    
    # Safety-critical medication metrics
    medication_precision: float = 0.95
    medication_recall: float = 0.90
    hallucination_score: float = 0.98
    
    # Field accuracy
    vital_signs_accuracy: float = 0.95
    symptoms_completeness: float = 0.80
    family_history_completeness: float = 0.80
    action_classification: float = 0.90
    null_value_score: float = 0.60
    
    # Performance
    max_api_time_seconds: float = 60.0
    
    def as_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return {
            "faithfulness": self.faithfulness,
            "context_recall": self.context_recall,
            "answer_relevancy": self.answer_relevancy,
            "medication_precision": self.medication_precision,
            "medication_recall": self.medication_recall,
            "hallucination_score": self.hallucination_score,
            "vital_signs_accuracy": self.vital_signs_accuracy,
            "symptoms_completeness": self.symptoms_completeness,
            "family_history_completeness": self.family_history_completeness,
            "action_classification": self.action_classification,
            "null_value_score": self.null_value_score,
            "max_api_time_seconds": self.max_api_time_seconds,
        }


# Pre-configured threshold profiles
PRODUCTION = QualityThresholds()

DEVELOPMENT = QualityThresholds(
    faithfulness=0.80,
    context_recall=0.75,
    medication_precision=0.80,
    medication_recall=0.75,
    hallucination_score=0.90,
    vital_signs_accuracy=0.85,
    symptoms_completeness=0.70,
    family_history_completeness=0.70,
    action_classification=0.80,
)

STRICT = QualityThresholds(
    faithfulness=0.95,
    context_recall=0.90,
    answer_relevancy=0.85,
    medication_precision=0.98,
    medication_recall=0.95,
    hallucination_score=0.99,
    vital_signs_accuracy=0.98,
    action_classification=0.95,
)
