"""
Evaluation result data models.

These dataclasses capture the results of evaluating a Medical Facts agent.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any


@dataclass
class MedicationEvaluation:
    """Detailed medication extraction evaluation results."""
    
    # Counts
    expected_total: int
    found_total: int
    correct_matches: int
    
    # Detailed lists
    missing_medications: list[str] = field(default_factory=list)
    extra_medications: list[str] = field(default_factory=list)
    hallucinations: list[str] = field(default_factory=list)
    
    # Accuracy scores (0.0 to 1.0)
    name_precision: float = 0.0
    name_recall: float = 0.0
    f1_score: float = 0.0
    
    dose_accuracy: float = 1.0
    frequency_accuracy: float = 1.0
    action_accuracy: float = 1.0
    
    # Null value tracking
    null_doses: list[str] = field(default_factory=list)
    null_frequencies: list[str] = field(default_factory=list)
    
    @classmethod
    def empty(cls, expected_total: int = 0) -> "MedicationEvaluation":
        """Create an empty evaluation (no agent response)."""
        return cls(
            expected_total=expected_total,
            found_total=0,
            correct_matches=0,
            name_precision=0.0,
            name_recall=0.0,
            f1_score=0.0,
        )
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a test case."""
    
    # Metadata
    test_name: str
    timestamp: str
    agent_id: str = ""
    
    # API performance
    api_time_seconds: float = 0.0
    api_error: Optional[str] = None
    
    # Raw output
    agent_output_raw: str = ""
    agent_facts: Optional[dict[str, Any]] = None
    parse_error: Optional[str] = None
    
    # RAGAS scores (0.0 to 1.0, None if failed)
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None
    
    # Detailed evaluations
    medication_eval: Optional[MedicationEvaluation] = None
    vital_signs_accuracy: float = 0.0
    symptoms_completeness: float = 0.0
    family_history_completeness: float = 0.0
    diagnostic_plans_accuracy: float = 0.0
    
    # Safety
    critical_hallucinations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    # Overall verdict
    passed: bool = False
    quality_score: float = 0.0
    failure_reasons: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.medication_eval:
            result['medication_eval'] = self.medication_eval.to_dict()
        return result
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical safety issues."""
        return len(self.critical_hallucinations) > 0
    
    @property
    def summary(self) -> str:
        """Short summary string for logging."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"{self.test_name}: {status} (Score: {self.quality_score:.1f}/100)"
