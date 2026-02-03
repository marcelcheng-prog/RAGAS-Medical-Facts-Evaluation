"""
Medical Facts Evaluation Package
================================
Modular evaluation framework for Medical Facts extraction agents.

Usage:
    python -m medical_facts_evaluation --agent-a <id> --verbose
    python -m medical_facts_evaluation --compare --agent-a <id> --agent-b <id>
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "MedicalFactsEvaluator":
        from .evaluator import MedicalFactsEvaluator
        return MedicalFactsEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MedicalFactsEvaluator"]
