"""
Main Medical Facts Evaluator orchestrator.

This module coordinates all evaluation components.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any

from openai import OpenAI

from .config.settings import Settings, get_settings
from .config.thresholds import QualityThresholds, PRODUCTION
from .models.ground_truth import GroundTruth
from .models.evaluation import EvaluationResult, MedicationEvaluation
from .models.loader import TestCase, load_test_case, get_default_test_case
from .clients.medical_facts import MedicalFactsClient, parse_medical_facts
from .clients.ragas_client import RagasEvaluator, RagasScores
from .metrics import medication, vital_signs, symptoms
from .reporters.console import ConsoleReporter
from .reporters.json_reporter import JsonReporter


class MedicalFactsEvaluator:
    """Main orchestrator for Medical Facts agent evaluation."""
    
    def __init__(
        self,
        client: MedicalFactsClient,
        ragas: RagasEvaluator,
        thresholds: QualityThresholds = PRODUCTION,
        reporter: Optional[ConsoleReporter] = None,
        verbose: bool = False,
    ):
        """
        Initialize the evaluator.
        
        Args:
            client: Medical Facts API client
            ragas: RAGAS evaluator
            thresholds: Quality thresholds for pass/fail
            reporter: Console reporter (created if not provided)
            verbose: Whether to print detailed output
        """
        self.client = client
        self.ragas = ragas
        self.thresholds = thresholds
        self.verbose = verbose
        self.reporter = reporter or ConsoleReporter(verbose=verbose, thresholds=thresholds)
    
    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """
        Run full evaluation on a test case.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            Complete evaluation result
        """
        self.reporter.print_header(
            "🔬 Evaluating Medical Facts Agent",
            f"Test: {test_case.name}"
        )
        
        # Call agent
        self.reporter.print_status("📡 Calling deployed Medical Facts Agent...")
        response = self.client.extract_facts(test_case.transcript)
        
        self.reporter.print_status(
            f"⏱️  API response time: {response.api_time_seconds:.2f}s",
            "success" if response.success else "error"
        )
        
        if response.error:
            self.reporter.print_status(f"❌ API Error: {response.error}", "error")
            return EvaluationResult(
                test_name=test_case.test_id,
                timestamp=datetime.now().isoformat(),
                api_time_seconds=response.api_time_seconds,
                api_error=response.error,
                agent_output_raw=response.content,
                agent_facts=None,
                parse_error=None,
                passed=False,
                failure_reasons=["API Error"],
            )
        
        # Parse response
        agent_facts, parse_error = parse_medical_facts(response.content)
        
        if self.verbose:
            self.reporter.print_agent_output(response.content, agent_facts)
        
        if parse_error:
            self.reporter.print_status(f"❌ Parse Error: {parse_error}", "error")
        
        # Evaluate medications
        self.reporter.print_status("💊 Evaluating medications...")
        med_eval = medication.evaluate_medications(
            agent_facts, test_case.ground_truth, test_case.transcript
        )
        self.reporter.print_medication_table(med_eval)
        
        # Evaluate other fields
        vital_acc = vital_signs.evaluate_vital_signs(agent_facts, test_case.ground_truth)
        symptom_comp = symptoms.evaluate_symptoms(agent_facts, test_case.ground_truth)
        
        # Run RAGAS
        self.reporter.print_status("📊 Running RAGAS evaluation...")
        ragas_scores = self.ragas.evaluate(
            test_case.transcript,
            response.content,
            test_case.ground_truth
        )
        
        self.reporter.print_ragas_table(
            ragas_scores.faithfulness,
            ragas_scores.context_recall,
            ragas_scores.answer_relevancy,
        )
        
        # Build result
        result = self._build_result(
            test_case=test_case,
            response=response,
            agent_facts=agent_facts,
            parse_error=parse_error,
            med_eval=med_eval,
            vital_acc=vital_acc,
            symptom_comp=symptom_comp,
            ragas_scores=ragas_scores,
        )
        
        # Print verdict
        self.reporter.print_verdict(result)
        
        return result
    
    def _build_result(
        self,
        test_case: TestCase,
        response: Any,
        agent_facts: Optional[dict],
        parse_error: Optional[str],
        med_eval: MedicationEvaluation,
        vital_acc: float,
        symptom_comp: float,
        ragas_scores: RagasScores,
    ) -> EvaluationResult:
        """Build the final evaluation result with pass/fail determination."""
        
        failure_reasons = []
        critical_hallucinations = []
        warnings = []
        
        # Check for errors
        if response.error:
            failure_reasons.append(f"API Error: {response.error}")
        
        if parse_error:
            failure_reasons.append(f"Parse Error: {parse_error}")
        
        # Critical safety check - hallucinations
        if med_eval.hallucinations:
            for h in med_eval.hallucinations:
                if any(h.lower() in f.lower() for f in test_case.ground_truth.forbidden_medications):
                    critical_hallucinations.append(f"CRITICAL: {h}")
                    failure_reasons.append(f"CRITICAL HALLUCINATION: {h}")
                else:
                    warnings.append(f"Hallucination: {h}")
                    failure_reasons.append(f"Hallucination: {h}")
        
        # Check thresholds
        checks = {
            "Medication Precision": (med_eval.name_precision, self.thresholds.medication_precision),
            "Medication Recall": (med_eval.name_recall, self.thresholds.medication_recall),
            "Action Classification": (med_eval.action_accuracy, self.thresholds.action_classification),
            "Vital Signs": (vital_acc, self.thresholds.vital_signs_accuracy),
        }
        
        if ragas_scores.faithfulness is not None:
            checks["Faithfulness"] = (ragas_scores.faithfulness, self.thresholds.faithfulness)
        if ragas_scores.context_recall is not None:
            checks["Context Recall"] = (ragas_scores.context_recall, self.thresholds.context_recall)
        
        for metric_name, (score, threshold) in checks.items():
            if score < threshold:
                failure_reasons.append(f"{metric_name}: {score:.1%} < {threshold:.1%}")
        
        # Warnings for null values
        if med_eval.null_doses:
            warnings.append(f"Null doses: {', '.join(med_eval.null_doses)}")
        if med_eval.null_frequencies:
            warnings.append(f"Null frequencies: {', '.join(med_eval.null_frequencies)}")
        
        passed = len(failure_reasons) == 0
        
        # Calculate quality score (0-100)
        quality_components = [
            med_eval.f1_score * 0.30,
            (1.0 - len(med_eval.hallucinations) / max(med_eval.found_total, 1)) * 0.25,
            med_eval.action_accuracy * 0.15,
            vital_acc * 0.10,
        ]
        
        if ragas_scores.faithfulness is not None:
            quality_components.append(ragas_scores.faithfulness * 0.15)
        if ragas_scores.context_recall is not None:
            quality_components.append(ragas_scores.context_recall * 0.05)
        
        quality_score = sum(quality_components) * 100
        
        return EvaluationResult(
            test_name=test_case.test_id,
            timestamp=datetime.now().isoformat(),
            api_time_seconds=response.api_time_seconds,
            api_error=response.error,
            agent_output_raw=response.content,
            agent_facts=agent_facts,
            parse_error=parse_error,
            faithfulness=ragas_scores.faithfulness,
            context_recall=ragas_scores.context_recall,
            answer_relevancy=ragas_scores.answer_relevancy,
            medication_eval=med_eval,
            vital_signs_accuracy=vital_acc,
            symptoms_completeness=symptom_comp,
            critical_hallucinations=critical_hallucinations,
            warnings=warnings,
            passed=passed,
            quality_score=quality_score,
            failure_reasons=failure_reasons,
        )
    
    def run_consistency_check(
        self,
        test_case: TestCase,
        iterations: int = 5,
    ) -> list[EvaluationResult]:
        """
        Run agent multiple times to check consistency.
        
        Args:
            test_case: Test case to evaluate
            iterations: Number of iterations
            
        Returns:
            List of evaluation results
        """
        self.reporter.print_header(
            f"🔄 Running Consistency Check ({iterations} iterations)",
            test_case.name
        )
        
        results = []
        
        for i in range(iterations):
            self.reporter.print_status(f"Iteration {i+1}/{iterations}...")
            result = self.evaluate(test_case)
            results.append(result)
        
        self.reporter.print_consistency_table(results)
        
        return results
    
    @classmethod
    def from_settings(
        cls,
        agent_id: str,
        settings: Optional[Settings] = None,
        thresholds: QualityThresholds = PRODUCTION,
        verbose: bool = False,
    ) -> "MedicalFactsEvaluator":
        """
        Create evaluator from application settings.
        
        Args:
            agent_id: The agent ID to connect to
            settings: Settings object (uses default if not provided)
            thresholds: Quality thresholds
            verbose: Whether to print detailed output
            
        Returns:
            Configured MedicalFactsEvaluator
        """
        if settings is None:
            settings = get_settings()
        
        # Validate settings
        errors = settings.validate()
        if errors:
            for error in errors:
                print(f"❌ ERROR: {error}")
            raise ValueError("Invalid settings")
        
        # Create OpenAI client
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Create components
        client = MedicalFactsClient.from_settings(agent_id, settings, verbose)
        ragas = RagasEvaluator.from_settings(openai_client, settings, verbose)
        reporter = ConsoleReporter(verbose=verbose, thresholds=thresholds)
        
        return cls(
            client=client,
            ragas=ragas,
            thresholds=thresholds,
            reporter=reporter,
            verbose=verbose,
        )


def run_agent_comparison(
    agent_a_id: str,
    agent_b_id: str,
    test_case: TestCase,
    settings: Optional[Settings] = None,
    thresholds: QualityThresholds = PRODUCTION,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[EvaluationResult, EvaluationResult, dict]:
    """
    Compare two Medical Facts agents.
    
    Args:
        agent_a_id: First agent ID
        agent_b_id: Second agent ID
        test_case: Test case to evaluate
        settings: Settings object
        thresholds: Quality thresholds
        output_dir: Directory for saving results
        verbose: Whether to print detailed output
        
    Returns:
        Tuple of (result_a, result_b, comparison_dict)
    """
    if settings is None:
        settings = get_settings()
    
    reporter = ConsoleReporter(verbose=verbose, thresholds=thresholds)
    
    reporter.print_header(
        "🆚 Comparing Two Agents",
        f"Agent A: {agent_a_id[:12]}...\nAgent B: {agent_b_id[:12]}..."
    )
    
    # Evaluate Agent A
    reporter.print_status(f"\n{'='*60}")
    reporter.print_status(f"Testing Agent A: {agent_a_id[:12]}...")
    reporter.print_status('='*60)
    
    evaluator_a = MedicalFactsEvaluator.from_settings(
        agent_a_id, settings, thresholds, verbose
    )
    result_a = evaluator_a.evaluate(test_case)
    
    # Evaluate Agent B
    reporter.print_status(f"\n{'='*60}")
    reporter.print_status(f"Testing Agent B: {agent_b_id[:12]}...")
    reporter.print_status('='*60)
    
    evaluator_b = MedicalFactsEvaluator.from_settings(
        agent_b_id, settings, thresholds, verbose
    )
    result_b = evaluator_b.evaluate(test_case)
    
    # Build comparison
    comparison = JsonReporter.build_comparison_dict(
        result_a, result_b, agent_a_id, agent_b_id
    )
    
    # Print comparison table
    reporter.print_comparison_table(
        comparison["agent_a"],
        comparison["agent_b"],
        agent_a_id,
        agent_b_id,
    )
    
    # Save results if output_dir provided
    if output_dir:
        json_reporter = JsonReporter(output_dir)
        json_reporter.save_comparison(
            comparison, result_a, result_b, agent_a_id, agent_b_id
        )
    
    return result_a, result_b, comparison
