"""
JSON reporter for saving evaluation results to files.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import asdict

from ..models.evaluation import EvaluationResult


class JsonReporter:
    """JSON file reporter for evaluation results."""
    
    def __init__(self, output_dir: Path | str):
        """
        Initialize JSON reporter.
        
        Args:
            output_dir: Directory to save JSON result files
        """
        self.output_dir = Path(output_dir)
    
    def save_results(
        self,
        results: list[EvaluationResult],
        prefix: str = "medical_facts_production",
    ) -> Path:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation results
            prefix: Filename prefix
            
        Returns:
            Path to saved JSON file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"{prefix}_{timestamp}.json"
        
        results_data = []
        for r in results:
            result_dict = r.to_dict()
            results_data.append(result_dict)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {json_path}")
        
        return json_path
    
    def save_comparison(
        self,
        comparison: dict,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
        agent_a_id: str,
        agent_b_id: str,
    ) -> Path:
        """
        Save agent comparison results.
        
        Args:
            comparison: Comparison summary dict
            result_a: Full result for agent A
            result_b: Full result for agent B
            agent_a_id: Agent A identifier
            agent_b_id: Agent B identifier
            
        Returns:
            Path to saved comparison file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = self.output_dir / f"comparison_{timestamp}.json"
        
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # Save individual results
        self.save_results([result_a], f"agent_a_{agent_a_id[:8]}")
        self.save_results([result_b], f"agent_b_{agent_b_id[:8]}")
        
        print(f"\n💾 Comparison report saved: {comparison_path}")
        
        return comparison_path
    
    @staticmethod
    def build_comparison_dict(
        result_a: EvaluationResult,
        result_b: EvaluationResult,
        agent_a_id: str,
        agent_b_id: str,
    ) -> dict:
        """
        Build comparison dictionary from two results.
        
        Args:
            result_a: Evaluation result for agent A
            result_b: Evaluation result for agent B
            agent_a_id: Agent A identifier
            agent_b_id: Agent B identifier
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "agent_a_id": agent_a_id,
            "agent_b_id": agent_b_id,
            "timestamp": datetime.now().isoformat(),
            "test_name": result_a.test_name,
            "agent_a": {
                "passed": result_a.passed,
                "quality_score": result_a.quality_score,
                "api_time": result_a.api_time_seconds,
                "faithfulness": result_a.faithfulness,
                "context_recall": result_a.context_recall,
                "medication_precision": (
                    result_a.medication_eval.name_precision 
                    if result_a.medication_eval else None
                ),
                "medication_recall": (
                    result_a.medication_eval.name_recall 
                    if result_a.medication_eval else None
                ),
                "hallucinations": (
                    result_a.medication_eval.hallucinations 
                    if result_a.medication_eval else []
                ),
                "missing_medications": (
                    result_a.medication_eval.missing_medications 
                    if result_a.medication_eval else []
                ),
                "failure_reasons": result_a.failure_reasons,
            },
            "agent_b": {
                "passed": result_b.passed,
                "quality_score": result_b.quality_score,
                "api_time": result_b.api_time_seconds,
                "faithfulness": result_b.faithfulness,
                "context_recall": result_b.context_recall,
                "medication_precision": (
                    result_b.medication_eval.name_precision 
                    if result_b.medication_eval else None
                ),
                "medication_recall": (
                    result_b.medication_eval.name_recall 
                    if result_b.medication_eval else None
                ),
                "hallucinations": (
                    result_b.medication_eval.hallucinations 
                    if result_b.medication_eval else []
                ),
                "missing_medications": (
                    result_b.medication_eval.missing_medications 
                    if result_b.medication_eval else []
                ),
                "failure_reasons": result_b.failure_reasons,
            },
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
        
        return comparison
