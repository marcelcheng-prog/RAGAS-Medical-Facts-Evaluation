"""
Console reporter using Rich library for beautiful terminal output.
"""

from typing import Optional, Any

from ..models.evaluation import EvaluationResult, MedicationEvaluation
from ..config.thresholds import QualityThresholds

# Check if Rich is available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ConsoleReporter:
    """Rich console reporter for evaluation results."""
    
    def __init__(
        self,
        verbose: bool = False,
        thresholds: Optional[QualityThresholds] = None,
    ):
        """
        Initialize console reporter.
        
        Args:
            verbose: Whether to show detailed output
            thresholds: Quality thresholds for pass/fail indicators
        """
        self.verbose = verbose
        self.thresholds = thresholds
        
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def _status(self, score: float, threshold: float) -> str:
        """Get status indicator for score vs threshold."""
        return "✅" if score >= threshold else "❌"
    
    def print_header(self, title: str, subtitle: str = "") -> None:
        """Print a header panel."""
        if self.console:
            content = f"[bold cyan]{title}[/bold cyan]"
            if subtitle:
                content += f"\n[dim]{subtitle}[/dim]"
            self.console.print(Panel.fit(content, border_style="cyan"))
        else:
            print(f"\n{'='*60}")
            print(title)
            if subtitle:
                print(subtitle)
            print('='*60)
    
    def print_status(self, message: str, status: str = "info") -> None:
        """Print a status message."""
        colors = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        color = colors.get(status, "white")
        
        if self.console:
            self.console.print(f"[{color}]{message}[/{color}]")
        else:
            print(message)
    
    def print_medication_table(
        self,
        med_eval: MedicationEvaluation,
        thresholds: Optional[QualityThresholds] = None,
    ) -> None:
        """Print medication evaluation results table."""
        thresholds = thresholds or self.thresholds
        
        if self.console and thresholds:
            table = Table(title="💊 Medication Extraction", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Threshold", style="yellow")
            table.add_column("Status", style="white")
            
            table.add_row(
                "Precision",
                f"{med_eval.name_precision:.1%}",
                f"{thresholds.medication_precision:.1%}",
                self._status(med_eval.name_precision, thresholds.medication_precision)
            )
            table.add_row(
                "Recall",
                f"{med_eval.name_recall:.1%}",
                f"{thresholds.medication_recall:.1%}",
                self._status(med_eval.name_recall, thresholds.medication_recall)
            )
            table.add_row(
                "F1 Score",
                f"{med_eval.f1_score:.1%}",
                "-",
                ""
            )
            table.add_row(
                "Dose Accuracy",
                f"{med_eval.dose_accuracy:.1%}",
                "-",
                ""
            )
            table.add_row(
                "Action Classification",
                f"{med_eval.action_accuracy:.1%}",
                f"{thresholds.action_classification:.1%}",
                self._status(med_eval.action_accuracy, thresholds.action_classification)
            )
            
            self.console.print(table)
            
            # Print issues
            if med_eval.hallucinations:
                self.console.print(
                    f"\n[red bold]🚨 HALLUCINATIONS: {', '.join(med_eval.hallucinations)}[/red bold]"
                )
            if med_eval.missing_medications:
                self.console.print(
                    f"[yellow]⚠️  Missing: {', '.join(med_eval.missing_medications)}[/yellow]"
                )
            if med_eval.null_doses:
                self.console.print(
                    f"[yellow]⚠️  Null doses: {', '.join(med_eval.null_doses)}[/yellow]"
                )
        else:
            print("\n💊 Medication Extraction")
            print(f"  Precision: {med_eval.name_precision:.1%}")
            print(f"  Recall: {med_eval.name_recall:.1%}")
            print(f"  F1 Score: {med_eval.f1_score:.1%}")
    
    def print_ragas_table(
        self,
        faithfulness: Optional[float],
        context_recall: Optional[float],
        answer_relevancy: Optional[float],
        thresholds: Optional[QualityThresholds] = None,
    ) -> None:
        """Print RAGAS evaluation results table."""
        thresholds = thresholds or self.thresholds
        
        if self.console and thresholds:
            table = Table(title="📊 RAGAS Scores", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Threshold", style="yellow")
            table.add_column("Status", style="white")
            
            if faithfulness is not None:
                table.add_row(
                    "Faithfulness",
                    f"{faithfulness:.1%}",
                    f"{thresholds.faithfulness:.1%}",
                    self._status(faithfulness, thresholds.faithfulness)
                )
            else:
                table.add_row("Faithfulness", "Failed", "-", "⚠️")
            
            if context_recall is not None:
                table.add_row(
                    "Context Recall",
                    f"{context_recall:.1%}",
                    f"{thresholds.context_recall:.1%}",
                    self._status(context_recall, thresholds.context_recall)
                )
            else:
                table.add_row("Context Recall", "Failed", "-", "⚠️")
            
            if answer_relevancy is not None:
                table.add_row(
                    "Answer Relevancy",
                    f"{answer_relevancy:.1%}",
                    f"{thresholds.answer_relevancy:.1%}",
                    self._status(answer_relevancy, thresholds.answer_relevancy)
                )
            else:
                table.add_row("Answer Relevancy", "Failed (embeddings)", "-", "⚠️")
            
            self.console.print(table)
        else:
            print("\n📊 RAGAS Scores")
            if faithfulness is not None:
                print(f"  Faithfulness: {faithfulness:.1%}")
            if context_recall is not None:
                print(f"  Context Recall: {context_recall:.1%}")
            if answer_relevancy is not None:
                print(f"  Answer Relevancy: {answer_relevancy:.1%}")
    
    def print_verdict(self, result: EvaluationResult) -> None:
        """Print final verdict for evaluation."""
        if self.console:
            if result.passed:
                self.console.print(f"\n[bold green]✅ AGENT PASSED ALL CHECKS[/bold green]")
                self.console.print(f"[green]Quality Score: {result.quality_score:.1f}/100[/green]")
            else:
                self.console.print(f"\n[bold red]❌ AGENT FAILED[/bold red]")
                self.console.print(f"[red]Quality Score: {result.quality_score:.1f}/100[/red]")
                
                if result.critical_hallucinations:
                    self.console.print(f"\n[bold red]🚨 CRITICAL SAFETY ISSUES:[/bold red]")
                    for issue in result.critical_hallucinations:
                        self.console.print(f"  [red]• {issue}[/red]")
                
                self.console.print(f"\n[bold]Failure Reasons:[/bold]")
                for reason in result.failure_reasons:
                    self.console.print(f"  [yellow]• {reason}[/yellow]")
                
                if result.warnings:
                    self.console.print(f"\n[bold]Warnings:[/bold]")
                    for warning in result.warnings:
                        self.console.print(f"  [dim]• {warning}[/dim]")
        else:
            status = "PASSED" if result.passed else "FAILED"
            print(f"\n{'✅' if result.passed else '❌'} AGENT {status}")
            print(f"Quality Score: {result.quality_score:.1f}/100")
            if result.failure_reasons:
                print("Failure Reasons:")
                for reason in result.failure_reasons:
                    print(f"  • {reason}")
    
    def print_agent_output(self, output: str, facts: Optional[dict] = None) -> None:
        """Print agent output (for verbose mode)."""
        if not self.verbose:
            return
        
        if self.console:
            self.console.print("\n[bold]📄 Agent Output:[/bold]")
            import json
            content = json.dumps(facts, indent=2, ensure_ascii=False) if facts else output[:500]
            syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print("\n📄 Agent Output:")
            print(output[:500] if len(output) > 500 else output)
    
    def print_comparison_table(
        self,
        agent_a: dict[str, Any],
        agent_b: dict[str, Any],
        agent_a_id: str,
        agent_b_id: str,
    ) -> None:
        """Print comparison table for two agents."""
        if not self.console:
            print("\nComparison not available without Rich library")
            return
        
        def better(a: Any, b: Any, lower_is_better: bool = False) -> str:
            if a is None or b is None:
                return "-"
            if lower_is_better:
                return "A ✅" if a < b else "B ✅" if b < a else "="
            return "A ✅" if a > b else "B ✅" if b > a else "="
        
        table = Table(title="📊 Metric Comparison", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Agent A", style="green")
        table.add_column("Agent B", style="green")
        table.add_column("Better", style="yellow")
        
        table.add_row(
            "Quality Score",
            f"{agent_a['quality_score']:.1f}",
            f"{agent_b['quality_score']:.1f}",
            better(agent_a['quality_score'], agent_b['quality_score'])
        )
        
        table.add_row(
            "Passed",
            "✅" if agent_a['passed'] else "❌",
            "✅" if agent_b['passed'] else "❌",
            "A ✅" if agent_a['passed'] and not agent_b['passed'] 
            else "B ✅" if agent_b['passed'] and not agent_a['passed'] else "="
        )
        
        if agent_a.get('faithfulness') and agent_b.get('faithfulness'):
            table.add_row(
                "Faithfulness",
                f"{agent_a['faithfulness']:.1%}",
                f"{agent_b['faithfulness']:.1%}",
                better(agent_a['faithfulness'], agent_b['faithfulness'])
            )
        
        if agent_a.get('context_recall') and agent_b.get('context_recall'):
            table.add_row(
                "Context Recall",
                f"{agent_a['context_recall']:.1%}",
                f"{agent_b['context_recall']:.1%}",
                better(agent_a['context_recall'], agent_b['context_recall'])
            )
        
        if agent_a.get('medication_precision') and agent_b.get('medication_precision'):
            table.add_row(
                "Med Precision",
                f"{agent_a['medication_precision']:.1%}",
                f"{agent_b['medication_precision']:.1%}",
                better(agent_a['medication_precision'], agent_b['medication_precision'])
            )
        
        if agent_a.get('medication_recall') and agent_b.get('medication_recall'):
            table.add_row(
                "Med Recall",
                f"{agent_a['medication_recall']:.1%}",
                f"{agent_b['medication_recall']:.1%}",
                better(agent_a['medication_recall'], agent_b['medication_recall'])
            )
        
        table.add_row(
            "API Time",
            f"{agent_a['api_time']:.1f}s",
            f"{agent_b['api_time']:.1f}s",
            better(agent_a['api_time'], agent_b['api_time'], lower_is_better=True)
        )
        
        table.add_row(
            "Hallucinations",
            str(len(agent_a.get('hallucinations', []))),
            str(len(agent_b.get('hallucinations', []))),
            better(
                len(agent_a.get('hallucinations', [])),
                len(agent_b.get('hallucinations', [])),
                lower_is_better=True
            )
        )
        
        self.console.print(table)
    
    def print_consistency_table(
        self,
        results: list[EvaluationResult],
    ) -> None:
        """Print consistency analysis table."""
        import pandas as pd
        
        quality_scores = [r.quality_score for r in results]
        passed_count = sum(1 for r in results if r.passed)
        iterations = len(results)
        
        if self.console:
            table = Table(title="Consistency Metrics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Iterations", str(iterations))
            table.add_row("Passed", f"{passed_count}/{iterations} ({passed_count/iterations:.1%})")
            table.add_row("Avg Quality Score", f"{sum(quality_scores)/len(quality_scores):.1f}")
            table.add_row("Min Quality Score", f"{min(quality_scores):.1f}")
            table.add_row("Max Quality Score", f"{max(quality_scores):.1f}")
            table.add_row("Std Dev", f"{pd.Series(quality_scores).std():.2f}")
            
            self.console.print(table)
            
            if pd.Series(quality_scores).std() > 10:
                self.console.print(
                    "\n[yellow]⚠️  High variance detected - agent produces inconsistent results[/yellow]"
                )
            else:
                self.console.print("\n[green]✅ Agent produces consistent results[/green]")
        else:
            print("\nConsistency Metrics:")
            print(f"  Iterations: {iterations}")
            print(f"  Passed: {passed_count}/{iterations}")
            print(f"  Avg Quality: {sum(quality_scores)/len(quality_scores):.1f}")
