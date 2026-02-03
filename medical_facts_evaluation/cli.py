"""
Command-line interface for Medical Facts Evaluation.
"""

import sys
import argparse
from pathlib import Path

from .config.settings import get_settings
from .config.thresholds import PRODUCTION, DEVELOPMENT
from .models.loader import load_test_case, load_all_test_cases, get_default_test_case
from .evaluator import MedicalFactsEvaluator, run_agent_comparison
from .reporters.json_reporter import JsonReporter


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Production Medical Facts Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show help message and options
    python -m medical_facts_evaluation --help

    # Run evaluation with default agent
    python -m medical_facts_evaluation

    # Run with verbose output
    python -m medical_facts_evaluation --verbose

    # Run consistency check (5 iterations)
    python -m medical_facts_evaluation --iterations 5

    # Compare two agents
    python -m medical_facts_evaluation --compare --agent-a <id1> --agent-b <id2>

    # Use specific test case
    python -m medical_facts_evaluation --test-case test_cases/michael_mueller.json
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=1,
        help="Number of iterations for consistency check (default: 1)"
    )
    
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare two agents side-by-side"
    )
    
    parser.add_argument(
        "--agent-a",
        type=str,
        default=None,
        help="First agent ID (default: from settings)"
    )
    
    parser.add_argument(
        "--agent-b",
        type=str,
        help="Second agent ID for comparison (required with --compare)"
    )
    
    parser.add_argument(
        "--test-case",
        type=str,
        help="Path to test case JSON file (default: built-in Michael test)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: from settings)"
    )
    
    parser.add_argument(
        "--thresholds",
        type=str,
        choices=["production", "development"],
        default="production",
        help="Threshold profile to use (default: production)"
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate comparison mode
    if args.compare and not args.agent_b:
        parser.error("--compare requires --agent-b to be specified")
    
    # Load settings
    settings = get_settings()
    
    # Validate settings
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"❌ ERROR: {error}")
        print("\n💡 TIP: Create a .env file with your API keys")
        print("   Copy .env.example to .env and fill in your values")
        return 1
    
    # Set agent ID
    agent_a_id = args.agent_a or settings.default_agent_id
    
    # Set thresholds
    thresholds = PRODUCTION if args.thresholds == "production" else DEVELOPMENT
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else settings.results_dir
    
    # Load test case
    if args.test_case:
        test_case = load_test_case(args.test_case)
    else:
        test_case = get_default_test_case()
    
    print(f"📋 Test: {test_case.name}")
    print(f"   Expected medications: {len(test_case.ground_truth.all_medication_names)}")
    print(f"   ({', '.join(test_case.ground_truth.all_medication_names)})")
    
    # Run comparison or single evaluation
    if args.compare:
        result_a, result_b, comparison = run_agent_comparison(
            agent_a_id=agent_a_id,
            agent_b_id=args.agent_b,
            test_case=test_case,
            settings=settings,
            thresholds=thresholds,
            output_dir=output_dir,
            verbose=args.verbose,
        )
        
        # Return exit code based on winner
        if comparison["winner"] == "tie":
            return 0
        return 0 if comparison["winner"] == "agent_a" else 1
    
    else:
        # Create evaluator
        evaluator = MedicalFactsEvaluator.from_settings(
            agent_id=agent_a_id,
            settings=settings,
            thresholds=thresholds,
            verbose=args.verbose,
        )
        
        # Run evaluation
        if args.iterations > 1:
            results = evaluator.run_consistency_check(test_case, args.iterations)
        else:
            result = evaluator.evaluate(test_case)
            results = [result]
        
        # Save results
        json_reporter = JsonReporter(output_dir)
        json_reporter.save_results(results)
        
        # Return exit code
        failed_count = sum(1 for r in results if not r.passed)
        return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
