"""
Command-line interface for Medical Facts Evaluation.
"""

import sys
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from .config.settings import get_settings
from .config.thresholds import PRODUCTION, DEVELOPMENT
from .models.loader import load_test_case, get_default_test_case
from .evaluator import MedicalFactsEvaluator, run_agent_comparison
from .reporters.json_reporter import JsonReporter
from .clients.medical_facts import MedicalFactsClient
from .clients.ragas_client import RagasEvaluator
from .evaluators.soap_evaluator import evaluate_soap_output, load_gold_sections


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

    parser.add_argument(
        "--mode",
        type=str,
        choices=["medical_facts", "soap"],
        default="medical_facts",
        help="Evaluation mode (default: medical_facts)"
    )

    parser.add_argument(
        "--verbose-soap",
        action="store_true",
        help="Show detailed SOAP output and section scores"
    )

    parser.add_argument(
        "--all-test-cases",
        action="store_true",
        help="Run on all test cases in --test-cases-dir"
    )

    parser.add_argument(
        "--test-cases-dir",
        type=str,
        default=None,
        help="Directory with test case JSON files (default: package test_cases/)"
    )

    parser.add_argument(
        "--gold-soap-dir",
        type=str,
        default="medical_facts_evaluation/gold_soap",
        help="Directory with gold SOAP JSON files"
    )

    parser.add_argument(
        "--soap-structure-threshold",
        type=float,
        default=1.0,
        help="SOAP structure threshold (default: 1.0)"
    )

    parser.add_argument(
        "--soap-content-threshold",
        type=float,
        default=0.70,
        help="SOAP content threshold (default: 0.70)"
    )
    
    return parser


def _resolve_test_case_paths(args) -> list[Path]:
    package_test_dir = Path(__file__).parent / "test_cases"
    test_cases_dir = Path(args.test_cases_dir) if args.test_cases_dir else package_test_dir

    if args.all_test_cases:
        paths = [p for p in sorted(test_cases_dir.glob("*.json")) if p.name != "schema.json"]
        if not paths:
            raise FileNotFoundError(f"No test case JSON files found in {test_cases_dir}")
        return paths

    if args.test_case:
        return [Path(args.test_case)]

    default_path = package_test_dir / "michael_mueller.json"
    if default_path.exists():
        return [default_path]

    # Fallback to loader default behavior
    default_case = get_default_test_case()
    fallback = Path(f"medical_facts_evaluation/test_cases/{default_case.test_id}.json")
    return [fallback]


def _print_soap_result(result: dict, verbose: bool = False) -> None:
    status = "PASSED" if result["passed"] else "FAILED"
    print(f"\n🧼 SOAP {status} - {result['test_name']}")
    print(f"   Overall: {result['overall_score']:.1f}/100")
    print(f"   Structure: {result['structure_score']:.1%}")
    print(f"   Content: {result['average_content_score']:.1%}")
    print(f"   Lexical content: {result.get('lexical_average_content_score', 0.0):.1%}")
    semantic_avg = result.get("semantic_average_content_score")
    if semantic_avg is not None:
        print(f"   Semantic content (GPT): {semantic_avg:.1%}")
    ragas_avg = result.get("ragas_average_content_score")
    if ragas_avg is not None:
        print(f"   Semantic content (RAGAS): {ragas_avg:.1%}")
    print(
        "   Sections: "
        f"S={result['section_scores']['S']:.1%}, "
        f"O={result['section_scores']['O']:.1%}, "
        f"A={result['section_scores']['A']:.1%}, "
        f"P={result['section_scores']['P']:.1%}"
    )
    effective = result.get("effective_section_scores")
    if effective:
        print(
            "   Effective sections: "
            f"S={effective['S']:.1%}, "
            f"O={effective['O']:.1%}, "
            f"A={effective['A']:.1%}, "
            f"P={effective['P']:.1%}"
        )
    if result["failure_reasons"]:
        print("   Failures: " + "; ".join(result["failure_reasons"]))
    if result["warnings"]:
        print("   Warnings: " + "; ".join(result["warnings"]))

    # Weakness summary (always visible)
    effective = result.get("effective_section_scores") or result.get("section_scores", {})
    weak_sorted = sorted(effective.items(), key=lambda x: x[1])
    weak_labels = []
    for sec, score in weak_sorted:
        if score < 0.6:
            weak_labels.append(f"{sec} ({score:.1%})")
    if weak_labels:
        print("   Main weaknesses: " + ", ".join(weak_labels))
    lexical = result.get("section_scores", {})
    lexical_weak = []
    for sec in ("S", "O", "A", "P"):
        score = lexical.get(sec, 0.0)
        if score < 0.6:
            lexical_weak.append(f"{sec} ({score:.1%})")
    if lexical_weak:
        print("   Lexical weaknesses: " + ", ".join(lexical_weak))

    if verbose:
        def _norm_item(text: str) -> str:
            t = (text or "").strip().lower()
            t = re.sub(r"\s+", " ", t)
            t = re.sub(r"[^\w\s/%.,-]", "", t)
            return t

        def _section_delta(pred_items: list[str], gold_items: list[str]) -> tuple[list[str], list[str], list[str]]:
            pred_map = {_norm_item(x): x for x in pred_items if _norm_item(x)}
            gold_map = {_norm_item(x): x for x in gold_items if _norm_item(x)}
            overlap_keys = [k for k in gold_map if k in pred_map]
            missing_keys = [k for k in gold_map if k not in pred_map]
            extra_keys = [k for k in pred_map if k not in gold_map]
            overlap = [gold_map[k] for k in overlap_keys]
            missing = [gold_map[k] for k in missing_keys]
            extra = [pred_map[k] for k in extra_keys]
            return overlap, missing, extra

        print("\n   Section Analysis (Predicted vs Gold):")
        for sec in ("S", "O", "A", "P"):
            pred_items = result.get("soap_sections", {}).get(sec, []) or []
            gold_items = result.get("gold_sections", {}).get(sec, []) or []
            overlap, missing, extra = _section_delta(pred_items, gold_items)

            sec_score = result.get("section_scores", {}).get(sec, 0.0)
            eff_score = (result.get("effective_section_scores", {}) or {}).get(sec, sec_score)

            print(f"\n   {sec} | lexical={sec_score:.1%} effective={eff_score:.1%}")
            print("   Predicted:")
            if pred_items:
                for item in pred_items:
                    print(f"     - {item}")
            else:
                print("     - <empty>")

            print("   Gold:")
            if gold_items:
                for item in gold_items:
                    print(f"     - {item}")
            else:
                print("     - <empty>")

            print("   Missing from prediction:")
            if missing:
                for item in missing:
                    print(f"     - {item}")
            else:
                print("     - <none>")

            print("   Extra in prediction:")
            if extra:
                for item in extra:
                    print(f"     - {item}")
            else:
                print("     - <none>")

            print(f"   Overlap count: {len(overlap)} / gold {len(gold_items)}")

        # Actionable fixes summary
        print("\n   Top Actionable Fixes:")
        fixes = []
        for sec in ("S", "O", "A", "P"):
            pred_items = result.get("soap_sections", {}).get(sec, []) or []
            gold_items = result.get("gold_sections", {}).get(sec, []) or []
            overlap, missing, extra = _section_delta(pred_items, gold_items)

            if not pred_items and gold_items:
                fixes.append(f"{sec}: Section is empty - force extraction for {len(gold_items)} expected items.")
            elif len(overlap) == 0 and gold_items:
                fixes.append(f"{sec}: No direct overlap - align section wording with gold schema patterns.")

            if len(missing) >= 2:
                fixes.append(f"{sec}: Missing {len(missing)} key items - prioritize these facts in prompt/examples.")
            if len(extra) >= 3:
                fixes.append(f"{sec}: Too many extras ({len(extra)}) - reduce non-essential or duplicated details.")

        if not fixes:
            print("     - No major structural/content weaknesses detected.")
        else:
            for fx in fixes[:5]:
                print(f"     - {fx}")


def _save_json_report(output_dir: Path, prefix: str, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{prefix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {path}")
    return path


def _make_soap_run_dir(output_dir: Path, agent_a_id: str, agent_b_id: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / "soap_agents"
    if agent_b_id:
        run_name = f"{ts}_compare_{agent_a_id[:8]}_vs_{agent_b_id[:8]}"
    else:
        run_name = f"{ts}_{agent_a_id[:8]}"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_per_test_result(run_dir: Path, case_path: Path, agent_id: str, result: dict) -> Path:
    agent_dir = run_dir / agent_id[:8]
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / f"{case_path.stem}.soap_eval.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def _run_soap_mode(args, settings, output_dir: Path, agent_a_id: str) -> int:
    test_case_paths = _resolve_test_case_paths(args)
    gold_soap_dir = Path(args.gold_soap_dir)
    verbose_soap = args.verbose or args.verbose_soap
    run_dir = _make_soap_run_dir(output_dir, agent_a_id, args.agent_b if args.compare else None)
    print(f"\n📁 SOAP run folder: {run_dir}")
    openai_client = OpenAI(api_key=settings.openai_api_key)
    ragas_evaluator = RagasEvaluator.from_settings(openai_client, settings, verbose=False)

    if args.iterations > 1:
        print("⚠️  --iterations is ignored in SOAP mode (single pass per test case).")

    def evaluate_agent_on_case(agent_id: str, case_path: Path) -> dict:
        test_case = load_test_case(case_path)
        print(f"\n📋 SOAP Test: {test_case.name}")

        client = MedicalFactsClient.from_settings(agent_id, settings, verbose=args.verbose)
        response = client.extract_facts(test_case.transcript)

        if response.error:
            failure = {
                "test_id": test_case.test_id,
                "test_name": test_case.name,
                "agent_id": agent_id,
                "api_time_seconds": response.api_time_seconds,
                "passed": False,
                "structure_score": 0.0,
                "section_scores": {"S": 0.0, "O": 0.0, "A": 0.0, "P": 0.0},
                "average_content_score": 0.0,
                "overall_score": 0.0,
                "missing_sections": ["S", "O", "A", "P"],
                "warnings": [],
                "failure_reasons": [f"API Error: {response.error}"],
                "raw_output": response.content,
                "soap_sections": {"S": [], "O": [], "A": [], "P": []},
                "gold_sections": load_gold_sections(case_path, gold_soap_dir),
            }
            _print_soap_result(failure, verbose=verbose_soap)
            _save_per_test_result(run_dir, case_path, agent_id, failure)
            return failure

        eval_result = evaluate_soap_output(
            output=response.content,
            gold_sections=load_gold_sections(case_path, gold_soap_dir),
            test_id=test_case.test_id,
            test_name=test_case.name,
            agent_id=agent_id,
            api_time_seconds=response.api_time_seconds,
            structure_threshold=args.soap_structure_threshold,
            content_threshold=args.soap_content_threshold,
            openai_client=openai_client,
            openai_model=settings.openai_model,
            ragas_evaluator=ragas_evaluator,
            transcript=test_case.transcript,
        )
        result_dict = eval_result.to_dict()
        _print_soap_result(result_dict, verbose=verbose_soap)
        _save_per_test_result(run_dir, case_path, agent_id, result_dict)
        return result_dict

    # Comparison mode
    if args.compare:
        all_rows = []
        score_a = 0.0
        score_b = 0.0

        for case_path in test_case_paths:
            a = evaluate_agent_on_case(agent_a_id, case_path)
            b = evaluate_agent_on_case(args.agent_b, case_path)
            score_a += a["overall_score"]
            score_b += b["overall_score"]

            if a["overall_score"] > b["overall_score"]:
                winner = "agent_a"
            elif b["overall_score"] > a["overall_score"]:
                winner = "agent_b"
            else:
                winner = "tie"

            all_rows.append({
                "test_case": case_path.name,
                "agent_a": a,
                "agent_b": b,
                "winner": winner,
            })

        n = max(len(all_rows), 1)
        avg_a = score_a / n
        avg_b = score_b / n
        if avg_a > avg_b:
            overall_winner = "agent_a"
        elif avg_b > avg_a:
            overall_winner = "agent_b"
        else:
            overall_winner = "tie"

        report = {
            "mode": "soap",
            "timestamp": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "agent_a_id": agent_a_id,
            "agent_b_id": args.agent_b,
            "thresholds": {
                "structure": args.soap_structure_threshold,
                "content": args.soap_content_threshold,
            },
            "summary": {
                "average_score_agent_a": avg_a,
                "average_score_agent_b": avg_b,
                "overall_winner": overall_winner,
                "test_case_count": len(all_rows),
            },
            "results": all_rows,
        }
        _save_json_report(output_dir, "soap_comparison", report)

        if overall_winner == "tie":
            return 0
        return 0 if overall_winner == "agent_a" else 1

    # Single-agent mode
    rows = []
    for case_path in test_case_paths:
        rows.append(evaluate_agent_on_case(agent_a_id, case_path))

    passed = sum(1 for r in rows if r["passed"])
    report = {
        "mode": "soap",
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "agent_id": agent_a_id,
        "thresholds": {
            "structure": args.soap_structure_threshold,
            "content": args.soap_content_threshold,
        },
        "summary": {
            "passed": passed,
            "failed": len(rows) - passed,
            "average_score": (sum(r["overall_score"] for r in rows) / max(len(rows), 1)),
            "test_case_count": len(rows),
        },
        "results": rows,
    }
    _save_json_report(output_dir, "soap_evaluation", report)

    return 0 if passed == len(rows) else 1


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

    # SOAP mode branch
    if args.mode == "soap":
        return _run_soap_mode(args, settings, output_dir, agent_a_id)
    
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
