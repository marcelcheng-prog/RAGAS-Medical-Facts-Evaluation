"""
Test case loader for Medical Facts Evaluation.

Loads test cases from JSON files in the test_cases directory.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .ground_truth import GroundTruth, Medication, VitalMeasurement


@dataclass
class TestCase:
    """A complete test case with transcript and ground truth."""
    
    test_id: str
    name: str
    description: str
    language: str
    transcript: str
    ground_truth: GroundTruth
    
    @property
    def short_name(self) -> str:
        """Short name for display (first 30 chars)."""
        return self.name[:30] + "..." if len(self.name) > 30 else self.name


def load_test_case(path: Path | str) -> TestCase:
    """
    Load a test case from a JSON file.
    
    Args:
        path: Path to the JSON test case file (can be relative to package or absolute)
        
    Returns:
        TestCase object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        KeyError: If required fields are missing
    """
    path = Path(path)
    
    # If path doesn't exist, try relative to package directory
    if not path.exists():
        package_dir = Path(__file__).parent.parent
        package_path = package_dir / path
        if package_path.exists():
            path = package_path
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse ground truth
    gt_data = data["ground_truth"]
    
    # Convert medication dicts to Medication objects
    medications_taken = [
        Medication(**m) for m in gt_data.get("medications_taken", [])
    ]
    medications_planned = [
        Medication(**m) for m in gt_data.get("medications_planned", [])
    ]
    
    # Convert vital measurements
    vital_measurements = []
    for v in gt_data.get("vital_measurements", []):
        if isinstance(v, dict):
            vital_measurements.append(VitalMeasurement(
                parameter=v.get("parameter", ""),
                value=v.get("value", ""),
                unit=v.get("unit", ""),
                source=v.get("source", "doctor_measured"),
            ))
    
    ground_truth = GroundTruth(
        medications_taken=medications_taken,
        medications_planned=medications_planned,
        all_medication_names=gt_data.get("all_medication_names", []),
        vital_measurements=vital_measurements,
        symptoms=gt_data.get("symptoms", []),
        medical_history=gt_data.get("medical_history", []),
        diagnostic_plans=gt_data.get("diagnostic_plans", []),
        therapeutic_interventions=gt_data.get("therapeutic_interventions", []),
        forbidden_medications=gt_data.get("forbidden_medications", []),
    )
    
    return TestCase(
        test_id=data["test_id"],
        name=data["name"],
        description=data.get("description", ""),
        language=data.get("language", "de"),
        transcript=data["transcript"],
        ground_truth=ground_truth,
    )


def load_all_test_cases(directory: Path | str) -> list[TestCase]:
    """
    Load all test cases from a directory.
    
    Args:
        directory: Path to directory containing JSON test case files
        
    Returns:
        List of TestCase objects
    """
    directory = Path(directory)
    test_cases = []
    
    for path in sorted(directory.glob("*.json")):
        # Skip schema file
        if path.name == "schema.json":
            continue
            
        try:
            test_case = load_test_case(path)
            test_cases.append(test_case)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    
    return test_cases


def get_default_test_case() -> TestCase:
    """
    Get the default test case (Michael Müller).
    
    This is a convenience function for quick testing.
    """
    # Get the path relative to this file
    this_dir = Path(__file__).parent
    test_cases_dir = this_dir.parent / "test_cases"
    michael_path = test_cases_dir / "michael_mueller.json"
    
    if michael_path.exists():
        return load_test_case(michael_path)
    
    # Fallback: try relative to current working directory
    fallback_path = Path("test_cases/michael_mueller.json")
    if fallback_path.exists():
        return load_test_case(fallback_path)
    
    raise FileNotFoundError(
        f"Default test case not found at {michael_path} or {fallback_path}"
    )
