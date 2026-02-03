"""
Ground truth data models for Medical Facts.

These dataclasses represent the expected (hand-labeled) data
for test cases.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Medication:
    """Medication with full context."""
    
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    action: Optional[str] = None  # e.g., "new", "stopped", "changed", "refused"
    reason: Optional[str] = None
    indication: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in {
            "name": self.name,
            "dose": self.dose,
            "frequency": self.frequency,
            "action": self.action,
            "reason": self.reason,
            "indication": self.indication,
            "notes": self.notes,
        }.items() if v is not None}


@dataclass
class VitalMeasurement:
    """A single vital sign measurement."""
    
    parameter: str  # e.g., "Blutdruck", "Herzfrequenz"
    value: str      # e.g., "160/90"
    unit: str = ""  # e.g., "mmHg"
    source: str = "doctor_measured"  # or "patient_reported"
    
    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
        }


@dataclass
class GroundTruth:
    """Hand-labeled ground truth for Medical Facts test case."""
    
    # Medications currently taken by patient
    medications_taken: list[Medication] = field(default_factory=list)
    
    # Medications discussed/planned (new, changed, stopped, refused)
    medications_planned: list[Medication] = field(default_factory=list)
    
    # All medication names mentioned in transcript
    all_medication_names: list[str] = field(default_factory=list)
    
    # Vital sign measurements
    vital_measurements: list[VitalMeasurement] = field(default_factory=list)
    
    # Patient-reported symptoms
    symptoms: list[str] = field(default_factory=list)
    
    # Relevant medical history
    medical_history: list[str] = field(default_factory=list)
    
    # Planned diagnostic procedures
    diagnostic_plans: list[str] = field(default_factory=list)
    
    # Therapeutic interventions discussed
    therapeutic_interventions: list[str] = field(default_factory=list)
    
    # Medications that should NEVER appear (for hallucination detection)
    forbidden_medications: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "medications_taken": [m.to_dict() for m in self.medications_taken],
            "medications_planned": [m.to_dict() for m in self.medications_planned],
            "all_medication_names": self.all_medication_names,
            "vital_measurements": [v.to_dict() for v in self.vital_measurements],
            "symptoms": self.symptoms,
            "medical_history": self.medical_history,
            "diagnostic_plans": self.diagnostic_plans,
            "therapeutic_interventions": self.therapeutic_interventions,
            "forbidden_medications": self.forbidden_medications,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        """Create GroundTruth from dictionary."""
        return cls(
            medications_taken=[
                Medication(**m) for m in data.get("medications_taken", [])
            ],
            medications_planned=[
                Medication(**m) for m in data.get("medications_planned", [])
            ],
            all_medication_names=data.get("all_medication_names", []),
            vital_measurements=[
                VitalMeasurement(**v) if isinstance(v, dict) else VitalMeasurement(
                    parameter=v.get("parameter", ""),
                    value=v.get("value", ""),
                    unit=v.get("unit", ""),
                    source=v.get("source", "doctor_measured"),
                )
                for v in data.get("vital_measurements", [])
            ],
            symptoms=data.get("symptoms", []),
            medical_history=data.get("medical_history", []),
            diagnostic_plans=data.get("diagnostic_plans", []),
            therapeutic_interventions=data.get("therapeutic_interventions", []),
            forbidden_medications=data.get("forbidden_medications", []),
        )
