"""SOAP evaluator for agent comparison and verbose diagnostics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from ..clients.ragas_client import RagasEvaluator

from ..clients.medical_facts import parse_medical_facts
from ..formatters.soap_formatter import build_soap_sections

SOAP_SECTIONS = ("S", "O", "A", "P")

SECTION_ALIASES = {
    "S": ["S", "subjective", "subjektiv"],
    "O": ["O", "objective", "objektiv"],
    "A": ["A", "assessment", "beurteilung"],
    "P": ["P", "plan", "planung"],
}


@dataclass
class SoapEvaluationResult:
    test_id: str
    test_name: str
    agent_id: str
    api_time_seconds: float
    passed: bool
    structure_score: float
    section_scores: Dict[str, float]
    semantic_section_scores: Optional[Dict[str, float]]
    ragas_section_scores: Optional[Dict[str, float]]
    effective_section_scores: Dict[str, float]
    average_content_score: float
    lexical_average_content_score: float
    semantic_average_content_score: Optional[float]
    ragas_average_content_score: Optional[float]
    overall_score: float
    missing_sections: List[str]
    warnings: List[str]
    failure_reasons: List[str]
    raw_output: str
    soap_sections: Dict[str, List[str]]
    gold_sections: Dict[str, List[str]]

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "agent_id": self.agent_id,
            "api_time_seconds": self.api_time_seconds,
            "passed": self.passed,
            "structure_score": self.structure_score,
            "section_scores": self.section_scores,
            "semantic_section_scores": self.semantic_section_scores,
            "ragas_section_scores": self.ragas_section_scores,
            "effective_section_scores": self.effective_section_scores,
            "average_content_score": self.average_content_score,
            "lexical_average_content_score": self.lexical_average_content_score,
            "semantic_average_content_score": self.semantic_average_content_score,
            "ragas_average_content_score": self.ragas_average_content_score,
            "overall_score": self.overall_score,
            "missing_sections": self.missing_sections,
            "warnings": self.warnings,
            "failure_reasons": self.failure_reasons,
            "raw_output": self.raw_output,
            "soap_sections": self.soap_sections,
            "gold_sections": self.gold_sections,
        }


def _normalize_sections(sections: dict) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {"S": [], "O": [], "A": [], "P": []}
    for key in SOAP_SECTIONS:
        values = sections.get(key, []) if isinstance(sections, dict) else []
        if isinstance(values, list):
            normalized[key] = [str(v).strip() for v in values if str(v).strip()]
        elif isinstance(values, str) and values.strip():
            normalized[key] = [values.strip()]
    return normalized


def _extract_sections_from_aliases(data: dict) -> Dict[str, List[str]]:
    mapped: Dict[str, List[str]] = {"S": [], "O": [], "A": [], "P": []}
    if not isinstance(data, dict):
        return mapped

    lower_keys = {str(k).lower(): k for k in data.keys()}
    for sec, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            key = lower_keys.get(alias.lower())
            if key is not None:
                value = data.get(key)
                if isinstance(value, list):
                    mapped[sec] = [str(v).strip() for v in value if str(v).strip()]
                elif isinstance(value, str) and value.strip():
                    mapped[sec] = [value.strip()]
                break
    return mapped


def parse_soap_text_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"S": [], "O": [], "A": [], "P": []}
    if not text:
        return sections

    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading = re.match(r"^([SOAP])\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if heading:
            current = heading.group(1).upper()
            tail = heading.group(2).strip()
            if tail:
                sections[current].append(tail.lstrip("- ").strip())
            continue

        if current is None:
            continue

        if line.startswith("-"):
            line = line[1:].strip()
        sections[current].append(line)

    return _normalize_sections(sections)


def extract_soap_sections(output: str) -> Dict[str, List[str]]:
    """Extract SOAP sections from JSON SOAP, JSON medical facts, or plain SOAP text."""
    # Try strict JSON parse first.
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            if "soap" in data and isinstance(data["soap"], dict):
                return _normalize_sections(data["soap"])
            if all(k in data for k in SOAP_SECTIONS):
                return _normalize_sections(data)
            alias_mapped = _extract_sections_from_aliases(data)
            if any(alias_mapped.values()):
                return _normalize_sections(alias_mapped)
            return _normalize_sections(build_soap_sections(data))
    except Exception:
        pass

    # Try medical-facts parser cleanup path.
    parsed, err = parse_medical_facts(output)
    if parsed is not None and err is None:
        if "soap" in parsed and isinstance(parsed["soap"], dict):
            return _normalize_sections(parsed["soap"])
        if all(k in parsed for k in SOAP_SECTIONS):
            return _normalize_sections(parsed)
        alias_mapped = _extract_sections_from_aliases(parsed)
        if any(alias_mapped.values()):
            return _normalize_sections(alias_mapped)
        return _normalize_sections(build_soap_sections(parsed))

    # Fallback to plain text SOAP parse.
    return parse_soap_text_sections(output)


def load_gold_sections(test_case_path: Path, gold_soap_dir: Path) -> Dict[str, List[str]]:
    base = test_case_path.stem
    candidate = gold_soap_dir / f"{base}.soap.json"

    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        soap = data.get("soap", {}) if isinstance(data, dict) else {}
        return _normalize_sections(soap)

    # Fallback to test-case ground_truth directly.
    with open(test_case_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and isinstance(raw.get("ground_truth"), dict):
        return _normalize_sections(build_soap_sections(raw["ground_truth"]))

    return _normalize_sections(build_soap_sections(raw if isinstance(raw, dict) else {}))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _token_f1(pred_items: List[str], gold_items: List[str]) -> float:
    pred_tokens = _tokenize(" ".join(pred_items))
    gold_tokens = _tokenize(" ".join(gold_items))

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    overlap = len(pred_set & gold_set)

    precision = overlap / max(len(pred_set), 1)
    recall = overlap / max(len(gold_set), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _llm_semantic_section_score(
    *,
    client: OpenAI,
    model: str,
    section: str,
    predicted_items: List[str],
    gold_items: List[str],
) -> float:
    """Score semantic equivalence between predicted and gold section entries with GPT."""
    # Fast-path
    if not predicted_items and not gold_items:
        return 1.0
    if not predicted_items or not gold_items:
        return 0.0

    system_prompt = (
        "You are a strict clinical evaluator for German SOAP notes. "
        "Return ONLY JSON with field 'score' in [0,1]. "
        "Score semantic equivalence between two item lists for one SOAP section. "
        "Count medically equivalent phrasing as match, including frequency equivalents "
        "like 'morgens' == '1-0-0', 'abends' == '0-0-1', '2x täglich' == '1-0-1'. "
        "Ignore wording style and ordering. Penalize missing/extra critical facts."
    )

    user_prompt = (
        f"Section: {section}\n"
        f"Predicted items:\n- " + "\n- ".join(predicted_items) + "\n\n"
        f"Gold items:\n- " + "\n- ".join(gold_items) + "\n\n"
        "Return JSON: {\"score\": <float 0..1>}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)
    raw_score = float(data.get("score", 0.0))
    if raw_score < 0:
        return 0.0
    if raw_score > 1:
        return 1.0
    return raw_score


def _combine_section_score(
    lexical: float,
    gpt_semantic: Optional[float],
    ragas_semantic: Optional[float],
) -> float:
    """Blend lexical + semantic signals, preventing semantic-only perfect jumps."""
    parts = [(lexical, 0.35)]
    if gpt_semantic is not None:
        parts.append((gpt_semantic, 0.30))
    if ragas_semantic is not None:
        parts.append((ragas_semantic, 0.35))

    total_w = sum(w for _, w in parts)
    if total_w <= 0:
        return lexical
    blended = sum(v * w for v, w in parts) / total_w
    # Keep score in bounds and ensure lexical signal is never fully ignored.
    blended = max(blended, lexical * 0.8)
    return min(max(blended, 0.0), 1.0)


def evaluate_soap_output(
    *,
    output: str,
    gold_sections: Dict[str, List[str]],
    test_id: str,
    test_name: str,
    agent_id: str,
    api_time_seconds: float,
    structure_threshold: float = 1.0,
    content_threshold: float = 0.70,
    openai_client: Optional[OpenAI] = None,
    openai_model: Optional[str] = None,
    ragas_evaluator: Optional[RagasEvaluator] = None,
    transcript: Optional[str] = None,
) -> SoapEvaluationResult:
    predicted = extract_soap_sections(output)

    present = [s for s in SOAP_SECTIONS if predicted.get(s)]
    missing = [s for s in SOAP_SECTIONS if s not in present]
    structure_score = len(present) / 4.0

    section_scores = {
        s: _token_f1(predicted.get(s, []), gold_sections.get(s, []))
        for s in SOAP_SECTIONS
    }
    lexical_avg_content = sum(section_scores.values()) / 4.0

    semantic_section_scores: Optional[Dict[str, float]] = None
    semantic_avg_content: Optional[float] = None
    ragas_section_scores: Optional[Dict[str, float]] = None
    ragas_avg_content: Optional[float] = None
    effective_section_scores = dict(section_scores)

    if openai_client is not None and openai_model:
        semantic_section_scores = {}
        for sec in SOAP_SECTIONS:
            try:
                semantic_section_scores[sec] = _llm_semantic_section_score(
                    client=openai_client,
                    model=openai_model,
                    section=sec,
                    predicted_items=predicted.get(sec, []),
                    gold_items=gold_sections.get(sec, []),
                )
            except Exception:
                semantic_section_scores[sec] = section_scores[sec]

        semantic_avg_content = sum(semantic_section_scores.values()) / 4.0
        # temporary combine lexical + GPT (RAGAS may further update below)
        effective_section_scores = {
            s: _combine_section_score(section_scores[s], semantic_section_scores[s], None)
            for s in SOAP_SECTIONS
        }

    if ragas_evaluator is not None and transcript:
        ragas_section_scores = {}
        for sec in SOAP_SECTIONS:
            ragas_score = ragas_evaluator.evaluate_semantic_list_recall(
                transcript=transcript,
                category_name=f"SOAP {sec}",
                predicted_items=predicted.get(sec, []),
                gold_items=gold_sections.get(sec, []),
            )
            if ragas_score is None:
                ragas_score = effective_section_scores[sec]
            ragas_section_scores[sec] = ragas_score

        ragas_avg_content = sum(ragas_section_scores.values()) / 4.0
        effective_section_scores = {
            s: _combine_section_score(
                section_scores[s],
                semantic_section_scores[s] if semantic_section_scores else None,
                ragas_section_scores[s],
            )
            for s in SOAP_SECTIONS
        }

    avg_content = sum(effective_section_scores.values()) / 4.0
    overall = 0.30 * structure_score + 0.70 * avg_content

    failure_reasons: List[str] = []
    warnings: List[str] = []

    if structure_score < structure_threshold:
        failure_reasons.append(
            f"SOAP structure: {structure_score:.1%} < {structure_threshold:.1%}"
        )
    if avg_content < content_threshold:
        failure_reasons.append(
            f"SOAP content: {avg_content:.1%} < {content_threshold:.1%}"
        )
    if missing:
        warnings.append("Missing sections: " + ", ".join(missing))

    passed = len(failure_reasons) == 0

    return SoapEvaluationResult(
        test_id=test_id,
        test_name=test_name,
        agent_id=agent_id,
        api_time_seconds=api_time_seconds,
        passed=passed,
        structure_score=structure_score,
        section_scores=section_scores,
        semantic_section_scores=semantic_section_scores,
        ragas_section_scores=ragas_section_scores,
        effective_section_scores=effective_section_scores,
        average_content_score=avg_content,
        lexical_average_content_score=lexical_avg_content,
        semantic_average_content_score=semantic_avg_content,
        ragas_average_content_score=ragas_avg_content,
        overall_score=overall * 100.0,
        missing_sections=missing,
        warnings=warnings,
        failure_reasons=failure_reasons,
        raw_output=output,
        soap_sections=predicted,
        gold_sections=gold_sections,
    )
